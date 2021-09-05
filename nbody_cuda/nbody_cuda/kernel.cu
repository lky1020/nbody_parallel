﻿#include "Kernels.cuh"
#include "Util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#include "Constants.h"
#include "Simulation.h"

int main() {
    /*
    * A SIMPLE PROCESS OF N BODY SIMULATION IN SERIAL
    *
    * STEP	DESCRIPTION
    * ---------------------
    *  1.	INITIALIZE THE WINDOW WIDTH AND HEIGHT
    *  2.	RANDOM GENERATE THE BODY BASED ON THE NUMBER OF BODIES	(BY DEFAULT NUM_BODIES = 1024)
    *  3.	SET THE ANGLE, RADIUS, VELOCITY OF THE BODY GENERTATE AND PLACE
    *  4.	PLACE A BODY AT THE CENTER (ACT AS A SUN) BY ASSIGNING A HEAVIEST VALUE OF MASS
    *  5.	A LOOP WITH i =  1023 (MINUS 1 BCOZ OF SUN) IS TAKE PLACE FOR PROCESS 3 AND 4
    *  6.	CREATE AND SET WINDOW AND VIEW
    *  7.	START THE STIMULATION
    *  8.	CHECK THE SFML EVENT (WINDOW EVENT, MOUSE SCROLL AND KEY PRESS)
    *  9.	UPDATE THE BODIES BY CALCULATING THE EFFECTS OF INTERACTION (POSITION, DISTANCE, FORCE AND ACCELERATION) BETWEEN 2 BODIES
    * 10.	UPDATE BODY VELOCITY, POSITION AND ACCELERATION
    * 11.	PRESENT THE BODIES IN WINDOWS
    * ***	ALL THE BODIES EXCEPT SUN (THE BODY AT THE CENTER) HAVING THE EQUAL VALUE OF MASS
    * ***	LIBRARY SFML IS USED TO DISPLAY BODIES
    * ***	TWO MAIN CLASS WHICH HANDLE DEFINES THE BODY (body.cpp) AND CARRY OUT SIMULATION (simulation.cpp)
    */

 
    /*
    * Initialize the windows size and generate random bodies
    */
    Simulation nBody_sim(WIDTH, HEIGHT);

    /*
    * Start Stimulation
    */
    nBody_sim.start();

    return EXIT_SUCCESS;
}

/* CUDA memory allocations and copy memory to the GPU*/
void updateInCUDA(std::vector<Body>& bodies_h, int nBodies, int nThreads) {

/* Declaration for necessary variables needed */
    /* Number of bytes required for bodies*/
    int size;
    /* Number of Blocks */
    int nBlocks;
    /* Buffer for bodies */
    Body* bodies_d;
 
/* Initialization */
    /* Dynamically allocate host memory */ 
    size = sizeof(Body) * nBodies;
    /* Number of Blocks */
    nBlocks = nBodies / nThreads;

/* Start CUDA Memory Allocation and Copy Memory to CPU */
    /* Allocate device Memory */
    cudaMalloc((void**)&bodies_d, size);

    /* Copy Host Memory to Device Memory */
    cudaMemcpy(bodies_d, &bodies_h[0], size, cudaMemcpyHostToDevice);

    /* Launch Kernel */
    interactAndUpdate << < nBlocks, nThreads >> > (bodies_d);

    /* Synchronize */
    /* Forces the program to wait for all previously issued commands in 
    *  all streams on the device to finish before continuing (from the CUDA C Programming Guide).
    *  So when GPU device is executing kernel, the CPU can continue to work on some other commands and 
    *  issue more instructions to the device
    */
    cudaDeviceSynchronize();

    /* Copy results to host */
    /* Straight away overwrite the original vector */
    cudaMemcpy(&bodies_h[0], bodies_d, size, cudaMemcpyDeviceToHost);

    /* Cleanup */
    cudaFree(bodies_d);
}

/* This function calculate the force of the body and update the velocity, 
*  position and acceleration of the body.
*/
__global__ void interactAndUpdate(Body* bodies) {

    /* Call function calculate the effects of an interaction between 2 bodies */
    interact(bodies);

    /* Waits until all threads within the same block has reached the command
    */
    __syncthreads();

    /* Update the Velocity and position, reset the acceleration of the body */
    update(bodies);
}

/* Function calculate the effects of an interaction between 2 bodies */
__device__ void interact(Body* bodies) {

    /* Get the position (Column) */
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < NUM_BODIES - 1; ++i) {
        if (j != i) {
            // vector to store the position difference between the 2 bodies
            vec3 posDiff{};
            // calculation
            posDiff.x = (bodies[i].position().x - bodies[j].position().x) * TO_METERS;
            posDiff.y = (bodies[i].position().y - bodies[j].position().y) * TO_METERS;
            posDiff.z = (bodies[i].position().z - bodies[j].position().z) * TO_METERS;
            // the actual distance is the length of the vector
            auto dist = sqrtf(posDiff.x * posDiff.x + posDiff.y * posDiff.y +
                posDiff.z * posDiff.z);

            // calculate force
            double F = TIME_STEP * (G * bodies[j].mass() * bodies[i].mass()) /
                ((dist * dist + SOFTENING * SOFTENING) * dist);

            // set this body's acceleration
            bodies[j].acceleration().x += F * posDiff.x / bodies[j].mass();
            bodies[j].acceleration().y += F * posDiff.y / bodies[j].mass();
            bodies[j].acceleration().z += F * posDiff.z / bodies[j].mass();
        }
    }
}

/* update body position */
__device__ void update(Body* bodies) {

    /* Get the position (Column) */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // update velocity
    bodies[i].velocity().x += bodies[i].acceleration().x;
    bodies[i].velocity().y += bodies[i].acceleration().y;
    bodies[i].velocity().z += bodies[i].acceleration().z;

    // reset acceleration
    bodies[i].acceleration().x = 0.0;
    bodies[i].acceleration().y = 0.0;
    bodies[i].acceleration().z = 0.0;

    // update position
    bodies[i].position().x += TIME_STEP * bodies[i].velocity().x / TO_METERS;
    bodies[i].position().y += TIME_STEP * bodies[i].velocity().y / TO_METERS;
    bodies[i].position().z += TIME_STEP * bodies[i].velocity().z / TO_METERS;
}


