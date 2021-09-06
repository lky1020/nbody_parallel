#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "Constants.h"
#include "Body.h"

/*
* Initialize Number of Threads 
*/
const int THREAD_NUM = 256;

// functions which run the kernels
void updateInCUDA(std::vector<Body>& bodies_h, int nBodies, int nThreads);

// kernels that will be launch
__global__ void interactAndUpdate(Body* bodies);
__device__ void interact(Body* bodies);
__device__ void update(Body* bodies);
#endif
