#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

#include "Body.h"
#include "Constants.h"

/*
* Initialize Number of Threads
* Current Number of Threads = 128
*/
const int THREAD_NUM = 128;

/* Function that call in order to link between cpp and kernel.cu */
void updateInCUDA(std::vector<Body>& bodies_h, int nBodies, int nThreads);

/* Actual kernel that to be launch */
__global__ void interactAndUpdate(Body* bodies);
__device__ void accumulate(Body* bodies);
#endif
