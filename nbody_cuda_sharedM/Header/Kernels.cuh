#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

#include "Body.h"
#include "Constants.h"

/*
* Initialize Number of Threads
* Current Number of Threads = 128
*/
const int THREAD_NUM = 256;

// functions which run the kernels
void updateInCUDA(std::vector<Body>& bodies_h, int nBodies, int nThreads);

// actual kernels
__global__ void interactAndUpdate(Body* bodies);
__device__ void accumulate(Body* bodies);
#endif
