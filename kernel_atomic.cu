#include <cuda_runtime.h>
#include <cmath>
#include "utils.h"

// Atomic Reduction Kernel
__global__ void forwardPassAtomic(float *input, float *w1, float *b1, float *output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    __shared__ float localSum;

    if (threadIdx.x == 0) localSum = 0.0;
    __syncthreads();

    atomicAdd(&localSum, input[idx] * w1[threadIdx.x]);
    __syncthreads();

    if (threadIdx.x == 0) {
        output[blockIdx.x] = relu(localSum + b1[blockIdx.x]);
    }
}

void runAtomicReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output) {
    dim3 block(128);
    dim3 grid((BATCH_SIZE + block.x - 1) / block.x);

    forwardPassAtomic<<<grid, block>>>(d_input, d_w1, d_b1, d_output, BATCH_SIZE);
    cudaDeviceSynchronize();
}
