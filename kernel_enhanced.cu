//Merge atomic and tree into 1 file
#include <cuda_runtime.h>
#include <cmath>
#include "utils.h"
#include "constant.h"
#include "relu.h"

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

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

// Tree Reduction Kernel
__global__ void forwardPassTree(float *input, float *w1, float *b1, float *output, int batch_size) {
    __shared__ float partialSum[128];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int tid = threadIdx.x;

    // Layer 1 Reduction
    float sum = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        sum += input[idx * INPUT_SIZE + i] * w1[i];
    }

    partialSum[tid] = sum;
    __syncthreads();

    // Tree Reduction in Shared Memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partialSum[tid] += partialSum[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = relu(partialSum[0] + b1[blockIdx.x]);
    }
}

void runTreeReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output) {
    dim3 block(128);
    dim3 grid((BATCH_SIZE + block.x - 1) / block.x);

    forwardPassTree<<<grid, block>>>(d_input, d_w1, d_b1, d_output, BATCH_SIZE);
    cudaDeviceSynchronize();
}
