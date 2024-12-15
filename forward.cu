#include "forward.h"
#include <cuda_runtime.h>

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void forwardPassKernel(float* input, float* w1, float* b1, float* output,
                                  int input_size, int hidden_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size) return;

    // Shared memory reduction
    __shared__ float hidden[HIDDEN_LAYER_1];
    for (int h = 0; h < hidden_size; h++) {
        float val = b1[h];
        for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
            val += input[idx * input_size + i] * w1[h * input_size + i];
        }

        // Tree Reduction within the block
        val = warpReduceSum(val);

        // Atomic Add to Hidden Activation
        if (threadIdx.x == 0) {
            hidden[h] = relu(val);
        }
    }

    __syncthreads();

    // Write results back to output
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float out_val = 0.0;
        for (int h = 0; h < hidden_size; h++) {
            out_val += hidden[h] * w1[o * hidden_size + h];
        }
        output[idx * OUTPUT_SIZE + o] = out_val;
    }
}

void forwardPassGPU(float* input, float* w1, float* b1, float* w2, float* b2,
                    float* w3, float* b3, float* output, int batch_size, int input_size) {
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    forwardPassKernel<<<gridDim, blockDim>>>(input, w1, b1, output, input_size, HIDDEN_LAYER_1, batch_size);
    cudaDeviceSynchronize();
}
