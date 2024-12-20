#include <cuda_runtime.h>
#include <cmath>
#include "utils.h"
#include "constant.h"
#include <cfloat>

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ void softmax(float *input, float *output, int size) {
    float max_val = -FLT_MAX;
    float sum = 0.0;

    // Find max value for numerical stability
    for (int i = 0; i < size; i++) {
        max_val = max(max_val, input[i]);
    }

    // Compute exponentials and sum
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Atomic Reduction Kernel
__global__ void forwardPassAtomic(float *input, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, float *output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Layer 1: Input to Hidden Layer 1
    __shared__ float hidden1[HIDDEN_LAYER_1];
    if (threadIdx.x < HIDDEN_LAYER_1) hidden1[threadIdx.x] = b1[threadIdx.x];
    __syncthreads();

    for (int i = threadIdx.x; i < INPUT_SIZE; i += blockDim.x) {
        for (int j = 0; j < HIDDEN_LAYER_1; j++) {
            atomicAdd(&hidden1[j], input[idx * INPUT_SIZE + i] * w1[j * INPUT_SIZE + i]);
        }
    }
    __syncthreads();

    if (threadIdx.x < HIDDEN_LAYER_1) hidden1[threadIdx.x] = relu(hidden1[threadIdx.x]);
    __syncthreads();

    // Layer 2: Hidden Layer 1 to Hidden Layer 2
    __shared__ float hidden2[HIDDEN_LAYER_2];
    if (threadIdx.x < HIDDEN_LAYER_2) hidden2[threadIdx.x] = b2[threadIdx.x];
    __syncthreads();

    for (int i = threadIdx.x; i < HIDDEN_LAYER_1; i += blockDim.x) {
        for (int j = 0; j < HIDDEN_LAYER_2; j++) {
            atomicAdd(&hidden2[j], hidden1[i] * w2[j * HIDDEN_LAYER_1 + i]);
        }
    }
    __syncthreads();

    if (threadIdx.x < HIDDEN_LAYER_2) hidden2[threadIdx.x] = relu(hidden2[threadIdx.x]);
    __syncthreads();

    // Layer 3: Hidden Layer 2 to Output Layer
    __shared__ float logits[OUTPUT_SIZE];
    if (threadIdx.x < OUTPUT_SIZE) logits[threadIdx.x] = b3[threadIdx.x];
    __syncthreads();

    for (int i = threadIdx.x; i < HIDDEN_LAYER_2; i += blockDim.x) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            atomicAdd(&logits[j], hidden2[i] * w3[j * HIDDEN_LAYER_2 + i]);
        }
    }
    __syncthreads();

    // Apply softmax to logits
    if (threadIdx.x < OUTPUT_SIZE) {
        float softmax_output[OUTPUT_SIZE];
        softmax(logits, softmax_output, OUTPUT_SIZE);
        output[idx * OUTPUT_SIZE + threadIdx.x] = softmax_output[threadIdx.x];
    }
}

// Tree Reduction Kernel
__global__ void forwardPassTree(float *input, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, float *output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    __shared__ float hidden1[HIDDEN_LAYER_1];
    __shared__ float hidden2[HIDDEN_LAYER_2];
    __shared__ float logits[OUTPUT_SIZE];

    // Initialize shared memory with biases
    if (threadIdx.x < HIDDEN_LAYER_1) hidden1[threadIdx.x] = b1[threadIdx.x];
    if (threadIdx.x < HIDDEN_LAYER_2) hidden2[threadIdx.x] = b2[threadIdx.x];
    if (threadIdx.x < OUTPUT_SIZE) logits[threadIdx.x] = b3[threadIdx.x];
    __syncthreads();

    // Layer 1: Input to Hidden Layer 1
    for (int i = threadIdx.x; i < INPUT_SIZE; i += blockDim.x) {
        for (int j = 0; j < HIDDEN_LAYER_1; j++) {
            atomicAdd(&hidden1[j], input[idx * INPUT_SIZE + i] * w1[j * INPUT_SIZE + i]);
        }
    }
    __syncthreads();

    // Apply ReLU activation
    if (threadIdx.x < HIDDEN_LAYER_1) hidden1[threadIdx.x] = relu(hidden1[threadIdx.x]);
    __syncthreads();

    // Layer 2: Hidden Layer 1 to Hidden Layer 2
    for (int i = threadIdx.x; i < HIDDEN_LAYER_1; i += blockDim.x) {
        for (int j = 0; j < HIDDEN_LAYER_2; j++) {
            atomicAdd(&hidden2[j], hidden1[i] * w2[j * HIDDEN_LAYER_1 + i]);
        }
    }
    __syncthreads();

    // Apply ReLU activation
    if (threadIdx.x < HIDDEN_LAYER_2) hidden2[threadIdx.x] = relu(hidden2[threadIdx.x]);
    __syncthreads();

    // Layer 3: Hidden Layer 2 to Output Layer
    for (int i = threadIdx.x; i < HIDDEN_LAYER_2; i += blockDim.x) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            atomicAdd(&logits[j], hidden2[i] * w3[j * HIDDEN_LAYER_2 + i]);
        }
    }
    __syncthreads();

    // Apply softmax to logits
    if (threadIdx.x < OUTPUT_SIZE) {
        float softmax_output[OUTPUT_SIZE];
        softmax(logits, softmax_output, OUTPUT_SIZE);
        output[idx * OUTPUT_SIZE + threadIdx.x] = softmax_output[threadIdx.x];
    }
}

void runForwardKernelWithStreams(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output, int num_streams,bool atomic=true) {
    cudaStream_t* streams = (cudaStream_t *)malloc(num_streams * sizeof(cudaStream_t));
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 block(128);
    dim3 grid((BATCH_SIZE + block.x - 1) / block.x);

    // Calculate batch size per stream
    int base_batch_size = BATCH_SIZE / num_streams;
    int remainder = BATCH_SIZE % num_streams;

    // Split the batch across the streams, handling the remainder
    for (int i = 0; i < num_streams; i++) {
        // Each stream gets at least `base_batch_size` elements
        int batch_size_for_stream = base_batch_size;

        // Distribute the remainder elements to the first few streams
        if (i < remainder)
            batch_size_for_stream++;

        int start_idx = i * base_batch_size + (i < remainder ? i : remainder);

        // Ensure that the kernel launches with the correct batch size for each stream
        if (atomic) {
            forwardPassTree<<<grid, block, 0, streams[i]>>>(d_input + start_idx * INPUT_SIZE, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output + start_idx * OUTPUT_SIZE, batch_size_for_stream);
        } else {
            forwardPassAtomic<<<grid, block, 0, streams[i]>>>(d_input + start_idx * INPUT_SIZE, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output + start_idx * OUTPUT_SIZE, batch_size_for_stream);
        }
    }

    // Synchronize all streams
    for (int i = 0; i < num_streams; i++)
        cudaStreamSynchronize(streams[i]);

    // Destroy streams after use
    for (int i = 0; i < num_streams; i++)
        cudaStreamDestroy(streams[i]);

    free(streams);
}
