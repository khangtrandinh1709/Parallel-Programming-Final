// Activation Functions
#include "lib.h"
#include "ANN.h"

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

__device__ void softmax(float *output, int size) {
    float max_val = -1e9;
    for (int i = 0; i < size; ++i) max_val = max(max_val, output[i]);

    float sum = 0.0;
    for (int i = 0; i < size; ++i) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; ++i) output[i] /= sum;
}

// GPU Kernel for Forward Propagation
__global__ void forward_pass(float *input, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, float *output, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float hidden1[HIDDEN_LAYER_1] = {0};
    float hidden2[HIDDEN_LAYER_2] = {0};

    // Layer 1 - Input to Hidden 1
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        hidden1[i] = b1[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden1[i] += input[idx * INPUT_SIZE + j] * w1[i * INPUT_SIZE + j];
        }
        hidden1[i] = relu(hidden1[i]);
    }

    // Layer 2 - Hidden 1 to Hidden 2
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        hidden2[i] = b2[i];
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            hidden2[i] += hidden1[j] * w2[i * HIDDEN_LAYER_1 + j];
        }
        hidden2[i] = relu(hidden2[i]);
    }

    // Layer 3 - Hidden 2 to Output
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[idx * OUTPUT_SIZE + i] = b3[i];
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            output[idx * OUTPUT_SIZE + i] += hidden2[j] * w3[i * HIDDEN_LAYER_2 + j];
        }
    }

    // Softmax
    softmax(&output[idx * OUTPUT_SIZE], OUTPUT_SIZE);
}