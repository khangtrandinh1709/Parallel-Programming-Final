#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// Hyperparameters
#define INPUT_SIZE 784
#define HIDDEN_LAYER_1 128
#define HIDDEN_LAYER_2 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01
#define EPOCHS 1

// Activation Functions
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

int main() {
    // Allocate memory for weights, biases, inputs, and outputs
    float *h_input, *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_output;
    float *d_input, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output;

    h_input = (float *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    h_w1 = (float *)malloc(HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    h_b1 = (float *)malloc(HIDDEN_LAYER_1 * sizeof(float));
    h_w2 = (float *)malloc(HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    h_b2 = (float *)malloc(HIDDEN_LAYER_2 * sizeof(float));
    h_w3 = (float *)malloc(OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    h_b3 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases randomly
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i) h_w1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) h_b1[i] = 0.0;

    for (int i = 0; i < HIDDEN_LAYER_2 * HIDDEN_LAYER_1; ++i) h_w2[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) h_b2[i] = 0.0;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_2; ++i) h_w3[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < OUTPUT_SIZE; ++i) h_b3[i] = 0.0;

    // Copy data to device
    cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, h_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Forward Pass
    forward_pass<<<(BATCH_SIZE + 31) / 32, 32>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, BATCH_SIZE);

    // Copy output back to host
    cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    free(h_input); free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_w3); free(h_b3); free(h_output);
    cudaFree(d_input); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_output);

    std::cout << "Forward pass completed!" << std::endl;

    return 0;
}
