#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"

// Hyperparameters
#define INPUT_SIZE 784
#define HIDDEN_LAYER_1 128
#define HIDDEN_LAYER_2 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01
#define EPOCHS 1

// Function declarations
void runTreeReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output);
void runAtomicReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output);

int main() {
    // Load dataset
    float *h_input, *h_labels;
    loadFashionMNIST("dataset/", h_input, h_labels, BATCH_SIZE);

    // Allocate memory for weights, biases, and outputs
    float *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_output;
    float *d_input, *d_labels, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output;

    initializeWeightsAndBiases(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);

    allocateDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);
    cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernels
    std::cout << "Running Tree Reduction Kernel..." << std::endl;
    runTreeReductionKernel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    std::cout << "Running Atomic Reduction Kernel..." << std::endl;
    runAtomicReductionKernel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    // Free memory
    freeHostMemory(h_input, h_labels, h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);
    freeDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}
