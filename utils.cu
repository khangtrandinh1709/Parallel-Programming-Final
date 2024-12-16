#include <iostream>
#include <fstream>
#include <cstdlib>
#include "utils.h"

// Load Fashion-MNIST dataset
void loadFashionMNIST(const std::string &path, float *&input, float *&labels, int batch_size) {
    // Simulate loading dataset for simplicity
    input = (float *)malloc(batch_size * INPUT_SIZE * sizeof(float));
    labels = (float *)malloc(batch_size * sizeof(float));
    for (int i = 0; i < batch_size * INPUT_SIZE; ++i) input[i] = (float)(rand() % 256) / 255.0;
    for (int i = 0; i < batch_size; ++i) labels[i] = rand() % 10;
}

// Initialize weights and biases
void initializeWeightsAndBiases(float *&w1, float *&b1, float *&w2, float *&b2, float *&w3, float *&b3) {
    w1 = (float *)malloc(HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    b1 = (float *)malloc(HIDDEN_LAYER_1 * sizeof(float));
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i) w1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) b1[i] = 0.0;
}

// Free host memory
void freeHostMemory(float *input, float *labels, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3) {
    free(input); free(labels); free(w1); free(b1); free(w2); free(b2); free(w3); free(b3);
}

// Allocate device memory
void allocateDeviceMemory(float *&d_input, float *&d_labels, float *&d_w1, float *&d_b1, float *&d_w2, float *&d_b2, float *&d_w3, float *&d_b3, float *&d_output) {
    cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_labels, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
}

// Free device memory
void freeDeviceMemory(float *d_input, float *d_labels, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output) {
    cudaFree(d_input); cudaFree(d_labels); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_output);
}
