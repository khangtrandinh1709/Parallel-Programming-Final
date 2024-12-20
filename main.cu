//Update constant.h to align with the change
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"
#include "constant.h"

// Function declarations for kernels
void trainModel(float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3, float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, float learning_rate, int batch_size);
float evaluateModel(float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3, float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, int batch_size);

// Allocate host memory for inputs and labels
void allocateHostMemory(int train_samples, int test_samples) {
    train_input = (float *)malloc(train_samples * INPUT_SIZE * sizeof(float));
    train_labels = (int *)malloc(train_samples * sizeof(int));

    test_input = (float *)malloc(test_samples * INPUT_SIZE * sizeof(float));
    test_labels = (int *)malloc(test_samples * sizeof(int));
}

int main() {
    std::cout << "Starting the program..." << std::endl;

    // Step 1: Dataset parameters
    int train_samples = 60000;  // Total number of training samples
    int test_samples = 10000;   // Total number of testing samples

    // Step 2: Allocate host memory
    allocateHostMemory(train_samples, test_samples);

    float *train_input = nullptr;
    int *train_labels = nullptr;

    float *test_input = nullptr;
    int *test_labels = nullptr;

    // Step 3: Load dataset into host memory
    std::cout << "Loading Fashion-MNIST training dataset..." << std::endl;
    loadFashionMNIST("dataset/", "train-images-idx3-ubyte", "train-labels-idx1-ubyte", train_input, train_labels, train_samples);

    std::cout << "Loading Fashion-MNIST testing dataset..." << std::endl;
    loadFashionMNIST("dataset/", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", test_input, test_labels, test_samples);

    // Step 4: Allocate device memory
    float *d_input, *d_labels, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output;
    allocateDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    // Step 5: Initialize weights and biases
    float *w1, *b1, *w2, *b2, *w3, *b3;
    initializeWeightsAndBiases(w1, b1, w2, b2, w3, b3);

    // Copy weights and biases to device memory
    cudaMemcpy(d_w1, w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Step 6: Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << std::endl;
        for (int batch = 0; batch < train_samples / BATCH_SIZE; ++batch) {
            // Copy batch data to device
            cudaMemcpy(d_input, &train_input[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, &train_labels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

            // Train the model on the batch
            trainModel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, nullptr, nullptr, d_output, d_labels, LEARNING_RATE, BATCH_SIZE);
        }
    }

    // Step 7: Evaluate on test set
    float accuracy = 0.0f;
    for (int batch = 0; batch < test_samples / BATCH_SIZE; ++batch) {
        cudaMemcpy(d_input, &test_input[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, &test_labels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        accuracy += evaluateModel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, nullptr, nullptr, d_output, d_labels, BATCH_SIZE);
    }
    accuracy /= (test_samples / BATCH_SIZE);
    std::cout << "Test Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;

    // Step 7: Free allocated memory
    freeHostMemory();
    freeDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}
