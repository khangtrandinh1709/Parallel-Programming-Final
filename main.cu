//Update constant.h to align with the change
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"
#include "constant.h"
#include "relu.h"

// Function declarations for kernels
void runTreeReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output);
void runAtomicReductionKernel(float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output);

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

    // Step 4: Allocate host memory for weights, biases, and output
    float *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_output;
    initializeWeightsAndBiases(h_w1, h_b1, h_w2, h_b2, h_w3, h_b3);
    h_output = (float *)malloc(TRAIN_BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Step 5: Allocate device memory
    float *d_input, *d_labels, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output;
    allocateDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    // Step 6: Copy a training batch to device and run kernels
    std::cout << "Running kernels on training batch..." << std::endl;

    cudaMemcpy(d_input, train_input, TRAIN_BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, train_labels, TRAIN_BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_w1, h_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);

    runTreeReductionKernel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);
    runAtomicReductionKernel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    // Copy output back to host
    cudaMemcpy(h_output, d_output, TRAIN_BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print sample output
    std::cout << "Kernel execution completed. Sample output:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "Output[" << i << "] = " << h_output[i] << std::endl;
    }

    // Step 7: Free allocated memory
    freeHostMemory();
    free(h_output);
    freeDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}
