#include "dataset.h"
#include "forward.h"
#include <iostream>

#define BATCH_SIZE 32
#define INPUT_SIZE 784
#define NUM_TRAIN 60000

int main() {
    // Allocate host memory
    float* h_input = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    int* h_labels = (int*)malloc(BATCH_SIZE * sizeof(int));

    float *h_w1, *h_b1, *h_output;
    cudaMallocManaged(&h_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    cudaMallocManaged(&h_b1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMallocManaged(&h_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Load dataset
    loadFashionMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                     h_input, h_labels, BATCH_SIZE, INPUT_SIZE);

    // Randomly initialize weights and biases
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; i++) h_w1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_1; i++) h_b1[i] = 0.0;

    // Forward Pass
    forwardPassGPU(h_input, h_w1, h_b1, nullptr, nullptr, nullptr, nullptr, h_output, BATCH_SIZE, INPUT_SIZE);

    std::cout << "Forward Pass Completed!" << std::endl;

    // Cleanup
    cudaFree(h_w1);
    cudaFree(h_b1);
    cudaFree(h_output);
    free(h_input);
    free(h_labels);

    return 0;
}
