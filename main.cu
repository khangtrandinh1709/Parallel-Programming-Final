#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "utils.h"
#include "constant.h"

// Function declarations for kernels
void trainModel(
    float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3,
    float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, float learning_rate, int batch_size, float* d_loss);
float evaluateModel(
    float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3,
    float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, int batch_size);

// Normalize the input data
void normalizeInput(float* input, int samples, int input_size) {
    for (int i = 0; i < samples * input_size; ++i) {
        input[i] /= 255.0f; // Normalize pixel values to [0, 1]
    }
}

// Allocate host memory for inputs and labels
void allocateHostMemory(int train_samples, int test_samples) {
    train_input = (float*)malloc(train_samples * INPUT_SIZE * sizeof(float));
    train_labels = (int*)malloc(train_samples * sizeof(int));

    test_input = (float*)malloc(test_samples * INPUT_SIZE * sizeof(float));
    test_labels = (int*)malloc(test_samples * sizeof(int));
}

int main() {
    std::cout << "Starting the program..." << std::endl;

    // Step 1: Dataset parameters
    int train_samples = 60000;  // Total number of training samples
    int test_samples = 10000;   // Total number of testing samples

    // Step 2: Allocate host memory
    allocateHostMemory(train_samples, test_samples);

    // Step 3: Load dataset into host memory
    std::cout << "Loading Fashion-MNIST training dataset..." << std::endl;
    loadFashionMNIST("dataset/", "train-images-idx3-ubyte", "train-labels-idx1-ubyte", train_input, train_labels, train_samples);

    std::cout << "Loading Fashion-MNIST testing dataset..." << std::endl;
    loadFashionMNIST("dataset/", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", test_input, test_labels, test_samples);

    // Step 4: Normalize input data
    std::cout << "Normalizing the input data..." << std::endl;
    normalizeInput(train_input, train_samples, INPUT_SIZE);
    normalizeInput(test_input, test_samples, INPUT_SIZE);

    // Step 5: Allocate device memory
    float *d_input, *d_labels, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output, *d_loss;
    allocateDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);

    // Step 6: Initialize weights and biases
    float *w1 = new float[HIDDEN_LAYER_1 * INPUT_SIZE];
    float *b1 = new float[HIDDEN_LAYER_1];
    float *w2 = new float[HIDDEN_LAYER_2 * HIDDEN_LAYER_1];
    float *b2 = new float[HIDDEN_LAYER_2];
    float *w3 = new float[OUTPUT_SIZE * HIDDEN_LAYER_2];
    float *b3 = new float[OUTPUT_SIZE];

    initializeWeightsAndBiases(w1, b1, w2, b2, w3, b3);

    cudaMemcpy(d_w1, w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Step 7: Training loop
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        std::cout << "Epoch " << (epoch + 1) << "/" << EPOCHS << std::endl;

        float epoch_loss = 0.0f;
        cudaMemset(d_loss, 0, sizeof(float));

        int correct_predictions = 0;

        for (int batch = 0; batch < train_samples / BATCH_SIZE; ++batch) {
            // Copy batch data to device
            cudaMemcpy(d_input, &train_input[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, &train_labels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

            // Train the model on the batch
            trainModel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, nullptr, nullptr, d_output, d_labels, LEARNING_RATE, BATCH_SIZE, d_loss);

            // Accumulate loss
            float batch_loss = 0.0f;
            cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            epoch_loss += batch_loss;

            // Calculate batch accuracy
            float *h_output = new float[BATCH_SIZE * OUTPUT_SIZE];
            cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < BATCH_SIZE; ++i) {
                int predicted_label = 0;
                float max_prob = h_output[i * OUTPUT_SIZE];
                for (int j = 1; j < OUTPUT_SIZE; ++j) {
                    if (h_output[i * OUTPUT_SIZE + j] > max_prob) {
                        max_prob = h_output[i * OUTPUT_SIZE + j];
                        predicted_label = j;
                    }
                }
                if (predicted_label == train_labels[batch * BATCH_SIZE + i]) {
                    ++correct_predictions;
                }
            }
            delete[] h_output;
        }

        std::cout << "Epoch " << (epoch + 1) << " Training Accuracy: "
                  << (static_cast<float>(correct_predictions) / train_samples) * 100.0f << "%" << std::endl;
        std::cout << "Epoch Loss: " << epoch_loss << std::endl;
    }

    // Step 8: Evaluate on test set
    float accuracy = 0.0f;
    for (int batch = 0; batch < test_samples / BATCH_SIZE; ++batch) {
        cudaMemcpy(d_input, &test_input[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, &test_labels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);
        accuracy += evaluateModel(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, nullptr, nullptr, d_output, d_labels, BATCH_SIZE);
    }
    accuracy /= (test_samples / BATCH_SIZE);
    std::cout << "Test Accuracy: " << (accuracy * 100.0f) << "%" << std::endl;

    // Step 9: Free allocated memory
    delete[] w1;
    delete[] b1;
    delete[] w2;
    delete[] b2;
    delete[] w3;
    delete[] b3;

    freeHostMemory();
    freeDeviceMemory(d_input, d_labels, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output);
    cudaFree(d_loss);

    std::cout << "Execution completed successfully!" << std::endl;
    return 0;
}

