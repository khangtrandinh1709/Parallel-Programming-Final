#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include "constant.h"
#include "utils.h"

// ReLU activation function
__device__ float relu(float x) {
    return fmaxf(0.0f, x);
}

// Derivative of ReLU activation function
__device__ float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

// Softmax activation function
__device__ void softmax(float* input, float* output, int length) {
    float max_val = input[0];
    for (int i = 1; i < length; ++i) {
        max_val = fmaxf(max_val, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < length; ++i) {
        output[i] /= sum;
    }
}

// Forward pass kernel
__global__ void forwardPassKernel(
    float* input, float* w1, float* b1, float* w2, float* b2, float* w3, float* b3,
    float* hidden1, float* hidden2, float* output, int batch_size) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;

    // Layer 1 (Input -> Hidden 1)
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < INPUT_SIZE; ++j) {
            sum += input[idx * INPUT_SIZE + j] * w1[i * INPUT_SIZE + j];
        }
        sum += b1[i];
        hidden1[idx * HIDDEN_LAYER_1 + i] = relu(sum);
    }

    // Layer 2 (Hidden 1 -> Hidden 2)
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            sum += hidden1[idx * HIDDEN_LAYER_1 + j] * w2[i * HIDDEN_LAYER_1 + j];
        }
        sum += b2[i];
        hidden2[idx * HIDDEN_LAYER_2 + i] = relu(sum);
    }

    // Layer 3 (Hidden 2 -> Output)
    float logits[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            sum += hidden2[idx * HIDDEN_LAYER_2 + j] * w3[i * HIDDEN_LAYER_2 + j];
        }
        sum += b3[i];
        logits[i] = sum;
    }

    // Apply softmax activation
    softmax(logits, &output[idx * OUTPUT_SIZE], OUTPUT_SIZE);
}

// Compute loss kernel
__global__ void computeLossKernel(float* output, int* labels, float* loss, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;

    int label = labels[idx];
    float predicted = output[idx * OUTPUT_SIZE + label];
    atomicAdd(loss, -logf(predicted + 1e-7f));
}

// Training kernel (Backward pass and weight update)
__global__ void trainKernel(
    float* input, float* w1, float* b1, float* w2, float* b2, float* w3, float* b3,
    float* hidden1, float* hidden2, float* output, float* labels, float learning_rate, int batch_size) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size) return;

    // Gradient placeholders
    float grad_output[OUTPUT_SIZE] = {0};
    float grad_hidden2[HIDDEN_LAYER_2] = {0};
    float grad_hidden1[HIDDEN_LAYER_1] = {0};

    // Compute output layer gradients (Softmax + Cross-entropy)
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        grad_output[i] = output[idx * OUTPUT_SIZE + i];
    }
    grad_output[labels[idx]] -= 1.0f;

    // Backpropagate to hidden layer 2
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            sum += grad_output[j] * w3[j * HIDDEN_LAYER_2 + i];
        }
        grad_hidden2[i] = sum * relu_derivative(hidden2[idx * HIDDEN_LAYER_2 + i]);
    }

    // Backpropagate to hidden layer 1
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            sum += grad_hidden2[j] * w2[j * HIDDEN_LAYER_1 + i];
        }
        grad_hidden1[i] = sum * relu_derivative(hidden1[idx * HIDDEN_LAYER_1 + i]);
    }

    // Update weights and biases
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            atomicAdd(&w3[i * HIDDEN_LAYER_2 + j], -learning_rate * grad_output[i] * hidden2[idx * HIDDEN_LAYER_2 + j]);
        }
        atomicAdd(&b3[i], -learning_rate * grad_output[i]);
    }

    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            atomicAdd(&w2[i * HIDDEN_LAYER_1 + j], -learning_rate * grad_hidden2[i] * hidden1[idx * HIDDEN_LAYER_1 + j]);
        }
        atomicAdd(&b2[i], -learning_rate * grad_hidden2[i]);
    }

    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            atomicAdd(&w1[i * INPUT_SIZE + j], -learning_rate * grad_hidden1[i] * input[idx * INPUT_SIZE + j]);
        }
        atomicAdd(&b1[i], -learning_rate * grad_hidden1[i]);
    }
}

// Host function to launch training
void trainModel(
    float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3,
    float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, float learning_rate, int batch_size) {

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    forwardPassKernel<<<num_blocks, threads_per_block>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_hidden1, d_hidden2, d_output, batch_size);
    trainKernel<<<num_blocks, threads_per_block>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_hidden1, d_hidden2, d_output, d_labels, learning_rate, batch_size);
}

// Host function to evaluate the model
float evaluateModel(float* d_input, float* d_w1, float* d_b1, float* d_w2, float* d_b2, float* d_w3, float* d_b3, float* d_hidden1, float* d_hidden2, float* d_output, float* d_labels, int batch_size) {
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    float* h_output = (float*)malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    forwardPassKernel<<<num_blocks, threads_per_block>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_hidden1, d_hidden2, d_output, batch_size);
    cudaMemcpy(h_output, d_output, batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    int correct_predictions = 0;
    for (int i = 0; i < batch_size; ++i) {
        int predicted_label = 0;
        float max_prob = h_output[i * OUTPUT_SIZE];
        for (int j = 1; j < OUTPUT_SIZE; ++j) {
            if (h_output[i * OUTPUT_SIZE + j] > max_prob) {
                max_prob = h_output[i * OUTPUT_SIZE + j];
                predicted_label = j;
            }
        }
        if (predicted_label == d_labels[i]) {
            correct_predictions++;
        }
    }

    free(h_output);
    return static_cast<float>(correct_predictions) / batch_size;
}
