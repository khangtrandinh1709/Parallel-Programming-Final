#include <iostream>
#include <fstream>
#include <vector>
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
#define EPOCHS 10

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

// GPU Kernel for Backward Propagation
__global__ void backward_pass(float *input, float *output, float *labels, 
                              float *w1, float *b1, float *w2, float *b2, float *w3, float *b3, 
                              float *dw1, float *db1, float *dw2, float *db2, float *dw3, float *db3, 
                              int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Arrays for gradients
    float d_hidden2[HIDDEN_LAYER_2] = {0};
    float d_hidden1[HIDDEN_LAYER_1] = {0};

    // Compute gradient of output layer
    float d_output[OUTPUT_SIZE] = {0};
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        d_output[i] = output[idx * OUTPUT_SIZE + i] - labels[idx * OUTPUT_SIZE + i];
    }

    // Compute gradient for W3 and B3
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        db3[i] += d_output[i];
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            dw3[i * HIDDEN_LAYER_2 + j] += d_output[i] * w2[j];
            d_hidden2[j] += d_output[i] * w3[i * HIDDEN_LAYER_2 + j];
        }
    }

    // Gradient through Hidden 2
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        d_hidden2[i] *= relu_derivative(d_hidden2[i]);
    }

    // Compute gradient for W2 and B2
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        db2[i] += d_hidden2[i];
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            dw2[i * HIDDEN_LAYER_1 + j] += d_hidden2[i] * w1[j];
            d_hidden1[j] += d_hidden2[i] * w2[i * HIDDEN_LAYER_1 + j];
        }
    }

    // Gradient through Hidden 1
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        d_hidden1[i] *= relu_derivative(d_hidden1[i]);
    }

    // Compute gradient for W1 and B1
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        db1[i] += d_hidden1[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            dw1[i * INPUT_SIZE + j] += d_hidden1[i] * input[idx * INPUT_SIZE + j];
        }
    }
}

void train(float *normalized_train_images, float *one_hot_train_labels, 
           float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, 
           float *d_w3, float *d_b3, float *d_output, int train_size) {
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        for (int i = 0; i < train_size; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, train_size - i);

            // Copy batch to device
            cudaMemcpy(d_input, normalized_train_images + i * INPUT_SIZE, 
                       current_batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass
            forward_pass<<<(current_batch_size + 31) / 32, 32>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, current_batch_size);

            // Backward pass (gradient calculation and weight update)
            backward_pass<<<(current_batch_size + 31) / 32, 32>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, 
                                                                   one_hot_train_labels + i * OUTPUT_SIZE, current_batch_size);
        }
        std::cout << "Epoch " << epoch + 1 << " completed." << std::endl;
    }
}

// Loss Function
__device__ float categorical_cross_entropy(float *predictions, float *labels, int size) {
    float loss = 0.0;
    for (int i = 0; i < size; ++i) {
        loss -= labels[i] * logf(predictions[i] + 1e-9);
    }
    return loss;
}

// Evaluate Model
float evaluate(float *test_images, float *test_labels, float *d_input, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output, int test_size) {
    int correct = 0;
    float *h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    for (int i = 0; i < test_size; i += BATCH_SIZE) {
        int current_batch_size = std::min(BATCH_SIZE, test_size - i);
        cudaMemcpy(d_input, test_images + i * INPUT_SIZE, current_batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        forward_pass<<<(current_batch_size + 31) / 32, 32>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, current_batch_size);
        cudaMemcpy(h_output, d_output, current_batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

        for (int j = 0; j < current_batch_size; ++j) {
            int predicted_label = 0;
            float max_prob = h_output[j * OUTPUT_SIZE];
            for (int k = 1; k < OUTPUT_SIZE; ++k) {
                if (h_output[j * OUTPUT_SIZE + k] > max_prob) {
                    max_prob = h_output[j * OUTPUT_SIZE + k];
                    predicted_label = k;
                }
            }

            int true_label = 0;
            for (int k = 0; k < OUTPUT_SIZE; ++k) {
                if (test_labels[(i + j) * OUTPUT_SIZE + k] == 1.0f) {
                    true_label = k;
                    break;
                }
            }

            if (predicted_label == true_label) correct++;
        }
    }

    free(h_output);
    return static_cast<float>(correct) / test_size;
}

int main() {
    // Define dataset sizes
    const int train_samples = 60000;
    const int test_samples = 10000;

    // Load and preprocess dataset
    std::vector<unsigned char> train_images, train_labels, test_images, test_labels;
    int train_rows = 0, train_cols = 0, test_rows = 0, test_cols = 0;

    read_idx_file("dataset/train-images-idx3-ubyte", train_images, train_rows, train_cols);
    read_idx_file("dataset/train-labels-idx1-ubyte", train_labels, train_rows, train_cols);
    read_idx_file("dataset/t10k-images-idx3-ubyte", test_images, test_rows, test_cols);
    read_idx_file("dataset/t10k-labels-idx1-ubyte", test_labels, test_rows, test_cols);

    // Normalize images and one-hot encode labels
    std::vector<float> normalized_train_images(train_images.begin(), train_images.end());
    std::vector<float> normalized_test_images(test_images.begin(), test_images.end());
    for (auto &pixel : normalized_train_images) pixel /= 255.0f;
    for (auto &pixel : normalized_test_images) pixel /= 255.0f;

    std::vector<float> one_hot_train_labels, one_hot_test_labels;
    one_hot_encode_labels(train_labels, one_hot_train_labels);
    one_hot_encode_labels(test_labels, one_hot_test_labels);

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

    // Copy weights and biases to device
    cudaMemcpy(d_w1, h_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Train the model
    train(normalized_train_images.data(), one_hot_train_labels.data(), 
          d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, train_samples);

    // Evaluate the model
    float accuracy = evaluate(normalized_test_images.data(), one_hot_test_labels.data(), 
                               d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, test_samples);
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;

    // Free allocated memory
    free(h_input);
    free(h_w1);
    free(h_b1);
    free(h_w2);
    free(h_b2);
    free(h_w3);
    free(h_b3);
    free(h_output);

    cudaFree(d_input);
    cudaFree(d_w1);
    cudaFree(d_b1);
    cudaFree(d_w2);
    cudaFree(d_b2);
    cudaFree(d_w3);
    cudaFree(d_b3);
    cudaFree(d_output);

    return 0;
}
