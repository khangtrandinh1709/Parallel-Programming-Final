#include "host.h"
#include "utils.h"

HostANN::HostANN() {
    weight1 = (float*)malloc(sizeof(float)*(INPUT_SIZE*HIDDEN_LAYER_1));
    weight2 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_1*HIDDEN_LAYER_2));
    weight3 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_2*OUTPUT_SIZE));
    biase1 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_1));
    biase2 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_2));
    biase3 = (float*)malloc(sizeof(float)*(OUTPUT_SIZE));
    hidden1 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_1));
    hidden2 = (float*)malloc(sizeof(float)*(HIDDEN_LAYER_2));
    output = (float*)malloc(sizeof(float)*(OUTPUT_SIZE));
    input = (float*)malloc(sizeof(float)*(INPUT_SIZE));

    float stddev1 = sqrt(2.0 / INPUT_SIZE);
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i)
        weight1[i] = (rand() / (float)RAND_MAX) * 2 * stddev1 - stddev1;

    // He Initialization for the second weight matrix (HIDDEN_LAYER_1 -> HIDDEN_LAYER_2)
    float stddev2 = sqrt(2.0 / HIDDEN_LAYER_1);
    for (int i = 0; i < HIDDEN_LAYER_1 * HIDDEN_LAYER_2; ++i)
        weight2[i] = (rand() / (float)RAND_MAX) * 2 * stddev2 - stddev2;

    // He Initialization for the third weight matrix (HIDDEN_LAYER_2 -> OUTPUT_SIZE)
    float stddev3 = sqrt(2.0 / HIDDEN_LAYER_2);
    for (int i = 0; i < HIDDEN_LAYER_2 * OUTPUT_SIZE; ++i)
        weight3[i] = (rand() / (float)RAND_MAX) * 2 * stddev3 - stddev3;

    // Initialize biases to zero (common practice)
    for (int i = 0; i < HIDDEN_LAYER_1; i++) biase1[i] = 0.0;
    for (int i = 0; i < HIDDEN_LAYER_2; i++) biase2[i] = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) biase3[i] = 0.0;
}

HostANN::~HostANN(){
    free(weight1);
    free(weight2);
    free(weight3);
    free(biase1);
    free(biase2);
    free(biase3);
    free(hidden1);
    free(hidden2);
    free(output);
    free(input);
}

float HostANN::relu(float x) {
    return x > 0 ? x : 0;
}

void HostANN::softmax(float *input, float *output, int size) {
    float max_val = -FLT_MAX;
    float sum = 0.0;
    // Find max value for numerical stability
    for (int i = 0; i < size; i++) {
        max_val = max(max_val, input[i]);
    }

    // Compute exponentials and sum
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

float* HostANN::forward(float* input) {
    // Input to hidden layer 1
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        hidden1[i] = biase1[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden1[i] += weight1[i * INPUT_SIZE + j] * input[j];
        }
        hidden1[i] = relu(hidden1[i]);
    }

    // Hidden layer 1 to hidden layer 2
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        hidden2[i] = biase2[i];
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            hidden2[i] += weight2[i * HIDDEN_LAYER_1 + j] * hidden1[j];
        }
        hidden2[i] = relu(hidden2[i]);
    }

    // Hidden layer 2 to output
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = biase3[i];
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            output[i] += weight3[i * HIDDEN_LAYER_2 + j] * hidden2[j];
        }
    }
    // Apply softmax to the output
    softmax(output, output, OUTPUT_SIZE);

    return output; // Return the computed output
}

void HostANN::backpropagation(float* target) {
    // Step 1: Output layer error (delta)
    float output_delta[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output_delta[i] = output[i] - target[i]; // Softmax + Cross-entropy derivative
    }

    // Step 2: Backpropagate to hidden layer 2
    float hidden2_delta[HIDDEN_LAYER_2];
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        hidden2_delta[i] = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hidden2_delta[i] += output_delta[j] * weight3[j * HIDDEN_LAYER_2 + i];
        }
        hidden2_delta[i] *= (hidden2[i] > 0.0f ? 1.0f : 0.0f); // ReLU derivative
    }

    // Step 3: Backpropagate to hidden layer 1
    float hidden1_delta[HIDDEN_LAYER_1];
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        hidden1_delta[i] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            hidden1_delta[i] += hidden2_delta[j] * weight2[j * HIDDEN_LAYER_1 + i];
        }
        hidden1_delta[i] *= (hidden1[i] > 0.0f ? 1.0f : 0.0f); // ReLU derivative
    }

    const float gradient_clip_value = 5.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        if (output_delta[i] > gradient_clip_value) {
            output_delta[i] = gradient_clip_value;
        } else if (output_delta[i] < -gradient_clip_value) {
            output_delta[i] = -gradient_clip_value;
        }
    }

    // Step 4: Update weights and biases for weight3, weight2, weight1, biase3, biase2, biase1
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            // Update weight3
            weight3[i * HIDDEN_LAYER_2 + j] -= LEARNING_RATE * output_delta[i] * hidden2[j];
        }
        // Update bias3
        biase3[i] -= LEARNING_RATE * output_delta[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            // Update weight2
            weight2[i * HIDDEN_LAYER_1 + j] -= LEARNING_RATE * hidden2_delta[i] * hidden1[j];
        }
        // Update bias2
        biase2[i] -= LEARNING_RATE * hidden2_delta[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            // Update weight1
            weight1[i * INPUT_SIZE + j] -= LEARNING_RATE * hidden1_delta[i] * input[j];
        }
        // Update bias1
        biase1[i] -= LEARNING_RATE * hidden1_delta[i];
    }
}

void HostANN::train(float* train_data, float* train_labels, int train_size) {
    int num_batches = train_size / TRAIN_BATCH_SIZE;
    float total_epoch_loss = 0.0f;  // To accumulate the total loss over the epoch

    // Allocate memory for device variables
    float* d_input;
    float* d_weight1;
    float* d_weight2;
    float* d_weight3;
    float* d_biase1;
    float* d_biase2;
    float* d_biase3;
    float* d_hidden1;
    float* d_hidden2;
    float* d_output;
    float* d_output_probs;
    float* d_output_delta;
    float* d_hidden2_delta;
    float* d_hidden1_delta;

    cudaMalloc(&d_input, TRAIN_BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weight1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weight2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_weight3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_biase1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_biase2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_biase3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_hidden2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output_probs, OUTPUT_SIZE * TRAIN_BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_output_delta, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden2_delta, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_hidden1_delta, HIDDEN_LAYER_1 * sizeof(float));

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f; // Reset epoch loss for each epoch

        for (int batch = 0; batch < num_batches; ++batch) {
            // Get the current batch
            float* batch_data = train_data + batch * TRAIN_BATCH_SIZE * INPUT_SIZE;
            float* batch_labels = train_labels + batch * TRAIN_BATCH_SIZE * OUTPUT_SIZE;

            // Copy data to the device
            cudaMemcpy(d_input, batch_data, TRAIN_BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_biase1, biase1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_biase2, biase2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_biase3, biase3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weight1, weight1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weight2, weight2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_weight3, weight3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass (kernel call)
            int threads_per_block = 256;
            int num_blocks = (HIDDEN_LAYER_1 + threads_per_block - 1) / threads_per_block;
            forward_kernel_tree<<<num_blocks, threads_per_block>>>(d_input, d_weight1, d_weight2, d_weight3, 
                                                               d_biase1, d_biase2, d_biase3, 
                                                               d_hidden1, d_hidden2, d_output);
            cudaDeviceSynchronize();

            // Apply softmax to the output
            int softmax_threads_per_block = 256;
            int softmax_num_blocks = (TRAIN_BATCH_SIZE * OUTPUT_SIZE + softmax_threads_per_block - 1) / softmax_threads_per_block;
            softmax_kernel<<<softmax_num_blocks, softmax_threads_per_block>>>(d_output, d_output_probs, TRAIN_BATCH_SIZE, OUTPUT_SIZE);
            cudaDeviceSynchronize();

            // Calculate batch loss (cross-entropy)
            float* output_probs = new float[OUTPUT_SIZE * TRAIN_BATCH_SIZE];
            cudaMemcpy(output_probs, d_output_probs, OUTPUT_SIZE * TRAIN_BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            float batch_loss = 0.0f;
            for (int i = 0; i < TRAIN_BATCH_SIZE; ++i) {
                for (int j = 0; j < OUTPUT_SIZE; ++j) {
                    batch_loss -= batch_labels[i * OUTPUT_SIZE + j] * logf(output_probs[i * OUTPUT_SIZE + j] + 1e-7f); // Add small value to avoid log(0)
                }
            }

            delete[] output_probs;
            epoch_loss += batch_loss;  // Accumulate batch loss

            // Backward pass (kernel call)
            backward_kernel<<<num_blocks, threads_per_block>>>(d_input, d_output, batch_labels, d_hidden1, 
                                                                d_hidden2, d_weight1, d_weight2, d_weight3, 
                                                                d_biase1, d_biase2, d_biase3,
                                                                d_output_delta, d_hidden2_delta, d_hidden1_delta,
                                                                LEARNING_RATE);
            cudaDeviceSynchronize();
        }

        // Average loss over all training samples
        epoch_loss /= train_size*TRAIN_BATCH_SIZE/OUTPUT_SIZE;
        total_epoch_loss += epoch_loss;  // Accumulate total loss for tracking progress
        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weight1);
    cudaFree(d_weight2);
    cudaFree(d_weight3);
    cudaFree(d_biase1);
    cudaFree(d_biase2);
    cudaFree(d_biase3);
    cudaFree(d_hidden1);
    cudaFree(d_hidden2);
    cudaFree(d_output);
    cudaFree(d_output_probs);
    cudaFree(d_output_delta);
    cudaFree(d_hidden2_delta);
    cudaFree(d_hidden1_delta);
}

float HostANN::evaluate(float* test_data, float* test_labels, int test_size) {
    int correct_predictions = 0;
    int num_batches = test_size / TEST_BATCH_SIZE;

    for (int batch = 0; batch < num_batches; ++batch) {
        float* batch_data = test_data + batch * TEST_BATCH_SIZE * INPUT_SIZE;
        float* batch_labels = test_labels + batch * TEST_BATCH_SIZE * OUTPUT_SIZE;

        for (int i = 0; i < TEST_BATCH_SIZE; ++i) {
            // Forward pass
            float* output = forward(batch_data + i * INPUT_SIZE);

            // Find the predicted class (max probability)
            int predicted_class = 0;
            float max_prob = output[0];
            for (int j = 1; j < OUTPUT_SIZE; ++j) {
                if (output[j] > max_prob) {
                    max_prob = output[j];
                    predicted_class = j;
                }
            }

            // Find the actual class
            int actual_class = 0;
            for (int j = 0; j < OUTPUT_SIZE; ++j) {
                if (batch_labels[i * OUTPUT_SIZE + j] == 1.0f) {
                    actual_class = j;
                    break;
                }
            }

            // Check if prediction is correct
            if (predicted_class == actual_class) {
                ++correct_predictions;
            }
        }
    }

    float accuracy = (float)correct_predictions / test_size;
    std::cout << "Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;
    return accuracy;
}

void HostANN::saveModel(const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary); // Open in binary mode
    if (!outFile) {
        std::cerr << "Error opening file for saving: " << filename << std::endl;
        return;
    }

    // Save all parameters as raw floats
    outFile.write(reinterpret_cast<const char*>(weight1), sizeof(float) * INPUT_SIZE * HIDDEN_LAYER_1);
    outFile.write(reinterpret_cast<const char*>(weight2), sizeof(float) * HIDDEN_LAYER_1 * HIDDEN_LAYER_2);
    outFile.write(reinterpret_cast<const char*>(weight3), sizeof(float) * HIDDEN_LAYER_2 * OUTPUT_SIZE);
    outFile.write(reinterpret_cast<const char*>(biase1), sizeof(float) * HIDDEN_LAYER_1);
    outFile.write(reinterpret_cast<const char*>(biase2), sizeof(float) * HIDDEN_LAYER_2);
    outFile.write(reinterpret_cast<const char*>(biase3), sizeof(float) * OUTPUT_SIZE);
    outFile.close();
    std::cout << "Model saved successfully to " << filename << std::endl;
}

// Function to load model parameters from a binary file
void HostANN::loadModel(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary); // Open in binary mode
    if (!inFile) {
        std::cerr << "Error opening file for loading: " << filename << std::endl;
        return;
    }

    // Load all parameters as raw floats
    inFile.read(reinterpret_cast<char*>(weight1), sizeof(float) * INPUT_SIZE * HIDDEN_LAYER_1);
    inFile.read(reinterpret_cast<char*>(weight2), sizeof(float) * HIDDEN_LAYER_1 * HIDDEN_LAYER_2);
    inFile.read(reinterpret_cast<char*>(weight3), sizeof(float) * HIDDEN_LAYER_2 * OUTPUT_SIZE);
    inFile.read(reinterpret_cast<char*>(biase1), sizeof(float) * HIDDEN_LAYER_1);
    inFile.read(reinterpret_cast<char*>(biase2), sizeof(float) * HIDDEN_LAYER_2);
    inFile.read(reinterpret_cast<char*>(biase3), sizeof(float) * OUTPUT_SIZE);
    inFile.close();
    std::cout << "Model loaded successfully from " << filename << std::endl;
}
