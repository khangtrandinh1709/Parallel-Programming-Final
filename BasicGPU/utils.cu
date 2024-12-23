#include "host.h"
#include "utils.h"

using namespace std;

float* read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2051) {
            throw runtime_error("Invalid MNIST image file!");
        }

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        unsigned char* raw_data = new unsigned char[number_of_images * image_size];

        file.read((char*)raw_data, number_of_images * image_size);

        float* dataset = new float[number_of_images * image_size];
        for (int i = 0; i < number_of_images * image_size; ++i) {
            dataset[i] = static_cast<float>(raw_data[i]) / 255.0f;
        }

        delete[] raw_data;

        return dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) {
            throw runtime_error("Invalid MNIST label file! Magic number mismatch.");
        }

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        float* _dataset = new float[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            unsigned char label;
            file.read((char*)&label, sizeof(label));
            _dataset[i] = static_cast<float>(label);
        }

        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

__global__ void forward_kernel(float* input, float* weight1, float* weight2, float* weight3,
                                float* biase1, float* biase2, float* biase3, 
                                float* hidden1, float* hidden2, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Input to hidden layer 1
    if (idx < HIDDEN_LAYER_1) {
        hidden1[idx] = biase1[idx];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden1[idx] += weight1[idx * INPUT_SIZE + j] * input[j];
        }
        hidden1[idx] = relu(hidden1[idx]);  // ReLU activation
    }

    // Hidden layer 1 to hidden layer 2
    if (idx < HIDDEN_LAYER_2) {
        hidden2[idx] = biase2[idx];
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            hidden2[idx] += weight2[idx * HIDDEN_LAYER_1 + j] * hidden1[j];
        }
        hidden2[idx] = relu(hidden2[idx]);  // ReLU activation
    }

    // Hidden layer 2 to output
    if (idx < OUTPUT_SIZE) {
        output[idx] = biase3[idx];
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            output[idx] += weight3[idx * HIDDEN_LAYER_2 + j] * hidden2[j];
        }
    }
}

__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__global__ void softmax_kernel(float* input, float* output, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one output element for a specific sample in the batch
    if (idx < batch_size * size) {
        int sample_idx = idx / size;  // Get the sample index
        int class_idx = idx % size;   // Get the class index

        // Calculate the offset for the current sample
        float* sample_input = input + sample_idx * size;

        // Compute the maximum value for numerical stability
        float max_val = -FLT_MAX;
        for (int i = 0; i < size; i++) {
            max_val = fmaxf(max_val, sample_input[i]);
        }

        // Compute exponentials and sum for the softmax function
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            output[sample_idx * size + i] = expf(sample_input[i] - max_val);  // Apply the exp function
            sum += output[sample_idx * size + i];  // Sum the exponentials
        }

        // Normalize the result to get probabilities
        output[sample_idx * size + class_idx] /= sum;
    }
}

__global__ void backward_kernel(float* input, float* target, float* output, float* hidden1, 
                                 float* hidden2, float* weight1, float* weight2, float* weight3, 
                                 float* biase1, float* biase2, float* biase3,
                                 float* output_delta, float* hidden2_delta, float* hidden1_delta,
                                 float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate output layer delta (error term)
    if (idx < OUTPUT_SIZE) {
        output_delta[idx] = output[idx] - target[idx]; // Softmax + Cross-entropy derivative
    }

    // Backpropagate to hidden layer 2
    if (idx < HIDDEN_LAYER_2) {
        hidden2_delta[idx] = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hidden2_delta[idx] += output_delta[j] * weight3[j * HIDDEN_LAYER_2 + idx];
        }
        hidden2_delta[idx] *= (hidden2[idx] > 0.0f ? 1.0f : 0.0f); // ReLU derivative
    }

    // Backpropagate to hidden layer 1
    if (idx < HIDDEN_LAYER_1) {
        hidden1_delta[idx] = 0.0f;
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            hidden1_delta[idx] += hidden2_delta[j] * weight2[j * HIDDEN_LAYER_1 + idx];
        }
        hidden1_delta[idx] *= (hidden1[idx] > 0.0f ? 1.0f : 0.0f); // ReLU derivative
    }

    // Update weights and biases for weight3, weight2, weight1, biase3, biase2, biase1
    if (idx < OUTPUT_SIZE) {
        for (int j = 0; j < HIDDEN_LAYER_2; ++j) {
            // Update weight3
            weight3[idx * HIDDEN_LAYER_2 + j] -= learning_rate * output_delta[idx] * hidden2[j];
        }
        // Update bias3
        biase3[idx] -= learning_rate * output_delta[idx];
    }

    if (idx < HIDDEN_LAYER_2) {
        for (int j = 0; j < HIDDEN_LAYER_1; ++j) {
            // Update weight2
            weight2[idx * HIDDEN_LAYER_1 + j] -= learning_rate * hidden2_delta[idx] * hidden1[j];
        }
        // Update bias2
        biase2[idx] -= learning_rate * hidden2_delta[idx];
    }

    if (idx < HIDDEN_LAYER_1) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            // Update weight1
            weight1[idx * INPUT_SIZE + j] -= learning_rate * hidden1_delta[idx] * input[j];
        }
        // Update bias1
        biase1[idx] -= learning_rate * hidden1_delta[idx];
    }
}