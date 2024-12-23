#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "utils.h"
#include "constant.h"

#define IMAGE_SIZE 784 // 28x28 for Fashion-MNIST images
#define NUM_CLASSES 10 // 10 categories in Fashion-MNIST
float *train_input = nullptr;
float *test_input = nullptr;
int *train_labels = nullptr;
int *test_labels = nullptr;

// Function to reverse bytes (big-endian to little-endian)
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Function to load MNIST images
float* loadMNISTImages(const std::string& full_path, int& number_of_images, int& image_size) {
    std::ifstream file(full_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }

    int magic_number = 0, n_rows = 0, n_cols = 0;

    // Read and validate header
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2051) {
        throw std::runtime_error("Invalid MNIST image file! Magic number: " + std::to_string(magic_number));
    }

    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    n_rows = reverseInt(n_rows);

    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    image_size = n_rows * n_cols;

    // Debugging: Verify header values
    std::cout << "Images File: Magic Number = " << magic_number
              << ", Num Images = " << number_of_images
              << ", Rows = " << n_rows
              << ", Cols = " << n_cols << std::endl;

    // Seek to the start of pixel data
    file.seekg(16, std::ios::beg);
    if (!file) {
        throw std::runtime_error("Failed to seek to pixel data!");
    }

    // Allocate memory for raw pixel data
    unsigned char* raw_data = new unsigned char[number_of_images * image_size];
    file.read(reinterpret_cast<char*>(raw_data), number_of_images * image_size);

    // Check for read errors
    if (!file) {
        delete[] raw_data;
        throw std::runtime_error("Error reading image data!");
    }

    // Debugging: Print the first 10 raw pixel values before normalization
    std::cout << "First 10 raw pixel values of the first image (before normalization):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<int>(raw_data[i]) << " ";
    }
    std::cout << std::endl;

    // Allocate memory for normalized dataset
    float* dataset = new float[number_of_images * image_size];
    if (!dataset) {
        delete[] raw_data;
        throw std::runtime_error("Memory allocation failed for dataset!");
    }

    // Normalize the pixel values and validate
    for (int i = 0; i < number_of_images * image_size; ++i) {
        dataset[i] = static_cast<float>(raw_data[i]) / 255.0f;

        // Validate range after normalization
        if (dataset[i] < 0.0f || dataset[i] > 1.0f) {
            std::cerr << "Error: Normalized value out of range at index " << i << " (value: " << dataset[i] << ")" << std::endl;
            delete[] raw_data;
            delete[] dataset;
            throw std::runtime_error("Data overflow detected during normalization!");
        }
    }

    // Debugging: Print the first 10 normalized pixel values
    std::cout << "First 10 normalized pixel values of the first image:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << dataset[i] << " ";
    }
    std::cout << std::endl;

    delete[] raw_data;
    return dataset;
}

// Function to load MNIST labels
int* loadMNISTLabels(const std::string& full_path, int& number_of_labels) {
    std::ifstream file(full_path, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }

    int magic_number = 0;

    // Read header
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2049) {
        throw std::runtime_error("Invalid MNIST label file! Magic number: " + std::to_string(magic_number));
    }

    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = reverseInt(number_of_labels);

    // Debugging: Verify header values
    std::cout << "Labels File: Magic Number = " << magic_number
              << ", Num Labels = " << number_of_labels << std::endl;

    // Allocate memory for labels
    unsigned char* raw_labels = new unsigned char[number_of_labels];
    file.read(reinterpret_cast<char*>(raw_labels), number_of_labels);

    // Check for reading errors
    if (!file) {
        delete[] raw_labels;
        throw std::runtime_error("Error reading label data from `" + full_path + "`!");
    }

    // Allocate memory for converted labels
    int* labels = new int[number_of_labels];
    for (int i = 0; i < number_of_labels; ++i) {
        labels[i] = static_cast<int>(raw_labels[i]);
    }

    // Debugging: Print the first 10 labels
    std::cout << "First 10 labels:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << labels[i] << " ";
    }
    std::cout << std::endl;

    delete[] raw_labels;
    return labels;
}

// Main function to load Fashion-MNIST
void loadFashionMNIST(const std::string& dataset_path, const std::string& images_file, const std::string& labels_file,
                      float*& input, int*& labels, int& num_samples) {
    int image_size;
    input = loadMNISTImages(dataset_path + images_file, num_samples, image_size);
    labels = loadMNISTLabels(dataset_path + labels_file, num_samples);
}

// Initialize weights and biases
void initializeWeightsAndBiases(float* w1, float* b1, float* w2, float* b2, float* w3, float* b3) {
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i) {
        w1[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) {
        b1[i] = 0.0f;
    }
    for (int i = 0; i < HIDDEN_LAYER_2 * HIDDEN_LAYER_1; ++i) {
        w2[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) {
        b2[i] = 0.0f;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_2; ++i) {
        w3[i] = static_cast<float>(rand()) / RAND_MAX * 0.01f;
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        b3[i] = 0.0f;
    }
}


// Free host memory
void freeHostMemory() {
    if (train_input) free(train_input);
    if (train_labels) free(train_labels);
    if (test_input) free(test_input);
    if (test_labels) free(test_labels);
}

// Allocate device memory
void allocateDeviceMemory(float *&d_input, float *&d_labels, float *&d_w1, float *&d_b1, float *&d_w2, float *&d_b2, 
                          float *&d_w3, float *&d_b3, float *&d_output) {
    cudaMalloc(&d_input, BATCH_SIZE * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_labels, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_w1, HIDDEN_LAYER_1 * IMAGE_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
}

// Free device memory
void freeDeviceMemory(float *d_input, float *d_labels, float *d_w1, float *d_b1, float *d_w2, float *d_b2, 
                      float *d_w3, float *d_b3, float *d_output) {
    cudaFree(d_input); cudaFree(d_labels); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_output);
}
