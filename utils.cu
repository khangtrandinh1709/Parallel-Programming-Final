#include <iostream>
#include <fstream>
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

// Function to load Fashion-MNIST dataset
void loadFashionMNIST(const std::string &dataset_path, const std::string &images_file, const std::string &labels_file,
                      float *&input, int *&labels, int &num_samples) {

    // Paths to files
    std::string images_path = dataset_path + images_file;
    std::string labels_path = dataset_path + labels_file;

    // Open images file
    std::ifstream images_stream(images_path, std::ios::binary);
    if (!images_stream.is_open()) {
        std::cerr << "Error: Could not open images file: " << images_path << std::endl;
        exit(1);
    }

    // Open labels file
    std::ifstream labels_stream(labels_path, std::ios::binary);
    if (!labels_stream.is_open()) {
        std::cerr << "Error: Could not open labels file: " << labels_path << std::endl;
        exit(1);
    }

    // Read images file header
    int magic_number, num_images, num_rows, num_cols;
    images_stream.read((char *)&magic_number, 4);
    images_stream.read((char *)&num_images, 4);
    images_stream.read((char *)&num_rows, 4);
    images_stream.read((char *)&num_cols, 4);

    magic_number = reverseInt(magic_number);
    num_images = reverseInt(num_images);
    num_rows = reverseInt(num_rows);
    num_cols = reverseInt(num_cols);

    // Read labels file header
    int labels_magic_number, num_labels;
    labels_stream.read((char *)&labels_magic_number, 4);
    labels_stream.read((char *)&num_labels, 4);

    labels_magic_number = reverseInt(labels_magic_number);
    num_labels = reverseInt(num_labels);

    if (num_images != num_labels) {
        std::cerr << "Error: Number of images and labels do not match!" << std::endl;
        exit(1);
    }

    // Allocate memory for input and labels
    num_samples = num_images;
    int image_size = num_rows * num_cols;

    input = (float *)malloc(num_samples * image_size * sizeof(float));
    labels = (int *)malloc(num_samples * sizeof(int));

    if (!input || !labels) {
        std::cerr << "Error: Memory allocation failed!" << std::endl;
        exit(1);
    }

    // Read images and normalize pixel values to [0, 1]
    unsigned char pixel;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < image_size; j++) {
            images_stream.read((char *)&pixel, 1);
            input[i * image_size + j] = pixel / 255.0f; // Normalize to [0, 1]
        }
    }

    // Read labels
    unsigned char label;
    for (int i = 0; i < num_samples; i++) {
        labels_stream.read((char *)&label, 1);
        labels[i] = (int)label;
    }

    // Close file streams
    images_stream.close();
    labels_stream.close();

    std::cout << "Loaded " << num_samples << " samples from " << images_file << " and " << labels_file << std::endl;
}

// Initialize weights and biases
void initializeWeightsAndBiases(float *&w1, float *&b1, float *&w2, float *&b2, float *&w3, float *&b3) {
    w1 = (float *)malloc(HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    b1 = (float *)malloc(HIDDEN_LAYER_1 * sizeof(float));
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i) w1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) b1[i] = 0.0;
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
