#include "dataset.h"
#include <fstream>
#include <iostream>

// Helper to read binary Fashion-MNIST file
void loadFashionMNIST(const std::string& images_path, const std::string& labels_path,
                      float* images, int* labels, int num_samples, int image_size) {
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file.is_open() || !labels_file.is_open()) {
        std::cerr << "Error opening Fashion-MNIST files!" << std::endl;
        exit(1);
    }

    // Skip headers
    images_file.seekg(16);
    labels_file.seekg(8);

    // Read data
    for (int i = 0; i < num_samples; i++) {
        labels_file.read((char*)&labels[i], 1);
        for (int j = 0; j < image_size; j++) {
            unsigned char pixel = 0;
            images_file.read((char*)&pixel, 1);
            images[i * image_size + j] = pixel / 255.0f;  // Normalize to [0,1]
        }
    }

    images_file.close();
    labels_file.close();
}
