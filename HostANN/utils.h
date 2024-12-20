#include <fstream>
#include <string>

float* read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
float* read_mnist_labels(std::string full_path, int& number_of_labels);