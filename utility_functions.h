#include <string>
using namespace std;

#ifndef UTILITY_FUNCTIONS_H
#define UTILITY_FUNCTIONS_H

float* read_mnist_images(string full_path, int& number_of_images, int& image_size);
float* read_mnist_labels(string full_path, int& number_of_labels);

#endif