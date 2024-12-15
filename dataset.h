#ifndef DATASET_H
#define DATASET_H

#include <string>

void loadFashionMNIST(const std::string& images_path, const std::string& labels_path,
                      float* images, int* labels, int num_samples, int image_size);

#endif
