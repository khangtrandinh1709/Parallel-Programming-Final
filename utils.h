#ifndef UTILS_H
#define UTILS_H

#include <string>

// Utility functions
void loadFashionMNIST(const std::string &path, float *&input, float *&labels, int batch_size);
void initializeWeightsAndBiases(float *&w1, float *&b1, float *&w2, float *&b2, float *&w3, float *&b3);
void freeHostMemory(float *input, float *labels, float *w1, float *b1, float *w2, float *b2, float *w3, float *b3);
void allocateDeviceMemory(float *&d_input, float *&d_labels, float *&d_w1, float *&d_b1, float *&d_w2, float *&d_b2, float *&d_w3, float *&d_b3, float *&d_output);
void freeDeviceMemory(float *d_input, float *d_labels, float *d_w1, float *d_b1, float *d_w2, float *d_b2, float *d_w3, float *d_b3, float *d_output);

#endif
