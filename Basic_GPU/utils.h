#include <fstream>
#include <string>
#include <chrono>
#pragma once
class Timer
{
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;

public:
    void Start()
    {
        start = std::chrono::high_resolution_clock::now();
    }

    void Stop()
    {
        stop = std::chrono::high_resolution_clock::now();
    }

    double Elapsed() const
    {
        return std::chrono::duration<double, std::milli>(stop - start).count();
    }
};

float* read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
float* read_mnist_labels(std::string full_path, int& number_of_labels);
__device__ float relu(float x);
__global__ void forward_kernel(float* input, float* weight1, float* weight2, float* weight3,
                                float* biase1, float* biase2, float* biase3, 
                                float* hidden1, float* hidden2, float* output);
__global__ void softmax_kernel(float* input, float* output, int batch_size, int size);
__global__ void backward_kernel(float* input, float* target, float* output, float* hidden1, 
                                 float* hidden2, float* weight1, float* weight2, float* weight3, 
                                 float* biase1, float* biase2, float* biase3,
                                 float* output_delta, float* hidden2_delta, float* hidden1_delta,
                                 float learning_rate);