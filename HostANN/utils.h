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