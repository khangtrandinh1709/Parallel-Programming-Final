#include <cmath>
#include "utils.h"
#include "constant.h"
#include <cfloat>
#include <iostream>
#include <vector>

class HostANN
{
    protected:
        float *weight1, *weight2, *weight3, *biase1, *biase2, *biase3, *hidden1, *hidden2, *input, *output;
    public:
        HostANN();
        float* forward(float* input);
        void backpropagation(float* target);
        // void backpropagation(float* target, float* weight1_grad, float* weight2_grad, float* weight3_grad,
        //                     float* biase1_grad, float* biase2_grad, float* biase3_grad);
        void train(float* train_data, float* train_labels, int train_size);
        float relu(float x);
        void softmax(float *input, float *output, int size);
        float evaluate(float* test_data, float* test_labels, int test_size);
        ~HostANN();
};