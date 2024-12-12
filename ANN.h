#ifndef ANN_H
#define ANN_H

void forward_pass(float *input, float *weights, float *biases, ...); 
float relu(float x);
float relu_derivative(float x);
void softmax(float *data, int size);

#endif