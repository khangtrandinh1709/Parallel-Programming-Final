#ifndef LIB_H
#define LIB_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
using namespace std;

// Hyperparameters
#define INPUT_SIZE 784
#define HIDDEN_LAYER_1 128
#define HIDDEN_LAYER_2 128
#define OUTPUT_SIZE 10
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01
#define EPOCHS 1

#endif