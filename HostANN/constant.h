#ifndef CONSTANTS_H
#define CONSTANTS_H

// Hyperparameters
#define INPUT_SIZE 784          // Number of features per input (28x28 images flattened)
#define HIDDEN_LAYER_1 128     // First hidden layer size
#define HIDDEN_LAYER_2 128      // Second hidden layer size
#define OUTPUT_SIZE 10 
#define BATCH_SIZE 32         // Number of classes (0-9)
#define TRAIN_BATCH_SIZE 2000     // Batch size for training
#define TEST_BATCH_SIZE 2000      // Batch size for testing
#define LEARNING_RATE 0.001      // Learning rate
#define EPOCHS 8
#define LAYERS 3

#endif // CONSTANTS_H
