#ifndef FORWARD_H
#define FORWARD_H

#define HIDDEN_LAYER_1 128
#define HIDDEN_LAYER_2 128
#define OUTPUT_SIZE 10
#define BLOCK_SIZE 32

void forwardPassGPU(float* input, float* w1, float* b1, float* w2, float* b2,
                    float* w3, float* b3, float* output, int batch_size, int input_size);

#endif
