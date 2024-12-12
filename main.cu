#include "lib.h"
#include "ANN.h"
#include "utility_functions.h"

int main(int argc, char** argv) {
    // Allocate memory for weights, biases, inputs, and outputs
    float *h_input, *h_w1, *h_b1, *h_w2, *h_b2, *h_w3, *h_b3, *h_output;
    float *d_input, *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3, *d_output;

    h_input = (float *)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
    h_w1 = (float *)malloc(HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    h_b1 = (float *)malloc(HIDDEN_LAYER_1 * sizeof(float));
    h_w2 = (float *)malloc(HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    h_b2 = (float *)malloc(HIDDEN_LAYER_2 * sizeof(float));
    h_w3 = (float *)malloc(OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    h_b3 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    h_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    string data_train_path = "./dataset/train-images-idx3-ubyte", data_test_path = "./dataset/t10k-images-idx3-ubyte";
    string label_train_path = "./dataset/train-labels-idx1-ubyte", label_test_path= "./dataset/t10k-labels-idx1-ubyte";
    
    if (argc == 5)
    {
        data_train_path = argv[1];
        label_train_path = argv[2];
        data_test_path = argv[3];
        label_test_path = argv[4];
    }

    int number_of_images = 0, image_size = 0;
    h_input = read_mnist_images(data_train_path, number_of_images, image_size);

    cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float));
    cudaMalloc(&d_b2, HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float));
    cudaMalloc(&d_b3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases randomly
    for (int i = 0; i < HIDDEN_LAYER_1 * INPUT_SIZE; ++i) h_w1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_1; ++i) h_b1[i] = 0.0;

    for (int i = 0; i < HIDDEN_LAYER_2 * HIDDEN_LAYER_1; ++i) h_w2[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < HIDDEN_LAYER_2; ++i) h_b2[i] = 0.0;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_2; ++i) h_w3[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < OUTPUT_SIZE; ++i) h_b3[i] = 0.0;

    // Copy data to device
    cudaMemcpy(d_input, h_input, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, h_w1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, h_w3, OUTPUT_SIZE * HIDDEN_LAYER_2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, h_b3, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Forward Pass
    forward_pass<<<(BATCH_SIZE + 31) / 32, 32>>>(d_input, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, d_output, BATCH_SIZE);

    // Copy output back to host
    cudaMemcpy(h_output, d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    free(h_input); free(h_w1); free(h_b1); free(h_w2); free(h_b2); free(h_w3); free(h_b3); free(h_output);
    cudaFree(d_input); cudaFree(d_w1); cudaFree(d_b1); cudaFree(d_w2); cudaFree(d_b2); cudaFree(d_w3); cudaFree(d_b3); cudaFree(d_output);

    std::cout << "Forward pass completed!" << std::endl;

    return 0;
}
