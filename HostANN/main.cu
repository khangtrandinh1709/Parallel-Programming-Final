#include "host.h"
#include "utils.h"

using namespace std;

int main(int argc, char **argv) {
    try {
        // Paths to MNIST dataset files
        string train_images_path = "../dataset/train-images-idx3-ubyte";
        string train_labels_path = "../dataset/train-labels-idx1-ubyte";
        string test_images_path = "../dataset/t10k-images-idx3-ubyte";
        string test_labels_path = "../dataset/t10k-labels-idx1-ubyte";

        // Variables to store dataset dimensions
        int num_train_images, train_image_size, num_train_labels;
        int num_test_images, test_image_size, num_test_labels;

        // Read MNIST datasets
        float* train_data = read_mnist_images(train_images_path, num_train_images, train_image_size);
        float* train_labels_raw = read_mnist_labels(train_labels_path, num_train_labels);

        float* test_data = read_mnist_images(test_images_path, num_test_images, test_image_size);
        float* test_labels_raw = read_mnist_labels(test_labels_path, num_test_labels);
        
        // Ensure the number of images matches the number of labels
        if (num_train_images != num_train_labels || num_test_images != num_test_labels) {
            throw runtime_error("Mismatch between number of images and labels.");
        }
        cout << "Train data size: " << num_train_images << ", Train label size: " << num_train_labels << endl;
        cout << "Test data size: " << num_test_images << ", Test label size: " << num_test_labels << endl;

        // Convert raw labels to one-hot encoded labels
        float* train_labels = new float[num_train_labels * OUTPUT_SIZE]();
        for (int i = 0; i < num_train_labels; ++i) {
            int label = static_cast<int>(train_labels_raw[i]);
            if (label < 0 || label >= OUTPUT_SIZE) {
                cout << i << " ";
                cout << label << endl;
                throw runtime_error("Invalid label value in training data.");
            }
            train_labels[i * OUTPUT_SIZE + label] = 1.0f;
        }
        delete[] train_labels_raw;

        float* test_labels = new float[num_test_labels * OUTPUT_SIZE]();
        for (int i = 0; i < num_test_labels; ++i) {
            int label = static_cast<int>(test_labels_raw[i]);
            if (label < 0 || label >= OUTPUT_SIZE) {
                throw runtime_error("Invalid label value in test data.");
            }
            test_labels[i * OUTPUT_SIZE + label] = 1.0f;
        }
        delete[] test_labels_raw;
        cout << "Start training"  << endl;
        // Create and train the neural network
        HostANN host;
        host.train(train_data, train_labels, num_train_images);
        cout << "Done training" << endl;
        // Evaluate the neural network
        host.evaluate(test_data, test_labels, num_test_images);

        // Free allocated memory
        delete[] train_data;
        delete[] train_labels;
        delete[] test_data;
        delete[] test_labels;

    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}