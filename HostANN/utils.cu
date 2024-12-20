#include "host.h"
#include "utils.h"

using namespace std;

float* read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if (magic_number != 2051) {
            throw runtime_error("Invalid MNIST image file!");
        }

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        unsigned char* raw_data = new unsigned char[number_of_images * image_size];

        file.read((char*)raw_data, number_of_images * image_size);

        float* dataset = new float[number_of_images * image_size];
        for (int i = 0; i < number_of_images * image_size; ++i) {
            dataset[i] = static_cast<float>(raw_data[i]) / 255.0f;
        }

        delete[] raw_data;

        return dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) {
            throw runtime_error("Invalid MNIST label file! Magic number mismatch.");
        }

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        float* _dataset = new float[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            unsigned char label;
            file.read((char*)&label, sizeof(label));
            _dataset[i] = static_cast<float>(label);
        }

        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}