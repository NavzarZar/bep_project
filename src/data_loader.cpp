#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

// helper for loading the SIFT files
std::vector<float> load_fvecs(const std::string& path, size_t& num_vectors, size_t& dim) {
    std::ifstream input(path, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("could not open the file: " + path);
    }

    int d;
    input.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = static_cast<size_t>(d);

    // calculate the total vectors based on file size
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (sizeof(int) + dim * sizeof(float));

    std::vector<float> data(num_vectors * dim);
    std::cout << "Done!" << std::endl;

    // back to start 
    input.seekg(0, std::ios::beg);

    for (size_t i = 0; i < num_vectors; i++) {
        // skip the dimension
        input.ignore(sizeof(int));
        input.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }

    return data;
}
