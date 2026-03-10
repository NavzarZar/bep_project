#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "hnswlib.h"

// helper for loading the SIFT files
std::vector<float> load_fvecs(const std::string& path, size_t& num_vectors, size_t& dim) {
    std::ifstream input(path, std::ios::binary);
    int d;
    input.read(reinterpret_cast<char*>(&d), sizeof(int));
    dim = d;

    // calculate the total vectors based on file size
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    num_vectors = file_size / (sizeof(int) + dim * sizeof(float));

    std::vector<float> data(num_vectors * dim);
    input.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num_vectors; i++) {
        // skip the dimension
        input.ignore(sizeof(int));
        input.read(reinterpret_cast<char*>(data.data() + i * dim), dim * sizeof(float));
    }

    return data;
}

int main() {
    size_t n, dim;
    std::string base_path = "data/sift_base.fvecs";

    std::cout << " Loading sift in memory...." << std::endl;
    auto data = load_fvecs(base_path, n, dim);
    std::cout << "Loaded " << n << " vectors." << std::endl;

    // Parameters for hnsw
    int M = 16; // max number of outgoing connections for each node
    int ef_construction = 200; // size of the candidate list during the construction

    hnswlib::L2Space space(dim);

    // Initialize the algo
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        &space,
        n,
        M,
        ef_construction
    );

    std:: cout << "Starting build....." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Add the points
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        alg_hnsw->addPoint(data.data() + i * dim, i);
        if (i > 0 && i % 100000 == 0) {
            std::cout << "Inserted " << i << " vectors..." << std::endl;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    
    std::cout << "-------The CPU Baseline Results------" << std::endl;
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << n / duration.count() << " vectors/sec" << std::endl;

    // cleaning
    delete alg_hnsw;
    return 0;
}