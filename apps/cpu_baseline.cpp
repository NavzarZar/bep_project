#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "hnswlib.h"
#include "data_loader.h"

int main() {
    size_t n, dim;
    std::string base_path = "data/sift_base.fvecs";

    std::cout << "Loading sift in memory...." << std::endl;
    auto data = load_fvecs(base_path, n, dim);
    std::cout << "Loaded " << n << " vectors." << std::endl;

    hnswlib::L2Space space(dim);

    int M = 16;
    int ef_construction = 200;

    // Initialize the algo
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        &space,
        n,
        M,
        ef_construction
    );

    std:: cout << "Starting Baseline CPU build....." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Add the points
    for (int i = 0; i < n; i++)
    {
        alg_hnsw->addPoint(data.data() + i * dim, i);
        if (i > 0 && i % 100000 == 0) {
            std::cout << "Inserted " << i << " vectors..." << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start; 

    std::cout << "Baseline build time: " << diff.count() << "s" << std::endl;
    std::cout << "Baseline Throughput: " << n / diff.count() << " vectors/sec" << std::endl;
    
    std::string index_path = "results/indices";
    alg_hnsw->saveIndex("results/indices/sift1m_baseline_M" + std::to_string(M) + "_EF" + std::to_string(ef_construction) + ".bin");
    std::cout << "Index saved to " << index_path << std::endl;

    // cleaning
    delete alg_hnsw;
    return 0;
}