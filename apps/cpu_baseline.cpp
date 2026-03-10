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