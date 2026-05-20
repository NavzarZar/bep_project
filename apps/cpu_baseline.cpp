#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include "hnswlib.h"
#include "data_loader.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./run_baseline <path_to_data.fvecs> <M> <ef_construction>" << std::endl;
        return 1;
    }

    std::string base_path = argv[1];
    int M = std::stoi(argv[2]);
    int ef_construction = std::stoi(argv[3]);

    size_t n, dim;

    std::cout << "Loading dataset in memory...." << std::endl;
    auto data = load_fvecs(base_path, n, dim);
    std::cout << "Loaded " << n << " vectors." << std::endl;

    std::cout << "--- BASELINE PARAMS ---" << std::endl;
    std::cout << "M: " << M << ", ef_con: " << ef_construction << std::endl;

    hnswlib::L2Space space(dim);

    // Initialize the algo
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        &space,
        n,
        M,
        ef_construction
    );

    int num_threads = omp_get_max_threads();
    std::cout << "Starting parallel CPU build using " << num_threads << " threads..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Add the points
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++)
    {
        alg_hnsw->addPoint(data.data() + i * dim, i);
        if (i > 0 && i % 100000 == 0) {
            #pragma omp critical
            {
                std::cout << "Inserted " << i << " vectors..." << std::endl;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start; 

    std::cout << "Baseline build time: " << diff.count() << "s" << std::endl;
    std::cout << "Baseline Throughput: " << n / diff.count() << " vectors/sec" << std::endl;
    
    std::string ds_name = (base_path.find("sift") != std::string::npos) ? "sift1m" : "768d";
    std::string index_name = "results/indices/" + ds_name + "_baseline_M" + std::to_string(M) + "_EF" + std::to_string(ef_construction) + ".bin";

    alg_hnsw->saveIndex(index_name);
    std::cout << "Index saved to " << index_name << std::endl;

    // cleaning
    delete alg_hnsw;
    return 0;
}