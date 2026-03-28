#include <iostream>
#include <vector>
#include "hnswlib.h"
#include "data_loader.h"
#include <hybrid_manager.h>

int main() {
    size_t n, dim;
    std::string base_path = "data/sift_base.fvecs";

    std::cout << "Loading SIFT..." << std::endl;
    auto data = load_fvecs(base_path, n, dim);

    std::cout << "Loaded " << n << " vectors of dimension " << dim << std::endl;

    if (n == 0) {
        std::cerr << "Error, loaded data is null. Check path" << std::endl;
        return 1;
    }

    int M = 16;
    int ef_construction = 200;
    int batch_size = 50000;
    int candidates_per_query = 3000;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_construction);

    std::cout << "GPU Manager..." << std::endl;
    HybridBatchManager gpu_manager(dim, batch_size, candidates_per_query, n);
    gpu_manager.uploadDatasetMirror(data);

    // build the first 10k on cpu to create a decent graph
    int seed_size = 200000;
    std::cout << "Building seed graph (CPU)..." << std::endl;
    for (int i = 0; i < seed_size; i++) {
        alg_hnsw->addPoint(data.data() + i * dim, i);
    }

    std::cout << "Running hybrid insertions..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> gpu_batch_ids;
    int current_id = seed_size;

    while (current_id < n) {
        gpu_batch_ids.clear();

        // fill the batch as the original algo does it
        while (gpu_batch_ids.size() < batch_size && current_id < n) {
            // use hnsw normal random layer generation
            int expected_level = alg_hnsw->getRandomLevel(alg_hnsw->mult_);

            if (expected_level > 0) {
                alg_hnsw->addPoint(data.data() + current_id * dim, current_id);
            } else {
                alg_hnsw->preRegisterHybridNode(current_id, data.data() + current_id * dim);
                gpu_batch_ids.push_back(current_id);
            }
            current_id++;
        }

        int current_batch_size = gpu_batch_ids.size();
        if (current_batch_size == 0) break;

        gpu_manager.setCurrentBatchCount(current_batch_size);


        if (current_id % 50000 < batch_size) {
            std::cout << "\nProcessed " << current_id << " / " << n << " vectors..." << std::endl;
        }

        // each cpu core should take a part of current_batch_size
        // lock-free
        #pragma omp parallel for
        for (int local_i = 0; local_i < current_batch_size; local_i++) {
            int vector_id = gpu_batch_ids[local_i];
            alg_hnsw->addPointHybridBatch(data.data() + vector_id * dim, vector_id, &gpu_manager, local_i);
        }    
        
        std::cout << "[Main] Batch gathered. Executing GPU..." << std::endl;
        gpu_manager.executeBatch();

        // graph linking
        // each cpu core sohuld link to the corresponding vector using the local_i mapping
        #pragma omp parallel for
        for (int local_i = 0; local_i < current_batch_size; local_i++)
        {
            int vector_id = gpu_batch_ids[local_i];
            std::vector<float> dists = gpu_manager.getResultsForQuery(local_i);
            std::vector<int> ids = gpu_manager.getIdsForQuery(local_i);

            alg_hnsw->linkBatchElement(vector_id, ids, dists);
        }
    }


    auto end_time = std::chrono::high_resolution_clock::now();

    alg_hnsw->diagnosticAnalyzeTopology(seed_size, n);
    
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Baseline build time: " << diff.count() << "s" << std::endl;
    std::cout << "Baseline Throughput: " << n / diff.count() << " vectors/sec" << std::endl;
    

    // save the index, no need to rebuild for every test
    std::string index_path = "results/indices/";
    alg_hnsw->saveIndex(index_path + "sift1m_hybrid_M" + std::to_string(M) + "_EF" + std::to_string(ef_construction) + ".bin");
    std::cout << "Index saved to " << index_path << std::endl;

    delete alg_hnsw;
    return 0;
}
