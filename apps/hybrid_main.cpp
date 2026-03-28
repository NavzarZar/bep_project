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
    int candidates_per_query = 1000;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_construction);

    std::cout << "GPU Manager..." << std::endl;
    HybridBatchManager gpu_manager(dim, batch_size, candidates_per_query, n);
    gpu_manager.uploadDatasetMirror(data);

    // build the first 10k on cpu to create a decent graph
    int seed_size = 100000;
    std::cout << "Building seed graph (CPU)..." << std::endl;
    for (int i = 0; i < seed_size; i++) {
        alg_hnsw->addPoint(data.data() + i * dim, i);
    }

    std::cout << "Running hybrid insertions..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = seed_size; i < n; i++)
    {
        // 
        alg_hnsw->addPointHybridBatch(data.data() + i * dim, i, &gpu_manager);

        if (gpu_manager.isFull()) {
            std::cout << "\n[Main] Batch full at i=" << i << ". Executing GPU..." << std::endl;
            // gpu math
            gpu_manager.executeBatch();
            std::cout << "[Main] GPU Finished. Starting Linking..." << std::endl;

            int batch_start_idx = i - batch_size + 1;


            #pragma omp parallel for
            for (int b = 0; b < batch_size; b++)
            {
                int vector_id = batch_start_idx + b;

                // get the data from manager
                std::vector<float> dists = gpu_manager.getResultsForQuery(b);
                std::vector<int> ids = gpu_manager.getIdsForQuery(b);

                // call linking
                if (b % 2000 == 0) std::cout << "[Main] Linking element " << b << " of batch" << std::endl;
                alg_hnsw->linkBatchElement(vector_id, ids, dists);
            }
            std::cout << "[Main] Batch link complete.\n" << std::endl;

            if (i % 50000 == 0 || i == n - 1) {
                std::cout << "Processed " << i << " / " << n << " vectors..." << std::endl;
            }

        }
    }


    if (!gpu_manager.isEmpty()) {
        int remaining = gpu_manager.getCurrentBatchCount();
        int batch_start_idx = n - remaining;

        std::cout << "Flushing final partial batch of " << remaining << "..." << std::endl;
        gpu_manager.executeBatch();

        for (int b = 0; b < remaining; b++)
        {
            int vector_id = batch_start_idx + b;
            std::vector<float> dists = gpu_manager.getResultsForQuery(b);
            std::vector<int> ids = gpu_manager.getIdsForQuery(b);
            alg_hnsw->linkBatchElement(vector_id, ids, dists);
        }
        
    }

    auto end_time = std::chrono::high_resolution_clock::now();
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
