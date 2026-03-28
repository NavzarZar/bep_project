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
    int batch_size = 10000;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_construction);

    std::cout << "GPU Manager..." << std::endl;
    HybridBatchManager gpu_manager(dim, batch_size, ef_construction, n);
    gpu_manager.uploadDatasetMirror(data);

    std::cout << "Running the test batch..." << std::endl;

    // mock input vectors
    for (int i = 0; i < n; i++)
    {
        std::cout << "Input: " << i << std::endl;
        alg_hnsw->addPointHybridBatch(data.data() + i * dim, i,  &gpu_manager);

        if (gpu_manager.isFull()) {
            std::cout << "Batch is full now, starting gpu work" << std::endl;
            gpu_manager.executeBatch();

            // check results of first query just to check
            std::vector<float> res = gpu_manager.getResultsForQuery(0);
            std::cout << "It worked! Distance to candidate 0: " << res[0] << std::endl;

            break;
        } 
    }

    delete alg_hnsw;
    return 0;
}
