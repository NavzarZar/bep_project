#include <iostream>
#include <vector>
#include "hnswlib.h"
#include "data_loader.h"
#include <hybrid_manager.h>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char** argv) {

    nvtxRangePushA("Setup: Load data & initialize gpu");

    size_t n = 0, dim = 0;
    std::string base_path = "data/sift_base.fvecs";

    // Default parameters
    int M = 16;
    int ef_construction = 200;
    int batch_size = 50000;
    int candidates_per_query = 3000;
    int beam_search_ef = 30;
    int seed_size = 200000;

    if (argc >= 7) {
        M = std::stoi(argv[1]);
        ef_construction = std::stoi(argv[2]);
        batch_size = std::stoi(argv[3]);
        candidates_per_query = std::stoi(argv[4]);
        beam_search_ef = std::stoi(argv[5]);
        seed_size = std::stoi(argv[6]);
    }

    std::cout << "--- HYBRID PARAMS ---" << std::endl;
    std::cout << "M: " << M << ", ef_con: " << ef_construction << ", batch: " << batch_size 
              << ", cands: " << candidates_per_query << ", beam_ef: " << beam_search_ef << std::endl;

    std::cout << "Loading SIFT..." << std::endl;
    auto data = load_fvecs(base_path, n, dim);


    if (n == 0) {
        std::cerr << "Error, loaded data is null. Check path" << std::endl;
        return 1;
    }


    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, M, ef_construction);

    std::cout << "GPU Manager..." << std::endl;
    HybridBatchManager gpu_manager(dim, batch_size, candidates_per_query, n);
    gpu_manager.uploadDatasetMirror(data);

    nvtxRangePop();

    auto start_time = std::chrono::high_resolution_clock::now();

    nvtxRangePushA("CPU: Build Seed Graph");
    // build the first 10k on cpu to create a decent graph
    std::cout << "Building seed graph (CPU)..." << std::endl;
    for (int i = 0; i < seed_size; i++) {
        alg_hnsw->addPoint(data.data() + i * dim, i);
    }
    nvtxRangePop();


    std::cout << "Running hybrid insertions..." << std::endl;
    std::vector<int> gpu_batch_ids;
    int current_id = seed_size;

    while (current_id < n) {
        gpu_batch_ids.clear();

        nvtxRangePushA("CPU: Route upper layers & pre register");
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
        nvtxRangePop();


        int current_batch_size = gpu_batch_ids.size();
        if (current_batch_size == 0) break;

        gpu_manager.setCurrentBatchCount(current_batch_size);


        if (current_id % 50000 < batch_size) {
            std::cout << "\nProcessed " << current_id << " / " << n << " vectors..." << std::endl;
        }
        
        nvtxRangePushA("CPU: Gather candidates (BFS)");
        // each cpu core should take a part of current_batch_size
        // lock-free
        #pragma omp parallel for
        for (int local_i = 0; local_i < current_batch_size; local_i++) {
            int vector_id = gpu_batch_ids[local_i];
            alg_hnsw->addPointHybridBatch(data.data() + vector_id * dim, vector_id, &gpu_manager, local_i, beam_search_ef);
        }    
        nvtxRangePop();

        nvtxRangePushA("GPU: Compute distances");
        std::cout << "[Main] Batch gathered. Executing GPU..." << std::endl;
        gpu_manager.executeBatch();
        nvtxRangePop();

        nvtxRangePushA("CPU: Graph linking");
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
        nvtxRangePop();
    }


    auto end_time = std::chrono::high_resolution_clock::now();

    nvtxRangePushA("Post: Diagnostics and save index");
    alg_hnsw->diagnosticAnalyzeTopology(seed_size, n);
    
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Hybrid build time: " << diff.count() << "s" << std::endl;
    std::cout << "Hybrid Throughput: " << n / diff.count() << " vectors/sec" << std::endl;
    

    // save the index, no need to rebuild for every test
    std::string index_path = "results/indices/";
    std::string index_name = "sift1m_hybrid_M" + std::to_string(M) + 
                             "_EF" + std::to_string(ef_construction) + 
                             "_B" + std::to_string(batch_size) + 
                             "_C" + std::to_string(candidates_per_query) + 
                             "_BM" + std::to_string(beam_search_ef) + ".bin";

    alg_hnsw->saveIndex(index_path + index_name);
    std::cout << "Index saved to " << index_path << std::endl;
    nvtxRangePop();


    delete alg_hnsw;
    return 0;
}
