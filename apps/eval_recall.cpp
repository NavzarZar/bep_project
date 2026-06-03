#include <iostream>
#include <vector>
#include <set>
#include "hnswlib.h"
#include "data_loader.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./eval_recall <path_to_index.bin> <path_to_query.fvecs> <path_to_groundtruth.ivecs>" << std::endl;
        return 1;
    }

    std::string index_path = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];
    size_t n_q, dim, n_gt, k_gt;

    std::cout << "Loading queries and ground truth..." << std::endl;
    auto queries = load_fvecs(query_path, n_q, dim);
    auto gt = load_ivecs(gt_path, n_gt, k_gt);

    hnswlib::L2Space space(dim);
    std::cout << "Loading Index: " << index_path << std::endl;
    hnswlib::HierarchicalNSW<float> alg(&space, index_path);

    // ef=100 is standard for testing SIFT1M recall
    alg.setEf(100);

    int top1_correct = 0;
    float total_recall_at_10 = 0;

    std::cout << "Calculating Recall..." << std::endl;
    for (size_t i = 0; i < n_q; i++) {
        // Search for top 10
        auto result = alg.searchKnn(queries.data() + i * dim, 10);
        
        std::set<int> neighbors;
        while(!result.empty()) {
            neighbors.insert(result.top().second);
            result.pop();
        }

        // Check Recall@1
        if (neighbors.count(gt[i * k_gt])) top1_correct++;

        // Check Recall@10
        int matches = 0;
        for (int j = 0; j < 10; j++) {
            if (neighbors.count(gt[i * k_gt + j])) matches++;
        }
        total_recall_at_10 += (matches / 10.0f);
    }

    std::cout << "------------------------------------" << std::endl;
    std::cout << "Results for " << index_path << ":" << std::endl;
    std::cout << "Recall@1:  " << (float)top1_correct / n_q << std::endl;
    std::cout << "Recall@10: " << total_recall_at_10 / n_q << std::endl;
    std::cout << "------------------------------------" << std::endl;

    return 0;
}