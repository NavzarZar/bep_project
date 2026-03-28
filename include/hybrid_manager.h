#pragma once
#include <vector>

class HybridBatchManager {
private:
    int dim;
    int max_batch_size;
    int current_batch_count;

    // vram pointers
    float* d_dataset_mirror;
    float* d_query_batch;
    int* d_candidate_ids;
    float* d_distances_out;

    // ram buffers
    std::vector<float> h_query_batch;
    std::vector<int> h_candidate_ids;
    std::vector<float> h_distances_out;

public:
    int candidates_per_query;

    HybridBatchManager(int dimension, int batch_size, int candidates_count, size_t total_dataset_vectors);
    ~HybridBatchManager();

    void uploadDatasetMirror(const std::vector<float>& full_dataset);
    void queueQuery(const float* query_vector, const std::vector<int>& candidate_ids);
    void executeBatch();

    bool isEmpty() const { return current_batch_count == 0; }
    bool isFull() const { return current_batch_count >= max_batch_size; }
    std::vector<float> getResultsForQuery(int batch_index);
    std::vector<int> getIdsForQuery(int batch_index);
    int getCurrentBatchCount() const { return current_batch_count; }

};