#include "hybrid_manager.h"
#include "gpu_worker.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <stdexcept>

HybridBatchManager::HybridBatchManager(int dimension, int batch_size, int candidates_count, size_t total_dataset_vectors) 
    : dim(dimension), max_batch_size(batch_size), candidates_per_query(candidates_count), current_batch_count(0) {
        h_query_batch.resize(max_batch_size * dim);
        h_candidate_ids.resize(max_batch_size * candidates_per_query);
        h_distances_out.resize(max_batch_size * candidates_per_query);

        cudaMalloc(&d_dataset_mirror, total_dataset_vectors * dim * sizeof(float));
        cudaMalloc(&d_query_batch, max_batch_size * dim * sizeof(float));
        cudaMalloc(&d_candidate_ids, max_batch_size * candidates_per_query * sizeof(int));
        cudaMalloc(&d_distances_out, max_batch_size * candidates_per_query * sizeof(float));
}

HybridBatchManager::~HybridBatchManager() {
    cudaFree(d_dataset_mirror); 
    cudaFree(d_query_batch);
    cudaFree(d_candidate_ids);
    cudaFree(d_distances_out);
}

void HybridBatchManager::uploadDatasetMirror(const std::vector<float>& full_dataset) {
    cudaMemcpy(d_dataset_mirror, full_dataset.data(), full_dataset.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void HybridBatchManager::queueQuery(int local_index, const float* query_vector, const std::vector<int>& candidate_ids) {
    std::memcpy(&h_query_batch[local_index * dim], query_vector, dim * sizeof(float));
    std::memcpy(&h_candidate_ids[local_index * candidates_per_query], candidate_ids.data(), candidates_per_query * sizeof(int));
}

void HybridBatchManager::executeBatch() {
    if (current_batch_count == 0) return;
    int total_pairs = current_batch_count * candidates_per_query;

    cudaMemcpy(d_query_batch, h_query_batch.data(), current_batch_count * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_candidate_ids, h_candidate_ids.data(), total_pairs * sizeof(int), cudaMemcpyHostToDevice);

    launch_batched_l2_kernel(d_query_batch, d_dataset_mirror, d_candidate_ids, d_distances_out, dim, current_batch_count, candidates_per_query);
    cudaDeviceSynchronize();

    cudaMemcpy(h_distances_out.data(), d_distances_out, total_pairs * sizeof(float), cudaMemcpyDeviceToHost);
    current_batch_count = 0;
}

std::vector<float> HybridBatchManager::getResultsForQuery(int batch_index) {
    std::vector<float> results(candidates_per_query);
    std::memcpy(results.data(), &h_distances_out[batch_index* candidates_per_query], candidates_per_query * sizeof(float));
    return results;
}

std::vector<int> HybridBatchManager::getIdsForQuery(int batch_index) {
    std::vector<int> ids(candidates_per_query);

    // h_candidate_ids should already contain the ids that we queued
    std::memcpy(ids.data(), &h_candidate_ids[batch_index * candidates_per_query], candidates_per_query * sizeof(int));
    return ids;
}