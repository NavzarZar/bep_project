#include "gpu_worker.cuh"
#include <iostream>

__global__ void batched_l2_warp_kernel(float* queries, float* dataset_mirror, int* candidate_ids, float* results, int dim, int total_pairs, int candidates_per_query) {
    int lane_id = threadIdx.x % 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warp_id < total_pairs) {
        int query_idx = warp_id / candidates_per_query;
        int target_id = candidate_ids[warp_id];

        float partial_sum = 0.0f;
        for (int i = lane_id; i < dim; i += 32) {
            float a = queries[query_idx * dim + i];
            float b = dataset_mirror[target_id * dim + i];
            float diff = a - b;
            partial_sum += diff * diff;
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (lane_id == 0) results[warp_id] = partial_sum;
    }
}

void launch_batched_l2_kernel(float* d_queries, float* d_dataset_mirror, int* d_candidate_ids, float* d_results, int dim, int num_queries, int candidates_per_query) {
    int total_pairs = num_queries * candidates_per_query;
    int threadsPerBlock = 256;
    int pairsPerBlock = threadsPerBlock / 32;
    int blocksPerGrid = (total_pairs + pairsPerBlock - 1) / pairsPerBlock;

    batched_l2_warp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_queries, d_dataset_mirror, d_candidate_ids, d_results, dim, total_pairs, candidates_per_query);
}