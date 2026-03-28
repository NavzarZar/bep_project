#ifndef GPU_WORKER_CUH
#define GPU_WORKER_CUH

#include <cuda_runtime.h>

void launch_batched_l2_kernel(
    float* d_queries, 
    float* d_dataset_mirror, 
    int* d_candidate_ids, 
    float* d_results, 
    int dim, 
    int num_queries, 
    int candidates_per_query
);

#endif