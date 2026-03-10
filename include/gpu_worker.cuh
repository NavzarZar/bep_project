#ifndef GPU_WORKER_CUH
#define GPU_WORKER_CUH

#include <cuda_runtime.h>

/**
 * this i s just a wrapper to start the l2 distance kernel
 * d_queries -> pointer to query vectors in vram
 * d_targets -> pointer to target vectors in vram
 * d_results -> pointer where results will be stored in vram
 * dim -> the dimensionality
 * num_pairs how many distances we are doing in this batch
 */
void launch_l2_kernel(float* d_queries, float* d_targets, float* d_results, int dim, int num_pairs);

#endif