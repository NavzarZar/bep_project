#include "gpu_worker.cuh"
#include <iostream>

// The kernel
__global__ void l2_distance_kernel(float* queries, float* targets, float* results, int dim, int num_pairs) {
    // Get which pair of vectors this thread has to use
    // blockidx (whch block am i in) * blockDim (how big is a block) + threadIdx (my id in the block)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pairs) {
        float sum = 0.0f;

        // l2 distance
        for (int i = 0; i < dim; i++)
        {
            float a = queries[idx * dim + i];
            float b = targets[idx * dim + i];
            float diff = a - b;
            sum += diff * diff;
        }

        // write results in vram
        results[idx] = sum;
        
    }
}

// this should run on the CPU to start the GPU
void launch_l2_kernel(float* d_queries, float* d_targets, float* d_results, int dim, int num_pairs) {
    int threadsPerBlock = 256; // random number for now

    // how many blocks we need to cover all pairs
    int blocksPerGrid = (num_pairs + threadsPerBlock - 1) / threadsPerBlock;
    l2_distance_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_queries, d_targets, d_results, dim, num_pairs);

    // Sync for debugging
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cuda kernel had an issue: " << cudaGetErrorString(err) << std::endl;
    }
}