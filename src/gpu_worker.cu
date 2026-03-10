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

__global__ void l2_distance_warp_kernel(float* queries, float* targets, float* results, int dim, int num_pairs) {
    // where am i in the warp
    // get the lane id 
    int lane_id = threadIdx.x % 32;

    // which vector pair is my warp working on?
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;


    if (warp_id < num_pairs) {
        float partial_sum = 0.0f;

        // all 32 threads read the memory next to eachother?
        // then they step by 32
        for (int i = lane_id; i < dim; i += 32) {
            float a = queries[warp_id * dim + i];
            float b = targets[warp_id * dim + i];
            float diff = a - b;
            partial_sum += diff * diff;
        }

        // warp shuffle reduction
        // we fold the 32 sums into a single total sum
        for (int offset = 16; offset > 0; offset /= 2) {
            // shuffl down sync gets the value from the thread that is offset lanes away
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
        }

        if (lane_id == 0) {
            results[warp_id] = partial_sum;
        }
    }


}

// this should run on the CPU to start the GPU
void launch_l2_kernel(float* d_queries, float* d_targets, float* d_results, int dim, int num_pairs) {
    int threadsPerBlock = 256; 

    // 32 threads = 1 pair
    // so a block of 256 threads only does 8 pairs
    int pairsPerBlock = threadsPerBlock / 32;

    // get blocks based on the new ratio
    int blocksPerGrid = (num_pairs + pairsPerBlock - 1) / pairsPerBlock;

    l2_distance_warp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_queries, d_targets, d_results, dim, num_pairs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel problem: " << cudaGetErrorString(err) << std::endl;
    }

}