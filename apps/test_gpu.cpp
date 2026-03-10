#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "gpu_worker.cuh"

int main() {
    int dim = 128;       
    int num_pairs = 1000000; 

    // host data
    std::vector<float> h_queries(num_pairs * dim, 1.0f); 
    std::vector<float> h_targets(num_pairs * dim, 2.0f);
    std::vector<float> h_results(num_pairs, 0.0f);

    // device data
    float *d_queries, *d_targets, *d_results;
    cudaMalloc(&d_queries, num_pairs * dim * sizeof(float));
    cudaMalloc(&d_targets, num_pairs * dim * sizeof(float));
    cudaMalloc(&d_results, num_pairs * sizeof(float));

    // cpu to gpu copying
    cudaMemcpy(d_queries, h_queries.data(), num_pairs * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets.data(), num_pairs * dim * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Launching kernel......" << std::endl;

    // start the timer 
    auto start = std::chrono::high_resolution_clock::now();

    launch_l2_kernel(d_queries, d_targets, d_results, dim, num_pairs);

    // sync or the cpu would stop before the gpu is done
    cudaDeviceSynchronize();


    // stop timer
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // gpu to cpu copying
    cudaMemcpy(h_results.data(), d_results, num_pairs * sizeof(float), cudaMemcpyDeviceToHost);

    // just to check
    std::cout << "GPU Result [0]: " << h_results[0] << " (Expected: 128.0)" << std::endl;
    std::cout << "GPU Result [99999]: " << h_results[99999] << " (Expected: 128.0)" << std::endl;
    
    double seconds = duration.count() / 1000000.0;
    std::cout << "GPU results ---" << std::endl;
    std::cout << "Time: " << seconds << " seconds" << std::endl;
    std::cout << "Throughput: " << (num_pairs / seconds) << " distances/sec" << std::endl;
    
    // cleaning
    cudaFree(d_queries); cudaFree(d_targets); cudaFree(d_results);
    
    return 0;
}