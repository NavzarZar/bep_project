#!/bin/bash
echo "--- HNSW BENCHMARK SUITE ---" | tee results.txt

echo "RUNNING BASELINE..." | tee -a results.txt
./build/cpu_baseline | tee -a results.txt

echo -e "\nRUNNING HYBRID..." | tee -a results.txt
./build/hybrid_main | tee -a results.txt

echo "Benchmarks complete. Results saved to results.txt"