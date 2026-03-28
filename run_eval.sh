#!/bin/bash

# If no argument is provided, show usage
if [ $# -eq 0 ]; then
    echo "Usage: ./run_eval.sh <index_filename_in_results_indices>"
    echo "Example: ./run_eval.sh sift_hybrid_M16_EF200.bin"
    exit 1
fi

INDEX_FILE="results/indices/$1"

# Check if the file actually exists
if [ ! -f "$INDEX_FILE" ]; then
    echo "Error: $INDEX_FILE not found!"
    exit 1
fi

echo "--- Starting Quality Evaluation ---"
./build/eval_recall "$INDEX_FILE"
echo "--- Evaluation Complete ---"