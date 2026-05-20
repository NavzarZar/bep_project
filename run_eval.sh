#!/bin/bash

# Enforce exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_eval.sh <index_file> <query_file> <groundtruth_file>"
    echo "Example: ./run_eval.sh results/indices/sift1m_baseline_M16_EF200.bin data/sift_query.fvecs data/sift_groundtruth.ivecs"
    exit 1
fi

INDEX_FILE=$1
QUERY_FILE=$2
GT_FILE=$3

# Verify the index exists before running
if [ ! -f "$INDEX_FILE" ]; then
    echo "Error: Index file '$INDEX_FILE' not found!"
    exit 1
fi

echo "--- Starting Quality Evaluation ---"
echo "Index: $INDEX_FILE"
echo "Query: $QUERY_FILE"
echo "GT:    $GT_FILE"

./build/eval_recall "$INDEX_FILE" "$QUERY_FILE" "$GT_FILE"

echo "--- Evaluation Complete ---"