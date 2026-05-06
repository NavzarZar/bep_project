#!/bin/bash
set -e

# Default parameters
M=${1:-16}
EF_CON=${2:-200}
BATCH=${3:-50000}
CANDS=${4:-3000}
BEAM=${5:-30}

echo "=========================================="
echo " Running Hybrid HNSW with Parameters:"
echo " M: $M | EF_CON: $EF_CON | BATCH: $BATCH | CANDS: $CANDS | BEAM: $BEAM"
echo "=========================================="

./build/hybrid_main $M $EF_CON $BATCH $CANDS $BEAM

INDEX_NAME="results/indices/sift1m_hybrid_M${M}_EF${EF_CON}_B${BATCH}_C${CANDS}_BM${BEAM}.bin"
./build/eval_recall $INDEX_NAME