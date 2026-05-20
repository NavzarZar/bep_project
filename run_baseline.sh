#!/bin/bash
set -e

echo "Starting CPU Baseline Experiment: SIFT..."
./build/cpu_baseline data/sift_base.fvecs 16 200

echo "Starting CPU Baseline Experiment: 768D..."
./build/cpu_baseline data/synthetic_768d_base.fvecs 16 200

echo "CPU baseline experiments done!"