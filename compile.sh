#!/bin/bash
set -e

mkdir -p build
cd build

echo "--- Configuring Project ---"
cmake -DCMAKE_BUILD_TYPE=Release ..

echo "--- Compiling All Targets ---"
# Using -j$(nproc) to use all CPU cores for compilation
make -j$(nproc)

echo "--- Compilation Finished ---"