#!/bin/bash

# stop if anything fails
set -e

echo "Cleaning the old build"
rm -rf build
mkdir build
cd build

echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=Debug ..

echo "Compiling with $(nproc) cores..."
make -j$(nproc)

echo "Build successful!"
echo "Running hybrid_main..."

cd ..
gdb ./build/hybrid_main