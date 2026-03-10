cmake_minimum_required(VERSION 3.18)
project(HNSW_Hybrid_BEP LANGUAGES CXX CUDA) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# cpu flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

# gpu flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_86,code=sm_86")

# include hnswlib
include_directories(extern/hnswlib/hnswlib)
include_directories(include)

find_package(OpenMP REQUIRED)

add_library(hnsw_core
    src/data_loader.cpp
    src/gpu_worker.cu
)
target_link_libraries(hnsw_core PUBLIC OpenMP::OpenMP_CXX)

# build cpu baseline
add_executable(cpu_baseline apps/cpu_baseline.cpp)
target_link_libraries(cpu_baseline PRIVATE hnsw_core)

# build gpu test for now
add_executable(test_gpu apps/test_gpu)
target_link_libraries(test_gpu PRIVATE hnsw_core)