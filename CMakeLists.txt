cmake_minimum_required(VERSION 3.18)
project(HNSWHybrid LANGUAGES CXX CUDA) 

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
    src/hybrid_manager.cpp
)
target_link_libraries(hnsw_core PUBLIC OpenMP::OpenMP_CXX)

# the hybrid test
add_executable(hybrid_main apps/hybrid_main.cpp)
target_link_libraries(hybrid_main PRIVATE hnsw_core)

# # build cpu baseline
# add_executable(cpu_baseline apps/cpu_baseline.cpp)
# target_link_libraries(cpu_baseline PRIVATE hnsw_core)

# # build gpu test for now
# add_executable(test_gpu apps/test_gpu)
# target_link_libraries(test_gpu PRIVATE hnsw_core)