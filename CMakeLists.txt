cmake_minimum_required(VERSION 3.10)
project(HNSW_Hybrid_BEP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# include hnswlib
include_directories(extern/hnswlib/hnswlib)

find_package(OpenMP REQUIRED)

add_executable(hnsw_bench main.cpp)

target_link_libraries(hnsw_bench PUBLIC OpenMP::OpenMP_CXX)