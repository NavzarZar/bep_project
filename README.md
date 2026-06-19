# Hybrid CPU-GPU HNSW Architecture

This repository contains the implementation of a decoupled CPU-GPU architecture for dynamic vector database ingestion using the Hierarchical Navigable Small World (HNSW) algorithm. The architecture utilizes the CPU for sequential graph topology management (routing and linking) and the GPU for batched parallel distance calculations.

## Prerequisites

To compile and run this project, ensure you have the following installed on your system:
* **C++ Compiler**: GCC with OpenMP support (`-fopenmp`)
* **CUDA Toolkit**: `nvcc` for compiling the GPU distance kernels (configured for `sm_86` architecture by default)
* **CMake**: Version 3.18 or higher
* **NVIDIA Nsight Systems (`nsys`)**: Optional, but highly recommended for hardware profiling

## Compilation

A compilation script is provided for convenience. Simply run:

```bash
./compile.sh
```

## Usage

### Hybrid CPU-GPU Pipeline (`hybrid_main`)
The hybrid pipeline executes the decoupled insertion algorithm.

**Execution Format:**
```bash
./build/hybrid_main <M> <ef_construction> <batch_size> <candidates_per_query> <beam_search_ef> <seed_size> [path_to_data.fvecs]
```
* **M**: Maximum number of outgoing connections for each node.
* **ef_construction**: Size of the dynamic candidate list during graph construction.
* **batch_size**: Number of vectors processed concurrently by the GPU.
* **candidates_per_query**: Number of neighbor IDs gathered by the CPU BFS and sent to the GPU.
* **beam_search_ef**: Depth of the CPU's local distance evaluation prior to offloading.
* **seed_size**: Number of vectors inserted sequentially on the CPU to build the initial routing foundation (set to 0 for unseeded).
* **path_to_data.fvecs**: Path to the base dataset. Defaults to data/synthetic_768d_base.fvecs.

**Example:**
```bash
    ./build/hybrid_main 16 200 50000 3000 30 200000 data/synthetic_768d_base.fvecs
```
### CPU Baseline (cpu_baseline)
The baseline establishes the CPU-only performance ceiling, utilizing a fully parallelized OpenMP implementation of the standard HNSW insertion pipeline.

**Execution Format:**
```bash
    ./build/cpu_baseline <path_to_data.fvecs> <M> <ef_construction>
```
**Example:**
```bash
    ./build/cpu_baseline data/sift_base.fvecs 16 200
```
### Evaluate Recall (eval_recall)
This tool loads a saved .bin index and evaluates its structural accuracy (Recall@1 and Recall@10) against a provided ground truth.

**Execution Format:**
```bash
    ./run_eval.sh <path_to_index.bin> <path_to_query.fvecs> <path_to_groundtruth.ivecs>
```

**Example:**
```bash
    ./eval_recall.sh results/indices/temp_hybrid_index.bin data/sift_query.fvecs data/sift_groundtruth.ivecs
```

## Hardware Profiling
The hybrid_main executable is instrumented with NVTX markers. To generate an execution trace and analyze the phase distribution (CPU overhead vs. GPU compute vs. PCIe transfer), run the hybrid executable through NVIDIA Nsight Systems:
```bash
    nsys profile -t cuda,osrt,nvtx -o hybrid_profile ./build/hybrid_main 16 200 50000 3000 10 200000 data/synthetic_768d_base.fvecs
```