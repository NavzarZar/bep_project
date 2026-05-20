import numpy as np 
import os
import time
import argparse
import faiss

def write_fvecs(filename, data):
    N, D = data.shape

    dt = np.dtype([('dim', np.int32), ('vec', np.float32, D)])
    arr = np.empty(N, dtype=dt)
    arr['dim'] = D
    arr['vec'] = data
    arr.tofile(filename)

def write_ivecs(filename, data):
    N, K = data.shape
    dt = np.dtype([('dim', np.int32), ('vec', np.int32, K)])
    arr = np.empty(N, dtype=dt)
    arr['dim'] = K
    arr['vec'] = data
    arr.tofile(filename)

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate synthetic fvecs/ivecs datasets.")
    parser.add_argument('--dim', type=int, default=768, help='Dimensionality of the vectors (default: 768)')
    args = parser.parse_args()
    
    DIM = args.dim

    os.makedirs("data", exist_ok=True)
    rng = np.random.default_rng(seed=42)

    print(f"Generating 1,000,000 base vectors ({DIM}-D)...")
    base_data = rng.random((1000000, DIM), dtype=np.float32)
    write_fvecs(f"data/synthetic_{DIM}d_base.fvecs", base_data)

    print(f"Generating 10,000 query vectors ({DIM}-D)...")
    query_data = rng.random((10000, DIM), dtype=np.float32)
    write_fvecs(f"data/synthetic_{DIM}d_query.fvecs", query_data)

    print(f"\nCalculating exact ground truth for {DIM}-D...")
    start_time = time.time()

    index = faiss.IndexFlatL2(DIM)

    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print("Running on GPU!")
    except AttributeError:
        print("Running on CPU!")
    
    index.add(base_data)
    distances, indices = index.search(query_data, 100)

    print(f"Ground truth calculated in {time.time() - start_time:.2f} seconds.")

    print("Saving ground truth to binary...")
    write_ivecs(f"data/synthetic_{DIM}d_groundtruth.ivecs", indices.astype(np.int32))

    print("Done! Data saved to the /data folder.")

if __name__ == "__main__":
    main()