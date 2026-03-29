import subprocess
import itertools
import csv
import re
import time
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define parameters
params = {
    'M': [16], 
    'ef_construction': [100, 200, 400],
    'batch_size': [1000, 10000, 50000],
    'candidates_per_query': [3000],
    'beam_search_ef': [1, 10, 30] 
}

# Generate all permutations
keys, values = zip(*params.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

csv_filename = 'results/hybrid_benchmark_results.csv'

# regex that should extract it from the c++ output
regex_time = re.compile(r"Baseline build time:\s+([0-9.]+)s")
regex_throughput = re.compile(r"Baseline Throughput:\s+([0-9.]+)\s+vectors/sec")
regex_recall_1 = re.compile(r"Recall@1:\s+([0-9.]+)")
regex_recall_10 = re.compile(r"Recall@10:\s+([0-9.]+)")

print(f"Starting benchmark suite. Total experiments to run: {len(experiments)}")

# Open CSV for writing
os.makedirs('results', exist_ok=True)
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['M', 'ef_construction', 'batch_size', 'candidates_per_query', 
                     'beam_search_ef', 'build_time_sec', 'throughput_vec_sec', 'recall_1', 'recall_10'])

    for idx, exp in enumerate(experiments):
        print(f"\n--- Running Experiment {idx+1}/{len(experiments)} ---")
        print(f"Params: {exp}")

        index_name = f"sift1m_hybrid_M{exp['M']}_EF{exp['ef_construction']}_B{exp['batch_size']}_C{exp['candidates_per_query']}_BM{exp['beam_search_ef']}.bin"     

        # Build the CLI command for hybrid_main
        build_cmd = [
            './build/hybrid_main', 
            str(exp['M']), 
            str(exp['ef_construction']), 
            str(exp['batch_size']), 
            str(exp['candidates_per_query']), 
            str(exp['beam_search_ef'])
        ]

        # run the build
        print("  -> Building Graph...")
        build_result = subprocess.run(build_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        
        build_time, throughput = "ERROR", "ERROR"
        if build_result.returncode == 0:
            match_time = regex_time.search(build_result.stdout)
            match_throughput = regex_throughput.search(build_result.stdout)
            if match_time: build_time = match_time.group(1)
            if match_throughput: throughput = match_throughput.group(1)
        else:
            print(f"  [!] Build crashed with exit code {build_result.returncode}")
            print("  --- C++ ERROR LOG ---")
            print(build_result.stderr.strip() if build_result.stderr else "No STDERR output.")
            print(build_result.stdout.strip())
            print("  ---------------------")
            continue

        # run the Evaluation Step
        print("  -> Evaluating Recall...")
        eval_cmd = ['./build/eval_recall', f"results/indices/{index_name}"]
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        recall_1, recall_10 = "ERROR", "ERROR"
        if eval_result.returncode == 0:
            match_r1 = regex_recall_1.search(eval_result.stdout)
            match_r10 = regex_recall_10.search(eval_result.stdout)
            if match_r1: recall_1 = match_r1.group(1)
            if match_r10: recall_10 = match_r10.group(1)

        print(f"  -> Results: Time={build_time}s | Tput={throughput} | R@10={recall_10}")

        # delete the big binaries, resource exhaustion
        index_file_path = os.path.join(PROJECT_ROOT, 'results', 'indices', index_name)
        if os.path.exists(index_file_path):
            os.remove(index_file_path)
            print("Cleaned up index files")

        # save to csv 
        writer.writerow([exp['M'], exp['ef_construction'], exp['batch_size'], 
                         exp['candidates_per_query'], exp['beam_search_ef'], 
                         build_time, throughput, recall_1, recall_10])
        
        file.flush()


print(f"\nBenchmark suite complete! Data saved to {csv_filename}")