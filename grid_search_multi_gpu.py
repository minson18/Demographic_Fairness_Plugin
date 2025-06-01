import itertools
import subprocess
import os
import json
import csv
import sys
from datetime import datetime
import time

"""
Must set following environment variables to run grid search:
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
"""


def run_grid_search_multi_gpu(param_grid, base_cmd, save_dir_prefix, num_gpus=None):
    # Read CUDA_VISIBLE_DEVICES from environment
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_env:
        gpu_list = [int(x) for x in cuda_env.split(",") if x.strip()]
    else:
        gpu_list = [0, 1, 2, 3]  # default
    if num_gpus is None:
        num_gpus = len(gpu_list)
    results = []
    procs = []
    proc_infos = []
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_jobs = len(combinations)
    completed_jobs = 0
    for i, params in enumerate(combinations):
        gpu_id = gpu_list[i % num_gpus]
        save_dir = f"{save_dir_prefix}/grid_{i}"
        os.makedirs(save_dir, exist_ok=True)
        cmd = base_cmd + ["--model_dir", save_dir]
        for k, v in params.items():
            cmd += [f"--{k}", str(v)]
        print(f"[Job {i+1}/{total_jobs}] Launching on GPU {gpu_id}: {params}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Only expose one GPU per process
        env["SELECTED_GPU_INDEX"] = str(gpu_id)  # for debugging, not used by torch
        log_path = os.path.join(save_dir, "job.log")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
        procs.append((proc, log_file))
        proc_infos.append((i, params, save_dir, proc, log_file))
        # If we've launched num_gpus jobs, wait for all to finish
        if len(procs) == num_gpus or i == len(combinations) - 1:
            for j, (idx, params, save_dir, proc, log_file) in enumerate(proc_infos):
                proc.wait()
                log_file.close()
                metrics_path = os.path.join(save_dir, "val_metrics.json")
                all_metrics = {}
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        all_metrics = json.load(f)
                    best_ndcg = all_metrics.get("ndcg@20", None)
                else:
                    best_ndcg = None
                results.append(
                    {
                        "params": params,
                        "ndcg@20": best_ndcg,
                        "all_metrics": all_metrics,
                        "save_dir": save_dir,
                    }
                )
                print(
                    f"[Job {idx+1}/{total_jobs}] Finished. Params: {params}, Best NDCG@20: {best_ndcg}"
                )
            completed_jobs += len(proc_infos)
            print(f"Progress: {completed_jobs}/{total_jobs} jobs completed.")
            procs = []
            proc_infos = []
            # Optional: short sleep to avoid race conditions
            time.sleep(2)
    return results


def save_results(results, param_grid, out_dir):
    with open(os.path.join(out_dir, "grid_search_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    csv_header = list(param_grid.keys()) + ["ndcg@20", "all_metrics", "save_dir"]
    with open(os.path.join(out_dir, "grid_search_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for entry in results:
            row = [entry["params"].get(k, None) for k in param_grid.keys()] + [
                entry["ndcg@20"],
                json.dumps(entry["all_metrics"]),
                entry["save_dir"],
            ]
            writer.writerow(row)


def retrain_and_test_best(
    best_params, base_cmd, save_dir_prefix, dataset, test_py_path
):
    retrain_dir = os.path.join(save_dir_prefix, "best_retrain")
    os.makedirs(retrain_dir, exist_ok=True)
    retrain_cmd = base_cmd + ["--model_dir", retrain_dir]
    for k, v in best_params.items():
        retrain_cmd += [f"--{k}", str(v)]
    # Save best params as JSON
    best_params_path = os.path.join(retrain_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Retraining best model: {' '.join(retrain_cmd)}")
    retrain_log_path = os.path.join(retrain_dir, "job.log")
    with open(retrain_log_path, "w") as retrain_log_file:
        subprocess.run(
            retrain_cmd, stdout=retrain_log_file, stderr=subprocess.STDOUT, text=True
        )
    # Test the retrained model
    test_cmd = [
        sys.executable,
        test_py_path,
        "--model_dir",
        retrain_dir,
        "--dataset",
        dataset,
    ]
    # Pass best params to test script as well
    for k, v in best_params.items():
        test_cmd += [f"--{k}", str(v)]
    print(f"Testing best model: {' '.join(test_cmd)}")
    test_log_path = os.path.join(retrain_dir, "test.log")
    with open(test_log_path, "w") as test_log_file:
        subprocess.run(
            test_cmd, stdout=test_log_file, stderr=subprocess.STDOUT, text=True
        )


def main(num_gpus=4):
    # Create a unique timestamped directory for this grid search
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir = f"saved_models/gridsearch_{timestamp}"
    os.makedirs(unique_dir, exist_ok=True)

    param_grid = {
        "lr": [0.0001, 0.001],
        "hidden_units": [64, 128],
        "num_blocks": [3],
        "dropout_rate": [0.3, 0.5],
        "batch_size": [64, 128, 256],
        "maxlen": [100],
        "num_heads": [1],
    }
    base_cmd = [
        sys.executable,
        "train.py",
        "--dataset",
        "ml-1m",
        # batch_size and maxlen are now tuned, do not hardcode
        "--cxt_size",
        "6",
        "--num_epochs",
        "50",
        "--device",
        "cuda",
    ]
    save_dir_prefix = unique_dir
    test_py_path = "test.py"
    dataset = "ml-1m"
    # 1. Run grid search (multi-GPU)
    results = run_grid_search_multi_gpu(
        param_grid, base_cmd, save_dir_prefix, num_gpus=num_gpus
    )
    # 2. Save all results
    save_results(results, param_grid, unique_dir)
    # 3. Find best params
    best_entry = max(
        results, key=lambda x: x["ndcg@20"] if x["ndcg@20"] is not None else -1
    )
    best_params = best_entry["params"]
    print(f"Best hyperparameters: {best_params}")
    # 4. Retrain best and test
    retrain_and_test_best(best_params, base_cmd, save_dir_prefix, dataset, test_py_path)


if __name__ == "__main__":
    main(num_gpus=4)
