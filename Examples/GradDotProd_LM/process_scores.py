import argparse
import glob
import os
import re

import numpy as np
import torch
from tqdm import tqdm


def parse_iter(filename):
    """Extract iteration number, handling -1 correctly for sorting."""
    match = re.search(r"iter_(-?\d+)", filename)
    if match:
        return int(match.group(1))
    return -float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the specific results folder",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    logs_dir = os.path.join(args.results_dir, "grad_dotprods")
    if args.output_dir is None:
        args.output_dir = args.results_dir

    print(f"Loading logs from: {logs_dir}")
    log_files = glob.glob(os.path.join(logs_dir, "dot_prod_log_iter_*.pt"))

    log_files = sorted(log_files, key=parse_iter)

    if not log_files:
        print("No log files found!")
        return

    score_accumulator = {}

    for log_file in tqdm(log_files, desc="Aggregating scores"):
        try:
            batch_list = torch.load(log_file, map_location="cpu")
            for entry in batch_list:
                scores = entry["dot_product"].numpy()
                indices = entry["batch_idx"].numpy()

                for idx, score in zip(indices, scores):
                    if idx not in score_accumulator:
                        score_accumulator[idx] = 0.0
                    score_accumulator[idx] += score

        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    print(f"Total unique samples scored: {len(score_accumulator)}")

    positive_indices = []
    for idx, total_score in score_accumulator.items():
        if total_score >= 0:  # Keep only non-negative
            positive_indices.append(idx)

    positive_indices = np.array(positive_indices, dtype=np.int64)
    print(
        f"Identified {len(positive_indices)} positive samples "
        f"({len(positive_indices) / len(score_accumulator) * 100:.2f}% of seen data)"
    )

    output_path = os.path.join(args.output_dir, "positive_indices.npy")
    np.save(output_path, positive_indices)
    print(f"Saved positive indices (Allowlist) to: {output_path}")


if __name__ == "__main__":
    main()
