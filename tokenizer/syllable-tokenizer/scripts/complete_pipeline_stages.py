#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def run_cmd(cmd):
    print(f"\n==========================================")
    print(f"Executing: {' '.join(cmd)}")
    print(f"==========================================")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        sys.exit(res.returncode)

def main():
    # Step 1: Run Batch 1 (reuses the existing pool, annotates it if needed,
    # and selects from a bounded streaming candidate reservoir).
    run_cmd([
        sys.executable, "-m", "dataset_builder.pipeline", "run-batch",
        "--batch-id", "1",
        "--target-size", "5000"
    ])
    
    # Step 2: Run Batches 2 to 10
    for b in range(2, 11):
        run_cmd([
            sys.executable, "-m", "dataset_builder.pipeline", "run-batch",
            "--batch-id", str(b),
            "--target-size", "5000"
        ])
        
    # Step 3: Merge all batches into final corpus 50k
    run_cmd([
        sys.executable, "-m", "dataset_builder.pipeline", "merge",
        "--output", "../dataset/asr_corpus/corpus_50k.jsonl"
    ])

    # Step 4: Final Status
    run_cmd([sys.executable, "-m", "dataset_builder.pipeline", "status"])

if __name__ == "__main__":
    main()
