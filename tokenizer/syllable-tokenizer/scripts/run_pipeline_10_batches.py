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
    script_dir = Path(__file__).resolve().parent
    
    # Step 1: Reset
    run_cmd([sys.executable, "-m", "dataset_builder.pipeline", "reset"])
    
    # Step 2: Run Batch 1 (extracts ALL sentences from ALL parquet shards, max_corpus=0 -> unlimited)
    run_cmd([
        sys.executable, "-m", "dataset_builder.pipeline", "run-batch",
        "--batch-id", "1",
        "--target-size", "5000",
        "--max-corpus", "0"
    ])
    
    # Step 3: Run Batches 2 to 10
    for b in range(2, 11):
        run_cmd([
            sys.executable, "-m", "dataset_builder.pipeline", "run-batch",
            "--batch-id", str(b),
            "--target-size", "5000"
        ])
        
    # Step 4: Merge all batches into final corpus 50k
    run_cmd([
        sys.executable, "-m", "dataset_builder.pipeline", "merge",
        "--output", "../dataset/asr_corpus/corpus_50k.jsonl"
    ])

    # Step 5: Final Status
    run_cmd([sys.executable, "-m", "dataset_builder.pipeline", "status"])

if __name__ == "__main__":
    main()
