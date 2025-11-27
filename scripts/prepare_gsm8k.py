#!/usr/bin/env python
import os
from pathlib import Path
from datasets import load_dataset


def main():

    # repo root = /data/<user>/quickstart
    repo_root = Path(__file__).resolve().parents[1]
    user_data_root = repo_root.parent / "datasets"
    gsm8k_dir = user_data_root / "gsm8k"

    gsm8k_dir.mkdir(parents=True, exist_ok=True)

    train_path = gsm8k_dir / "train.parquet"
    test_path = gsm8k_dir / "test.parquet"

    print(f"[INFO] Repo root:        {repo_root}")
    print(f"[INFO] Dataset root:     {user_data_root}")
    print(f"[INFO] GSM8K dir:        {gsm8k_dir}")
    print()

    if train_path.exists() and test_path.exists():
        print("[INFO] GSM8K parquet files already exist. Nothing to download.")
        print(f"       {train_path}")
        print(f"       {test_path}")
        return

    print("[INFO] Downloading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("gsm8k", "main")

    print("[INFO] Saving parquet files...")
    dataset["train"].to_parquet(train_path)
    dataset["test"].to_parquet(test_path)

    print("[INFO] Done!")
    print(f"       train.parquet: {train_path}")
    print(f"       test.parquet:  {test_path}")


if __name__ == "__main__":
    main()
