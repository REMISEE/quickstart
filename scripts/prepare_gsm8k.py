#!/usr/bin/env python
import os
from pathlib import Path

from datasets import load_dataset


def main():
    # repo_root = /data/<user>/quickstart
    repo_root = Path(__file__).resolve().parents[1]

    # datasets_root = /data/<user>/datasets
    datasets_root = repo_root.parent / "datasets"
    gsm8k_dir = datasets_root / "gsm8k"

    gsm8k_dir.mkdir(parents=True, exist_ok=True)

    train_path = gsm8k_dir / "train.parquet"
    test_path = gsm8k_dir / "test.parquet"

    print(f"[INFO] Repo root:        {repo_root}")
    print(f"[INFO] Datasets root:    {datasets_root}")
    print(f"[INFO] GSM8K dir:        {gsm8k_dir}")

    if train_path.exists() and test_path.exists():
        print("[INFO] GSM8K parquet already exists, skip downloading.")
        print(f"       {train_path}")
        print(f"       {test_path}")
        return

    print("[INFO] Downloading GSM8K (gsm8k, main) from HuggingFace...")
    ds = load_dataset("gsm8k", "main")

    print("[INFO] Saving to parquet...")
    ds["train"].to_parquet(train_path)
    ds["test"].to_parquet(test_path)

    print("[INFO] Done.")
    print(f"       train.parquet: {train_path}")
    print(f"       test.parquet : {test_path}")


if __name__ == "__main__":
    main()
