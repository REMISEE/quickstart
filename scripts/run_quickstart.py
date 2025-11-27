#!/usr/bin/env python
import datetime
import subprocess
from pathlib import Path


def main() -> None:
    # ============================================
    # 0. 位置与基础路径
    # repo_root: /data/<name>/quickstart
    # ============================================
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    print(f"[INFO] REPO_ROOT  = {repo_root}")
    print(f"[INFO] SCRIPT_DIR = {script_dir}")

    data_dir = repo_root.parent / "datasets" / "gsm8k"

    log_dir = repo_root / "logs"
    ckpt_dir = repo_root / "checkpoints"

    ckpt_project_name = "gsm8k_ppo"
    experiment_name = "qwen0_5b_quickstart"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ============================================
    # 1. 数据检查
    # ============================================
    train_file = data_dir / "train.parquet"
    val_file = data_dir / "test.parquet"

    if not train_file.is_file():
        raise FileNotFoundError(f"[ERROR] train.parquet 不存在: {train_file}")

    if not val_file.is_file():
        raise FileNotFoundError(f"[ERROR] test.parquet 不存在: {val_file}")

    # ============================================
    # 2. 日志文件
    # ============================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gsm8k_ppo_train_{timestamp}.log"

    print(f"[INFO] DATA_DIR   = {data_dir}")
    print(f"[INFO] LOG_FILE   = {log_file}")
    print("[INFO] 开始训练...")

    # ============================================
    # 3. 构造命令
    # ============================================
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        "data.train_batch_size=256",
        "data.max_prompt_length=512",
        "data.max_response_length=512",
        "actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "critic.optim.lr=1e-5",
        "critic.model.path=Qwen/Qwen2.5-0.5B-Instruct",
        "critic.ppo_micro_batch_size_per_gpu=4",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.logger=console",
        "trainer.val_before_train=False",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.project_name={ckpt_project_name}",
        f"trainer.experiment_name={experiment_name}",
        "trainer.save_freq=10",
        "trainer.test_freq=10",
        "trainer.total_epochs=15",
    ]

    # ============================================
    # 4. 启动训练并同时写 log（类似 2>&1 | tee）
    # ============================================
    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()

        if process.returncode != 0:
            raise SystemExit(process.returncode)

    print("[INFO] 训练结束。")


if __name__ == "__main__":
    main()
