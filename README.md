# Qwen2.5-0.5B on GSM8K (Quickstart)

- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`
- 数据集：`gsm8k`（HuggingFace Hub）
> **注意**：下面所有命令里的 `<username>` 请替换成你的用户名。

---

# 1. 仓库结构

  克隆后，主要结构如下：
  /data/<username>/quickstart/
  ├── environment.yml              # conda 环境（名字：quickstart）
  ├── README.md
  └── scripts/
      ├── prepare_gsm8k.py         # 下载并准备 GSM8K 数据集
      └── run_quickstart.py        # 启动 PPO 训练，写日志 & ckpt
  
  运行过程中会自动在这些地方生成文件：
  /data/<username>/datasets/gsm8k/   # prepare_gsm8k.py 下载的数据
      ├── train.parquet
      └── test.parquet
  
  /data/<username>/quickstart/logs/  # 训练日志
  /data/<username>/quickstart/checkpoints/  # 训练产生的 checkpoints

---

# 2. 克隆代码 & 创建 conda 环境
这一步只需要做一次。

  cd /data/<username>
  
  1.克隆仓库：
  git clone https://github.com/REMISEE/quickstart.git quickstart
  cd quickstart
  
  2.加载 conda 初始化脚本：
  source /data/conda_a100/ourconda_bashrc
  
  3. 使用仓库里的 environment.yml 创建环境：
  conda env create -f environment.yml
  
  4.激活环境：
  conda activate quickstart

  ---

# 3.下载 GSM8K 数据集
  cd /data/<username>/quickstart
  conda activate quickstart
  
  运行数据准备脚本：
  python scripts/prepare_gsm8k.py
  
  运行前手动设置 HF cache（可选）：
  export HF_HOME=/data/$USER/hf_cache
  export HF_DATASETS_CACHE=$HF_HOME/datasets
  export TRANSFORMERS_CACHE=$HF_HOME/transformers

  ---

# 4. 训练：运行 PPO Quickstart
  cd /data/<username>
  1) 加载 conda
  source /data/conda_a100/ourconda_bashrc
  
  2) 激活环境
  conda activate quickstart
  
  3) 进入项目
  cd quickstart
  
  4) 选择显卡，启动训练
  更改run_quickstart.py 中的 export CUDA_VISIBLE_DEVICES=0  # 例如只用第 0 块卡
  python scripts/run_quickstart.py
