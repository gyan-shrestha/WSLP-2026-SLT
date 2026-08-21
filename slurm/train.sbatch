#!/bin/bash
#SBATCH --job-name=wslp-train
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_ACCOUNT             # normal QOS -- the only one with a GPU
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G          # features are mmapped; 200G just tripped the 250G group QOS cap
#SBATCH --time=12:00:00
#SBATCH --output=${SLT_ROOT}/logs/train-%j.out
#
#   sbatch --export=ALL,CONFIG=configs/base.json slurm/train.sbatch
set -euo pipefail

ROOT=${SLT_ROOT}
: "${CONFIG:=configs/base.json}"

cd "$ROOT/code"
export PYTHONPATH="$ROOT/code/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/.hf"           # keep model downloads off the 40 GB home quota
# The env ships a newer libstdc++ than /lib64; without this, conda's libicui18n
# fails to resolve CXXABI_1.3.15 and any sqlite3 import dies.
export LD_LIBRARY_PATH="$ROOT/env/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ROOT/env/bin/python" src/train/train.py --config "$CONFIG"
