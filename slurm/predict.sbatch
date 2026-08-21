#!/bin/bash
#SBATCH --job-name=wslp-pred
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=${SLT_ROOT}/logs/pred-%j.out
#
# Decode a checkpoint into submission-ready CSVs.
#
#   sbatch --export=ALL,CKPT=runs/base/best.pt,SOURCE=test slurm/predict.sbatch
#   sbatch --export=ALL,CKPT=runs/base/best.pt,SOURCE=test,SWEEP=1 slurm/predict.sbatch
#
# SWEEP=1 decodes the beam x length-penalty grid. With no local dev set for the
# eval domain (findings F3), the grid is scored on Codabench, so keep it small
# enough to fit the 20/day budget.
set -euo pipefail

ROOT=${SLT_ROOT}
: "${CKPT:=runs/base/best.pt}"
: "${SOURCE:=test}"

cd "$ROOT/code"
export PYTHONPATH="$ROOT/code/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$ROOT/.hf"
export LD_LIBRARY_PATH="$ROOT/env/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$ROOT/preds"
ARGS=(--ckpt "$ROOT/$CKPT" --source "$SOURCE" -o "$ROOT/preds/${SOURCE}.csv")
[ "${SWEEP:-0}" = "1" ] && ARGS+=(--sweep)

"$ROOT/env/bin/python" src/eval/predict.py "${ARGS[@]}"
ls -la "$ROOT/preds/"
