#!/bin/bash
#SBATCH --job-name=wslp-dl
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_ACCOUNT-b          # burst QOS: 0 GPUs, so our single GPU stays free
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=${SLT_ROOT}/logs/dl-%j.out
#SBATCH --error=${SLT_ROOT}/logs/dl-%j.out
#
# Fetch iSign + WSLP task data. Resumable -- re-running skips complete files,
# so a dropped transfer costs only the file in flight.
#
#   sbatch slurm/download.sbatch
set -euo pipefail

ROOT=${SLT_ROOT}
mkdir -p "$ROOT"/{isign,task,logs}

module load conda 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
source ~/.hf_env                       # HF_TOKEN, mode 600
export HF_HUB_ENABLE_HF_TRANSFER=0     # plain requests: resumable, less brittle

hf auth whoami >/dev/null || { echo "FATAL: HF auth failed"; exit 1; }

echo "=== [1/3] task val+test (~14 GB) ==="
# Most urgent: nothing can be submitted without the test poses.
hf download Exploration-Lab/WSLP --repo-type dataset \
    --include "Shared_task_MT/*" --local-dir "$ROOT/task"

echo "=== [2/3] iSign annotations (~10 MB) ==="
hf download Exploration-Lab/iSign --repo-type dataset \
    --include "*.csv" --local-dir "$ROOT/isign"

echo "=== [3/3] iSign poses (~170 GB, 4 parts) ==="
# Videos (58 GB) deliberately not fetched -- the system is pose-only.
hf download Exploration-Lab/iSign --repo-type dataset \
    --include "iSign-poses_v1.1_part_*" --local-dir "$ROOT/isign" \
    --max-workers 8

echo "=== done ==="
du -sh "$ROOT"/* 2>/dev/null || true
df -h /blue/YOUR_ACCOUNT | tail -1
