#!/bin/bash
#SBATCH --job-name=wslp-prep
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=YOUR_ACCOUNT-b          # burst: 288 CPU, 0 GPU -- our one GPU stays free
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=${SLT_ROOT}/logs/prep-%A_%a.out
#
# Decode .pose archives into packed float16 shards. Shards are independent,
# so this is a plain array job.
#
#   sbatch --array=0-63 --export=SOURCE=isign slurm/preprocess.sbatch
#   sbatch --array=0-3  --export=SOURCE=test  slurm/preprocess.sbatch
#   sbatch --array=0-3  --export=SOURCE=val   slurm/preprocess.sbatch
set -euo pipefail

ROOT=${SLT_ROOT}
: "${SOURCE:?set SOURCE=isign|val|test}"

N=$((SLURM_ARRAY_TASK_COUNT))
I=$((SLURM_ARRAY_TASK_ID))

echo "=== $SOURCE shard $I/$N on $(hostname) ==="
"$ROOT/env/bin/python" "$ROOT/code/scripts/preprocess.py" \
    --source "$SOURCE" --shard "$I" --num-shards "$N" \
    --max-frames 512 --stride 1
