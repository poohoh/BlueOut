#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-datasets}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-5}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
DEVICES="${DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTDIR="${OUTDIR:-results/train_stage1}"

"${PYTHON}" -u train_stage1_global_unet.py \
  --config configs/outpainting/stage1_global_unet.yaml \
  --data_root "${DATA_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --accum "${ACCUM}" \
  --image_size 512 \
  --max_epochs "${MAX_EPOCHS}" \
  --precision fp16 \
  --devices "${DEVICES}" \
  --outdir "${OUTDIR}" \
  "$@"

