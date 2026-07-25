#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
GLOBAL_CKPT="${GLOBAL_CKPT:-checkpoints/stage1_global_unet.pt}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-inpainting}"
INSTDIFF_MODEL="${INSTDIFF_MODEL:-kyeongry/instancediffusion_sd15}"

IMAGES_ROOT="${IMAGES_ROOT:-datasets/images}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-datasets/annotations}"
INCLUDE_DATASETS="${INCLUDE_DATASETS:-}"

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-26}"
ACCUM="${ACCUM:-10}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
LR="${LR:-5e-5}"
MAX_EPOCHS="${MAX_EPOCHS:-0}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTDIR="${OUTDIR:-results/train_stage2_main_only}"

if [[ ! -f "${GLOBAL_CKPT}" ]]; then
  echo "Missing GLOBAL_CKPT: ${GLOBAL_CKPT}" >&2
  exit 1
fi

ARGS=(
  --global_checkpoint "${GLOBAL_CKPT}"
  --base_model "${BASE_MODEL}"
  --instdiff_model "${INSTDIFF_MODEL}"
  --images_root "${IMAGES_ROOT}"
  --annotations_root "${ANNOTATIONS_ROOT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_epochs "${MAX_EPOCHS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${ACCUM}"
  --mixed_precision "${MIXED_PRECISION}"
  --lr "${LR}"
  --output_dir "${OUTDIR}"
)

if [[ -n "${INCLUDE_DATASETS}" ]]; then
  ARGS+=(--include_datasets "${INCLUDE_DATASETS}")
fi

if [[ "${NUM_GPUS}" -le 1 ]]; then
  "${PYTHON}" -u train_stage2_main_adapters.py "${ARGS[@]}" "$@"
else
  "${PYTHON}" -m accelerate.commands.launch \
    --num_machines 1 \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    train_stage2_main_adapters.py "${ARGS[@]}" "$@"
fi
