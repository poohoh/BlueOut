#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
GLOBAL_CKPT="${GLOBAL_CKPT:-checkpoints/stage1_global_unet.pt}"
MAIN_CKPT="${MAIN_CKPT:-checkpoints/stage2_main_adapters.pt}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-inpainting}"
INSTDIFF_MODEL="${INSTDIFF_MODEL:-kyeongry/instancediffusion_sd15}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

IMAGES_ROOT="${IMAGES_ROOT:-datasets/images/iconart}"
CAPTIONS_FILE="${CAPTIONS_FILE:-datasets/caption/iconart/blip2_prompts.json}"
ANNOTATIONS_ROOT="${ANNOTATIONS_ROOT:-datasets/annotations/iconart}"
OUTDIR="${OUTDIR:-results/ours}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -f "${GLOBAL_CKPT}" ]]; then
  echo "Missing GLOBAL_CKPT: ${GLOBAL_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${MAIN_CKPT}" ]]; then
  echo "Missing MAIN_CKPT: ${MAIN_CKPT}" >&2
  exit 1
fi

ARGS=(
  scripts/infer_ours_mgpu.py
  --global_ckpt "${GLOBAL_CKPT}"
  --main_ckpt "${MAIN_CKPT}"
  --base_model "${BASE_MODEL}"
  --instdiff_model "${INSTDIFF_MODEL}"
  --outdir "${OUTDIR}"
  --images_root "${IMAGES_ROOT}"
  --captions_file "${CAPTIONS_FILE}"
  --annotations_root "${ANNOTATIONS_ROOT}"
  --num_workers "${NUM_WORKERS}"
  --num_samples "${NUM_SAMPLES}"
)

if [[ "${NPROC_PER_NODE}" -le 1 ]]; then
  "${PYTHON}" -u "${ARGS[@]}" "$@"
else
  "${PYTHON}" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    -- \
    "${ARGS[@]}" "$@"
fi
