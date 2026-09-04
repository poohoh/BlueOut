# High-Resolution Artwork Outpainting with Global Blueprint Guidance and Layout Control

Official PyTorch implementation of the ECCV 2026 paper.

> **High-Resolution Artwork Outpainting with Global Blueprint Guidance and Layout Control**<br>
> Junha Kim, Hyunjoon Park, Donghyeon Cho<br>
> Hanyang University<br>
> ECCV 2026
>
> [[Project Page]](https://poohoh.github.io/BlueOut.github.io/) [Paper] [[ArXiv]](https://arxiv.org/abs/2607.06162) [[Video]](https://www.youtube.com/watch?v=0_czRYLmRzs)

A two-stage diffusion framework for layout-controllable high-resolution
artwork outpainting: **Stage 1** generates a low-resolution global blueprint
with bounding-box layout control, and **Stage 2** synthesizes high-resolution
patches in parallel, guided by blueprint features and initialized from the
blueprint via forward diffusion.

## Installation

```bash
git clone https://github.com/poohoh/BlueOut.git
cd BlueOut

# main environment: Stage 2 training + inference (Diffusers stack)
conda create -n blueout python=3.10 -y
conda activate blueout
pip install -r requirements.txt

# Stage 1 training environment (LDM / PyTorch-Lightning stack)
conda create -n blueout-stage1 python=3.10 -y
conda activate blueout-stage1
pip install -r requirements_stage1.txt
```

`requirements.txt` installs the vendored Diffusers fork at
`third_party/diffusers` (adds the `unifusion` attention type; stock PyPI
diffusers will not work), so run it from the repository root.
Inference only needs the `blueout` environment.
Tested with Python 3.10, PyTorch 2.7 (cu118); CUDA is required.

## Checkpoints

Download the checkpoints from
[Google Drive](https://drive.google.com/drive/folders/1TntLEn7eJgU9AdEmy1FWUZl6T7dPnC8p?usp=sharing)
and place them at:

```text
checkpoints/stage1_global_unet.pt
checkpoints/stage2_main_adapters.pt
```

The upstream models (`runwayml/stable-diffusion-inpainting`,
`kyeongry/instancediffusion_sd15`) are downloaded automatically from
Hugging Face on first run. See [checkpoints/README.md](checkpoints/README.md).

## Inference

```bash
conda activate blueout
bash scripts/infer_ours_mgpu.sh

# multi-GPU
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/infer_ours_mgpu.sh
```

Options (environment variables):

| Variable | Default | Description |
| --- | --- | --- |
| `GLOBAL_CKPT` | `checkpoints/stage1_global_unet.pt` | Stage 1 global U-Net checkpoint |
| `MAIN_CKPT` | `checkpoints/stage2_main_adapters.pt` | Stage 2 main adapter checkpoint |
| `IMAGES_ROOT` | `datasets/images/iconart` | Source image directory |
| `CAPTIONS_FILE` | `datasets/caption/iconart/blip2_prompts.json` | JSONL captions (`file`, `prompt`) |
| `OUTDIR` | `results/ours` | Output directory |
| `NUM_SAMPLES` | `-1` (all) | Number of images to process |
| `NPROC_PER_NODE` | `1` | Number of GPUs |

Sampling options pass through to the Python entrypoint, e.g.
`bash scripts/infer_ours_mgpu.sh --steps 30 --guidance_scale 3.0 --seed 42`
(paper defaults). See `python scripts/infer_ours_mgpu.py --help`.

## Training

### 1. Pretrained weights (for Stage 1)

Place `sd-v1-5-inpainting.ckpt` (from `runwayml/stable-diffusion-inpainting`)
at `checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt`.

Then download the
[official InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion)
checkpoint and extract the ID modules:

```bash
python scripts/extract_instancediffusion_modules.py \
  --src /path/to/instancediffusion_sd15.pth \
  --trust-source
```

### 2. Data

Training: **WikiArt**, **Human-Art**, **LAION-High-Resolution**
(aesthetic-filtered). Evaluation: **IconArt**.

Place images under `datasets/images/<dataset_name>/` with `<dataset_name>`
in `wikiart`, `humanart`, `laion-high-resolution` (as `<chunk>/<image>`),
`iconart`, then run:

```bash
# 1) aesthetic filtering (LAION only)
python scripts/data_prep/compute_aesthetic_scores.py \
  --input-root datasets/images/laion-high-resolution \
  --output-root datasets/AES_score/laion-high-resolution \
  --aesthetic-weights /path/to/aesthetic_v2_clip_vit_l_14_linear.pt
python scripts/data_prep/filter_low_aesthetic.py --threshold 5.2 --execute

# 2) captioning (BLIP; needs salesforce-lavis)
for dataset_name in wikiart humanart iconart; do
  python scripts/data_prep/caption_images.py \
    --data-root "datasets/images/${dataset_name}" \
    --output-root "datasets/caption/${dataset_name}" \
    --per-dir folder --device cuda --resume
done
python scripts/data_prep/caption_images.py \
  --data-root datasets/images/laion-high-resolution \
  --output-root datasets/caption/laion-high-resolution \
  --per-dir json --device cuda --resume

# 3) bbox annotation (RAM + GroundingDINO) — set up the clone first:
#    https://github.com/IDEA-Research/Grounded-Segment-Anything#installation
git clone --recursive https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
cp ../scripts/data_prep/make_data.py .
python make_data.py --datasets_root ../datasets \
  --only_main wikiart humanart laion-high-resolution iconart --auto_mgpu
cd ..

# 4) CLIP object embeddings
python scripts/data_prep/generate_clip_text_embeddings.py \
  --annotations-root datasets/annotations
```

The resulting annotation JSONs (`datasets/annotations/<dataset_name>/**/*.json`):

```json
{
  "image_path": "relative/path/to/image.png",
  "caption": "global image caption",
  "objects": ["object text"],
  "boxes": [[0.1, 0.1, 0.4, 0.4]],
  "text_embedding_before": ["base64-encoded-float32-vector"]
}
```

`boxes` are normalized `[x1, y1, x2, y2]`; `text_embedding_before` holds
base64-encoded float32 CLIP text embeddings per object.

For IconArt (inference), run steps 2-4 with the dataset lists reduced to
`iconart`.

### 3. Stage 1: global U-Net training

```bash
conda activate blueout-stage1
bash scripts/train_stage1.sh
```

Defaults: `BATCH_SIZE=32`, `ACCUM=5`, `MAX_EPOCHS=100`, `DATA_ROOT=datasets`,
`OUTDIR=results/train_stage1` (override via environment variables).

Checkpoints: `<OUTDIR>/<run>/ckpt/stage1_epochXXXX.pt`.

### 4. Stage 2: main adapter training

```bash
conda activate blueout
GLOBAL_CKPT=results/train_stage1/<run>/ckpt/stage1_epochXXXX.pt \
bash scripts/train_stage2.sh
```

Defaults: `NUM_GPUS=1`, `BATCH_SIZE=26`, `ACCUM=10`, `LR=5e-5`,
`OUTDIR=results/train_stage2_main_only` (override via environment variables).

Checkpoints: `<OUTDIR>/ckpt/main_only_epochXXXX.pt`.

## Code structure

```text
train_stage1_global_unet.py      # Stage 1 training entrypoint
train_stage2_main_adapters.py    # Stage 2 training entrypoint
scripts/infer_ours_mgpu.py       # inference entrypoint
scripts/                         # runnable wrappers and tools
scripts/data_prep/               # dataset preprocessing (aesthetic filter, captioning, bbox annotation)
ldm/                             # LDM modules used by Stage 1
InstanceDiffusion/               # minimal bbox-only InstanceDiffusion modules
diffusers_local/                 # project-specific Diffusers components and global-feature hooks
third_party/diffusers/           # vendored Diffusers fork with InstanceDiffusion support
data/                            # training and inference dataloaders
```

Stage 1 extends the official InstanceDiffusion training code (CompVis/LDM
style); Stage 2 and inference are built on Diffusers. The Stage 1 checkpoint
is converted automatically (LDM → Diffusers keys) when Stage 2 loads it.

## Acknowledgements

This codebase builds on
[Stable Diffusion](https://github.com/CompVis/stable-diffusion),
[InstanceDiffusion](https://github.com/frank-xwang/InstanceDiffusion),
[ProOut](https://github.com/EadCat/ProOut), and
[AlignNoise](https://github.com/WHUIR/AlignNoise).

`third_party/diffusers` is a fork of the
[Diffusers](https://github.com/huggingface/diffusers) library, based on
[Kyeongryeol Go's InstanceDiffusion port](https://github.com/huggingface/diffusers/pull/10079)
(`gokyeongryeol/diffusers`), who also released the
`kyeongry/instancediffusion_sd15` weights.

We thank the authors for releasing their code.

## License

This repository is released under the MIT License (see [LICENSE](LICENSE)).

Bundled third-party components and model weights retain their original
licenses (see `licenses/` and the `LICENSE` files in the respective
directories).
