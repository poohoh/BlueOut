# Checkpoints

Expected layout:

```text
checkpoints/
├── stage1_global_unet.pt                               # ours Stage 1 global U-Net
├── stage2_main_adapters.pt                             # ours Stage 2 (main adapters)
└── pretrained/                                         # for Stage 1 training only
    ├── inpainting/sd-v1-5-inpainting.ckpt
    └── InstanceDiffusion/instancediffusion_modules.pth
```

## Ours checkpoints

For inference, download both files from Google Drive and place them as above:

```text
https://drive.google.com/drive/u/0/folders/1K8P41CimIOjEjnON1GI7eMN2hTinlktf
```

If you train from scratch instead, copy your chosen epoch checkpoints to the
same paths:

- Stage 1 (`scripts/train_stage1.sh`) → `results/train_stage1/<RUN>/ckpt/stage1_epochXXXX.pt`
  → `checkpoints/stage1_global_unet.pt`
- Stage 2 (`scripts/train_stage2.sh`) → `results/train_stage2_main_only/ckpt/main_only_epochXXXX.pt`
  → `checkpoints/stage2_main_adapters.pt`

Paths can be overridden with `GLOBAL_CKPT` / `MAIN_CKPT`.

## Pretrained weights (for Stage 1 training)

- `sd-v1-5-inpainting.ckpt`: download the Stable Diffusion v1.5 inpainting
  single-file checkpoint (`runwayml/stable-diffusion-inpainting`) and place it
  at `checkpoints/pretrained/inpainting/sd-v1-5-inpainting.ckpt`.
- `instancediffusion_modules.pth`: download the pretrained checkpoint from the
  official InstanceDiffusion repository
  (https://github.com/frank-xwang/InstanceDiffusion), then extract the ID
  modules:

```bash
python scripts/extract_instancediffusion_modules.py \
  --src /path/to/instancediffusion_sd15.pth \
  --trust-source
```

Inference does not need the `pretrained/` files.
