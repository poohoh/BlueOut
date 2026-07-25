# About this vendored Diffusers fork

This directory is a trimmed copy (source + packaging metadata; tests/docs/
examples removed) of a Diffusers fork used by BlueOut.

Provenance:

- Base: Kyeongryeol Go's InstanceDiffusion port of Diffusers
  (`gokyeongryeol/diffusers`, branch `instancediffusion`;
  upstream PR: https://github.com/huggingface/diffusers/pull/10079),
  which adds the `unifusion` attention type, UniFusion position net,
  gated fusers, and ScaleU used by `kyeongry/instancediffusion_sd15`.
- BlueOut modifications on top of that branch:
  - `models/unets/unet_2d_condition.py`: skip ScaleU when the UNet has no
    ScaleU parameters (required for our main U-Net, which keeps fusers but
    removes ScaleU), plus a bbox-only position-net helper.
  - `models/embeddings.py`: bbox-only UniFusion projection.
  - `pipelines/stable_diffusion_instdiff/`: additional inpaint pipeline.
  - `scripts/`: checkpoint conversion utilities
    (`convert_instdiff_to_diffusers.py`, `convert_refnet_bbox_only_to_diffusers.py`).

Diffusers is licensed under Apache-2.0 (see `LICENSE` in this directory).
