# SeedVR2 INT4 SVDQuant W4A4 r128 Method

This path produces the first calibrated SeedVR2 INT4 ceiling artifact:
calibrated SVDQuant W4A4 with low-rank rank 128, bfloat16 auxiliary tensors, and
raw-input low-rank branch basis.

## Layer Selection

Use the same target selection for activation capture and export:

- Quantize only 2D `.weight` tensors.
- Quantize only layers with `N % 128 == 0` and `K % 64 == 0`.
- Keep `blocks.35.*` high precision for 7B-family checkpoints.
- Keep untile-packable input/output projections high precision.
- Copy 1D tensors and baked conditioning tensors unchanged.

## Activation Capture

Capture per-input-channel activation statistics for every selected linear.
Each layer must produce an `input_amax` vector with length `K`.

```bash
PYTHONPATH=src /path/to/ComfyUI/.venv/bin/python scripts/seedvr2_capture_activation_stats.py \
  --comfy-root /path/to/ComfyUI \
  --source /path/to/ComfyUI/models/diffusion_models/seedvr2_3b_fp16.safetensors \
  --workflow /path/to/api_workflow.json \
  --unet-name seedvr2_3b_fp16.safetensors \
  --out /path/to/scratch/seedvr2_3b_activation_stats.json
```

## Calibrated r128 Export

The calibrated export uses activation-aware smoothing and residual low-rank
factorization. It does not use GPTQ or output-error low-rank calibration.

```bash
PYTHONPATH=src /path/to/ComfyUI/.venv/bin/python scripts/seedvr2_int4_export.py \
  --src /path/to/ComfyUI/models/diffusion_models/seedvr2_3b_fp16.safetensors \
  --out /path/to/ComfyUI/models/diffusion_models/seedvr2_3b_int4_svdquant_w4a4_calibrated_r128.safetensors \
  --rank 128 \
  --activation-stats /path/to/scratch/seedvr2_3b_activation_stats.json \
  --calibrated \
  --scale-dtype bfloat16 \
  --lowrank-branch-input-basis raw \
  --quant-device cuda:0 \
  --device cuda:0
```

If `--rank` is omitted, calibrated mode defaults to rank 128. Data-free mode
keeps rank 32 as its default so the original inert-low-rank probe remains
reproducible.

## Required Checks

- Selected layer count matches the data-free 3B count.
- Every calibrated selected layer has nonzero `proj_down` and `proj_up`.
- Every calibrated selected layer has non-identity `smooth_factor`.
- Auxiliary tensors are bfloat16.
- The artifact loads in ComfyUI and completes the API workflow.
- No rank reduction or 7B-family artifact is produced until the 3B r128 path
  passes mechanical, load, inference, and quality-evaluation gates.
