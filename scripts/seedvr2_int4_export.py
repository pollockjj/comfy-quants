#!/usr/bin/env python3
"""SeedVR2 INT4 (SVDQuant W4A4) data-free export for native ComfyUI.

Quantizes every tile-packable 2D Linear weight to INT4 SVDQuant W4A4 via the
calibration-free natural_svdquant path (identity smoothing, zero low-rank),
then tile-packs into comfy-kitchen's kitchen_tile_packed_w4a4 layout and writes
one ComfyUI-loadable checkpoint.

Keep-high policy mirrors the SeedVR2 fp8/nvfp4 legs: the input/output projections
that cannot tile-pack (vid_in.proj K%64!=0, vid_out.proj N%128!=0) stay fp16, the
7B last block (blocks.35.*) stays fp16, and 1-D params and the baked conditioning
copy through unchanged.

Aux tensors (weight_scale, smooth_factor, proj_down, proj_up) are emitted in
bfloat16 to match SeedVR2's bf16 runtime compute dtype; the fused W4A4 kernel
requires the activation and these params to share a dtype.

Source is a SeedVR2 fp16/bf16 safetensors with conditioning already baked
(positive_conditioning / negative_conditioning copy through verbatim).
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comfy_quants.algorithms.int4_svdquant.weight_quant import (  # noqa: E402
    quantize_linear_weight_to_calibrated_natural_svdquant,
    quantize_linear_weight_to_natural_svdquant,
)
from comfy_quants.algorithms.int4_svdquant.branch_basis import (  # noqa: E402
    fold_proj_down_for_raw_branch,
)
from comfy_quants.algorithms.int4_svdquant.stats import load_activation_stats_map  # noqa: E402
from comfy_quants.formats.kitchen_tilepack import (  # noqa: E402
    KITCHEN_BLOCK_N,
    KITCHEN_GROUP_SIZE,
    patch_svdquant_comfy_quant,
)
from comfy_quants.backends.int4_kitchen_export import (  # noqa: E402
    write_svdquant_w4a4_kitchen_checkpoint,
)

NAT_KEYS = ("weight", "weight_scale", "smooth_factor", "proj_down", "proj_up")


def should_quantize(key, shape):
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    if key.startswith("blocks.35."):  # 7B last-block carve-out (no-op on 3B)
        return False
    n, k = shape
    return n % KITCHEN_BLOCK_N == 0 and k % KITCHEN_GROUP_SIZE == 0


def main():
    ap = argparse.ArgumentParser(description="Export a SeedVR2 INT4 SVDQuant W4A4 checkpoint.")
    ap.add_argument("--src", required=True, help="SeedVR2 fp16/bf16 safetensors with conditioning baked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--activation-stats", help="SeedVR2 per-linear activation stats JSON for calibrated export")
    ap.add_argument("--calibrated", action="store_true", help="Use calibrated SVDQuant instead of data-free natural SVDQuant")
    ap.add_argument(
        "--lowrank-branch-input-basis",
        default="raw",
        choices=["raw", "post_smoothing"],
        help="Store proj_down for raw activations by default, matching ComfyUI's SVDQuant runtime.",
    )
    ap.add_argument("--scale-dtype", default="bfloat16", choices=["bfloat16", "float16", "source"])
    ap.add_argument("--device", default="cpu", help="tile-pack device")
    args = ap.parse_args()

    if args.calibrated and not args.activation_stats:
        raise SystemExit("--calibrated requires --activation-stats")
    activation_stats = load_activation_stats_map(args.activation_stats, device="cpu") if args.activation_stats else {}

    f = safe_open(args.src, framework="pt", device="cpu")
    out = {}
    quantized, kept_high = 0, []
    for k in f.keys():
        v = f.get_tensor(k)
        if should_quantize(k, tuple(v.shape)):
            prefix = k[: -len(".weight")]
            if args.calibrated:
                if prefix not in activation_stats:
                    raise KeyError(f"missing activation stats for {prefix}")
                nat = quantize_linear_weight_to_calibrated_natural_svdquant(
                    v,
                    activation_stats=activation_stats[prefix],
                    rank=args.rank,
                    group_size=KITCHEN_GROUP_SIZE,
                    scale_dtype=args.scale_dtype,
                ).to_dict()
                if args.lowrank_branch_input_basis == "raw":
                    nat["proj_down"] = fold_proj_down_for_raw_branch(nat["proj_down"], nat["smooth_factor"])
            else:
                nat = quantize_linear_weight_to_natural_svdquant(
                    v, rank=args.rank, group_size=KITCHEN_GROUP_SIZE, scale_dtype=args.scale_dtype
                ).to_dict()
            for nk in NAT_KEYS:
                out[f"{prefix}.{nk}"] = nat[nk]
            out[f"{prefix}.comfy_quant"] = patch_svdquant_comfy_quant()
            quantized += 1
        else:
            out[k] = v
            if k.endswith(".weight") and v.dim() == 2:
                kept_high.append((k, tuple(v.shape)))

    print(f"quantized 2D linears -> int4: {quantized}")
    print("kept-high 2D linears:")
    for k, s in kept_high:
        print(f"  {k}  {s}")

    report = write_svdquant_w4a4_kitchen_checkpoint(
        tensors=out,
        output_checkpoint=args.out,
        source_checkpoint=args.src,
        source_layout="single_file",
        device=args.device,
        require_svdquant=True,
        hash_output=True,
    )
    print(f"repacked_layers={report.repacked_layer_count} tensors={report.output_tensor_count} "
          f"bytes={report.output_bytes}")
    print(f"dtype_counts={report.dtype_counts}")
    print(f"hash={report.output_hash}")
    print(f"out={report.output_checkpoint}")


if __name__ == "__main__":
    main()
