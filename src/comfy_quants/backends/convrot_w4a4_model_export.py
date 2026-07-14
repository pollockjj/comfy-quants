"""Deterministic stock-ComfyUI ConvRot W4A4 weight producer."""

from __future__ import annotations

from comfy_quants.core.errors import PayloadWriteError
from comfy_quants.formats.convrot import build_hadamard, rotate_weight
from comfy_quants.formats.convrot_w4a4 import CONVROT_W4A4_QUANT_GROUP_SIZE
from comfy_quants.formats.int4_common import pack_signed_int4_pairs


def _quantize_convrot_w4a4_per_row(
    tensor, *, group_size: int, stochastic_rounding: int | None = 0
):
    """Return packed W4 weight and FP32 per-row scale matching comfy-kitchen eager."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError("torch is required for ConvRot W4A4 export") from exc

    weight = tensor.detach()
    if weight.dim() != 2:
        raise PayloadWriteError("ConvRot W4A4 export requires a rank-2 weight tensor")
    if weight.shape[1] % group_size:
        raise PayloadWriteError(
            f"in_features {weight.shape[1]} not divisible by convrot group size {group_size}"
        )
    if weight.shape[1] % CONVROT_W4A4_QUANT_GROUP_SIZE:
        raise PayloadWriteError(
            f"in_features {weight.shape[1]} not divisible by quant group size "
            f"{CONVROT_W4A4_QUANT_GROUP_SIZE}"
        )

    hadamard = build_hadamard(group_size, device=weight.device, dtype=weight.dtype)
    rotated = rotate_weight(weight, hadamard, group_size)
    absmax = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = absmax / 7.0
    scaled = rotated / scale
    if stochastic_rounding is not None and stochastic_rounding > 0:
        generator = torch.Generator(device=scaled.device)
        generator.manual_seed(stochastic_rounding)
        scaled.add_(torch.rand(scaled.shape, dtype=scaled.dtype, device=scaled.device, generator=generator))
        quantized = scaled.floor_()
    else:
        quantized = scaled.round_()
    quantized = quantized.clamp_(-7, 7).to(torch.int8)
    packed = pack_signed_int4_pairs(quantized)
    return packed, scale.reshape(weight.shape[0]).to(torch.float32).contiguous()
