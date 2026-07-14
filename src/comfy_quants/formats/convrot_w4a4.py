"""Stock-ComfyUI ConvRot W4A4 checkpoint format."""

from __future__ import annotations

from comfy_quants.formats.base import QuantFormatSpec
from comfy_quants.registry.global_registry import registry

CONVROT_W4A4_FORMAT_NAME = "convrot_w4a4"
CONVROT_W4A4_QUANT_GROUP_SIZE = 64


def convrot_w4a4_checkpoint_quant_config(
    *, convrot_groupsize: int = 256, linear_dtype: str = "int4"
) -> dict[str, str | int]:
    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(f"linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    conf: dict[str, str | int] = {
        "format": CONVROT_W4A4_FORMAT_NAME,
        "convrot_groupsize": int(convrot_groupsize),
    }
    if linear_dtype != "int4":
        conf["linear_dtype"] = linear_dtype
    return conf


CONVROT_W4A4_FORMAT = QuantFormatSpec(
    name=CONVROT_W4A4_FORMAT_NAME,
    storage_dtype="int8",
    bits=4,
    category="integer_weight_activation",
    scale_required=True,
    default_scale_granularity="per_channel",
    compatible_families=("diffusion_gemma", "gemma4"),
    notes=(
        "Stock-ComfyUI signed packed W4 weights with dynamic A4 activations.",
        "Offline ConvRot uses a regular Hadamard and per-output-row FP32 scales.",
        "Loaded by TensorCoreConvRotW4A4Layout; quant_group_size is fixed at 64.",
    ),
    metadata={
        "weight_tensor": "weight",
        "scale_tensor": "weight_scale",
        "marker_tensor": "comfy_quant",
        "weight_scale_shape": "per_row_1d",
        "quant_group_size": CONVROT_W4A4_QUANT_GROUP_SIZE,
        "symmetric": True,
        "quant_min": -7,
        "quant_max": 7,
        "packing": "signed_int4_low_nibble_first",
        "no_input_scale": True,
        "downstream_loader": "stock ComfyUI QUANT_ALGOS[convrot_w4a4]",
    },
)

registry.register_format(CONVROT_W4A4_FORMAT)
