#!/usr/bin/env python3
"""Capture SeedVR2 per-linear activation stats for calibrated INT4 export.

The script executes a ComfyUI API workflow in-process, swaps the UNETLoader
target to a dense SeedVR2 checkpoint, hooks the same tile-packable linears used
by ``seedvr2_int4_export.py``, and writes an INT4 activation-stats JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comfy_quants.algorithms.int4_svdquant.stats import ActivationStats, write_activation_stats_map  # noqa: E402
from comfy_quants.formats.kitchen_tilepack import KITCHEN_BLOCK_N, KITCHEN_GROUP_SIZE  # noqa: E402


def should_quantize(key: str, shape: tuple[int, ...]) -> bool:
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    if key.startswith("blocks.35."):
        return False
    n, k = shape
    return n % KITCHEN_BLOCK_N == 0 and k % KITCHEN_GROUP_SIZE == 0


class _ServerStub:
    client_id = None
    last_node_id = None

    def send_sync(self, event, data, client_id=None):
        return None


class _OnlineAmax:
    def __init__(self, width: int):
        self.width = int(width)
        self.amax = torch.zeros((self.width,), dtype=torch.float32)
        self.sumsq = torch.zeros((self.width,), dtype=torch.float64)
        self.rows = 0
        self.calls = 0

    def update(self, sample: torch.Tensor) -> None:
        if int(sample.ndim) == 0:
            raise ValueError("linear activation sample must have at least one dimension")
        if int(sample.shape[-1]) != self.width:
            raise ValueError(f"expected input width {self.width}, got {int(sample.shape[-1])}")
        flat = sample.detach().to(device="cpu", dtype=torch.float32).reshape(-1, self.width)
        if int(flat.shape[0]) == 0:
            return
        self.amax = torch.maximum(self.amax, flat.abs().amax(dim=0))
        self.sumsq += flat.to(dtype=torch.float64).square().sum(dim=0)
        self.rows += int(flat.shape[0])
        self.calls += 1

    def to_stats(self) -> ActivationStats:
        if self.rows <= 0:
            raise ValueError("no activation rows captured")
        rms = torch.sqrt(self.sumsq / float(self.rows)).to(dtype=torch.float32)
        return ActivationStats(
            input_amax=self.amax.contiguous(),
            input_rms=rms.contiguous(),
            sample_count=self.calls,
            element_count=self.rows,
        )


def _target_shapes(source: Path) -> dict[str, tuple[int, int]]:
    targets: dict[str, tuple[int, int]] = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
            if should_quantize(key, shape):
                targets[key[: -len(".weight")]] = (shape[0], shape[1])
    if not targets:
        raise RuntimeError(f"no SeedVR2 INT4 target linears found in {source}")
    return targets


def _load_prompt(path: Path, *, unet_name: str, filename_prefix: str) -> dict:
    prompt = json.loads(path.read_text(encoding="utf-8"))
    for node in prompt.values():
        if node.get("class_type") == "UNETLoader":
            node["inputs"]["unet_name"] = unet_name
        if node.get("class_type") == "SaveImage":
            node["inputs"]["filename_prefix"] = filename_prefix
    return prompt


def _register_hooks(model_patcher, targets: dict[str, tuple[int, int]], accumulators: dict[str, _OnlineAmax]):
    diffusion_model = model_patcher.model.diffusion_model
    hooks = []
    seen = set()
    for name, module in diffusion_model.named_modules():
        if name not in targets:
            continue
        if not hasattr(module, "weight") or int(getattr(module.weight, "ndim", 0)) != 2:
            continue
        _n, k = targets[name]
        accumulators[name] = _OnlineAmax(k)
        seen.add(name)

        def hook(_module, inputs, _output, layer_name=name):
            if not inputs:
                raise RuntimeError(f"{layer_name}: missing linear input")
            accumulators[layer_name].update(inputs[0])

        hooks.append(module.register_forward_hook(hook))
    missing = sorted(set(targets) - seen)
    if missing:
        for hook in hooks:
            hook.remove()
        raise RuntimeError(f"target linears missing from loaded model: {missing[:10]} total={len(missing)}")
    return hooks


def _execute_prompt(prompt: dict, outputs: list[str]):
    import execution

    executor = execution.PromptExecutor(
        _ServerStub(),
        cache_type=execution.CacheType.NONE,
        cache_args={"ram": 16.0, "ram_inactive": 8.0},
    )
    executor.execute(prompt, "seedvr2-int4-calibration", extra_data={}, execute_outputs=outputs)
    if not executor.success:
        raise RuntimeError(f"ComfyUI prompt execution failed: {executor.status_messages}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture SeedVR2 INT4 activation stats from a ComfyUI API workflow.")
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Dense SeedVR2 safetensors used for target selection")
    parser.add_argument("--workflow", required=True, type=Path, help="ComfyUI API prompt")
    parser.add_argument("--unet-name", required=True, help="Dense model filename visible to UNETLoader")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--filename-prefix", default="seedvr2_calibration_capture")
    args = parser.parse_args()

    comfy_root = args.comfy_root.resolve()
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    os.chdir(str(comfy_root))

    targets = _target_shapes(args.source)
    prompt = _load_prompt(args.workflow, unet_name=args.unet_name, filename_prefix=args.filename_prefix)

    # Hide script arguments from ComfyUI's global cli parser before importing it.
    sys.argv = [sys.argv[0], "--base-directory", str(comfy_root), "--disable-all-custom-nodes"]

    script_dir = str(Path(__file__).resolve().parent)
    for path in (str(_SRC), script_dir):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(comfy_root))
    loaded_utils = sys.modules.get("utils")
    if loaded_utils is not None and not str(getattr(loaded_utils, "__file__", "")).startswith(str(comfy_root)):
        sys.modules.pop("utils", None)

    import folder_paths
    import nodes

    folder_paths.set_output_directory(str(args.out.parent / "comfy_output"))
    folder_paths.set_temp_directory(str(args.out.parent / "comfy_temp"))
    asyncio.run(nodes.load_custom_node(str(comfy_root / "comfy_extras/nodes_latent.py"), module_parent="comfy_extras"))
    asyncio.run(nodes.load_custom_node(str(comfy_root / "comfy_extras/nodes_post_processing.py"), module_parent="comfy_extras"))
    asyncio.run(nodes.load_custom_node(str(comfy_root / "comfy_extras/nodes_seedvr.py"), module_parent="comfy_extras"))

    original_load_unet = nodes.UNETLoader.load_unet
    accumulators: dict[str, _OnlineAmax] = {}
    hooks = []

    def wrapped_load_unet(self, unet_name, weight_dtype):
        result = original_load_unet(self, unet_name, weight_dtype)
        hooks.extend(_register_hooks(result[0], targets, accumulators))
        return result

    nodes.UNETLoader.load_unet = wrapped_load_unet
    try:
        output_nodes = [node_id for node_id, node in prompt.items() if node.get("class_type") == "SaveImage"]
        _execute_prompt(prompt, output_nodes)
    finally:
        nodes.UNETLoader.load_unet = original_load_unet
        for hook in hooks:
            hook.remove()

    missing = sorted(set(targets) - set(accumulators))
    empty = sorted(name for name, acc in accumulators.items() if acc.rows <= 0)
    if missing or empty:
        raise RuntimeError(f"incomplete capture: missing={missing[:10]} empty={empty[:10]}")
    stats = {name: accumulators[name].to_stats() for name in sorted(accumulators)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_activation_stats_map(args.out, stats, schema_version="seedvr2_int4_activation_stats.v1")
    print(json.dumps({
        "status": "ok",
        "output": str(args.out),
        "target_layer_count": len(targets),
        "captured_layer_count": len(stats),
        "sample_count_min": min(item.sample_count for item in stats.values()),
        "element_count_min": min(item.element_count for item in stats.values()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
