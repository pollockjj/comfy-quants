#!/usr/bin/env python3
"""
Convert ByteDance SeedVR2 checkpoints to ComfyUI safetensors.

Each source tensor is cast to the target dtype and written to safetensors with the
original key names (safetensors sorts keys; no metadata). DiT files additionally embed
the fixed text conditioning (--cond) as positive_conditioning / negative_conditioning,
copied through as-is (bf16).

Precisions:
  fp16                          every tensor -> float16
  fp8_e4m3fn                    every tensor -> float8_e4m3fn
  fp8_e4m3fn_mixed_block35_fp16 float8_e4m3fn, but tensors under "blocks.35." kept float16
                                (keeping the last DiT block in fp16 avoids line/tile
                                 artifacts on the 7B model)

Examples:
  # 3B DiT -> fp16 and fp8, conditioning baked in (one load serves both jobs)
  python seedvr2_convert.py --src seedvr2_ema_3b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_3b_fp16.safetensors \
      --job fp8_e4m3fn:seedvr2_3b_fp8_e4m3fn.safetensors

  # 7B DiT -> fp16 and block35-mixed fp8
  python seedvr2_convert.py --src seedvr2_ema_7b.pth --cond pos_emb.pt,neg_emb.pt \
      --job fp16:seedvr2_7b_fp16.safetensors \
      --job fp8_e4m3fn_mixed_block35_fp16:seedvr2_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors

  # VAE (no conditioning)
  python seedvr2_convert.py --src ema_vae.pth --job fp16:ema_vae_fp16.safetensors

A job may carry an expected SHA256 (PRECISION:OUT:SHA256) to verify the written file.

==========================================================================================
Provenance
==========================================================================================
Source checkpoints (Apache-2.0), pinned to the exact HuggingFace revision converted from:

  ByteDance-Seed/SeedVR2-3B  @ 37255ff8cccfb01071b87f635a5948ca8d53117c
  https://huggingface.co/ByteDance-Seed/SeedVR2-3B/tree/37255ff8cccfb01071b87f635a5948ca8d53117c
    6bcc5ac59447e97b100477480aebb01be2ec724c8340bb83faae21f64848604b  seedvr2_ema_3b.pth   (2025-06-22 "update ckpt")
    c7df8a67e68b7f9aca3d5d2153d2ce8ab4373687741a0f9ce87cb356ace51cac  ema_vae.pth
    fa07a14844314772266b66c3b95deb0027696d8fe7065721263db5176f45d799  pos_emb.pt
    6a43e5800ef2354f1c156d27535834da055cbec8248298b8923492bba2076581  neg_emb.pt

  ByteDance-Seed/SeedVR2-7B  @ eb0c4281d41ba3767d4f14370f0e37e9e9180c16
  https://huggingface.co/ByteDance-Seed/SeedVR2-7B/tree/eb0c4281d41ba3767d4f14370f0e37e9e9180c16
    e1b2ae25505607e61f2a7dc7967ba778aaf3e3626d9969ce6e24c52d9ddebfcd  seedvr2_ema_7b.pth
    ced5706c976d5879efcab9e108349d67abcbd8a9b36a1f48bf0f19c24164a264  seedvr2_ema_7b_sharp.pth

Conditioning embedded in every DiT output (from the 3B repo above):
    positive_conditioning  <-  pos_emb.pt  (fa07a148...)
    negative_conditioning  <-  neg_emb.pt  (6a43e580...)

Outputs  ( sha256  file  <-  source.pth, precision [+ conditioning] ):
  20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1  ema_vae_fp16.safetensors  <- ema_vae.pth, fp16 (no cond)
  98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660  seedvr2_3b_fp16.safetensors  <- seedvr2_ema_3b.pth, fp16 + cond
  a0226eaa2c3e6f47ae5ce83225120f16479da890ced1a3bc32b1a14619787914  seedvr2_3b_fp8_e4m3fn.safetensors  <- seedvr2_ema_3b.pth, fp8_e4m3fn + cond
  2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252  seedvr2_7b_fp16.safetensors  <- seedvr2_ema_7b.pth, fp16 + cond
  d89ac95ee1566dfc1ee50c6075a2bfe4028d811dd8751f584505de89ef5c4cf3  seedvr2_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b.pth, fp8_e4m3fn_mixed_block35_fp16 + cond
  70823bca54b9c24eeb56e1c452697c7c2a430867e58db0e376c6e260f3a4489d  seedvr2_7b_sharp_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, fp16 + cond
  700ee64fe0859c3df3abfa40c89f3a16068651bf8c8e5294726b6369e7b0d1e3  seedvr2_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors  <- seedvr2_ema_7b_sharp.pth, fp8_e4m3fn_mixed_block35_fp16 + cond
"""
import argparse
import collections
import hashlib

import torch
from safetensors.torch import save_file

FP8 = torch.float8_e4m3fn


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state_dict(pth):
    obj = torch.load(pth, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(obj, dict):
        if any(torch.is_tensor(v) for v in obj.values()):
            return obj
        for key in ("state_dict", "ema", "model", "module", "params", "ema_model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    raise SystemExit(f"Unrecognized checkpoint structure: {type(obj)}")


def cast(sd, precision):
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if precision == "fp16":
            out[k] = v.to(torch.float16)
        elif precision == "fp8_e4m3fn":
            out[k] = v.to(FP8)
        elif precision == "fp8_e4m3fn_mixed_block35_fp16":
            out[k] = v.to(torch.float16) if k.startswith("blocks.35.") else v.to(FP8)
        else:
            raise SystemExit(f"unknown precision: {precision}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Convert ByteDance SeedVR2 .pth to ComfyUI safetensors.")
    ap.add_argument("--src", required=True, help="source .pth checkpoint")
    ap.add_argument("--job", action="append", required=True, metavar="PRECISION:OUT[:SHA256]",
                    help="repeatable; one source load serves every job")
    ap.add_argument("--cond", default=None, metavar="pos_emb.pt,neg_emb.pt",
                    help="embed text conditioning as positive_conditioning/negative_conditioning")
    ap.add_argument("--dump", action="store_true", help="print source tensor count and dtypes")
    args = ap.parse_args()

    sd = load_state_dict(args.src)

    cond = None
    if args.cond:
        pos_path, neg_path = args.cond.split(",")
        cond = {
            "positive_conditioning": torch.load(pos_path, map_location="cpu", weights_only=True),
            "negative_conditioning": torch.load(neg_path, map_location="cpu", weights_only=True),
        }

    if args.dump:
        tensor_keys = [k for k in sd if torch.is_tensor(sd[k])]
        dtypes = collections.Counter(str(sd[k].dtype) for k in tensor_keys)
        print(f"{len(tensor_keys)} tensors, dtypes={dict(dtypes)}")

    for job in args.job:
        parts = job.split(":")
        precision, out = parts[0], parts[1]
        expected = parts[2] if len(parts) > 2 and parts[2] else None
        tensors = cast(sd, precision)
        if cond:
            tensors.update(cond)
        save_file(tensors, out)
        digest = sha256(out)
        verdict = "" if expected is None else ("  OK" if digest == expected else "  MISMATCH")
        print(f"{precision:30s} {digest}  {out}{verdict}")


if __name__ == "__main__":
    main()
