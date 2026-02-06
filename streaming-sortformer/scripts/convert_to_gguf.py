#!/usr/bin/env python3
"""Convert NeMo SortFormer (.nemo) model to GGUF format.

Usage:
    uv run python scripts/convert_to_gguf.py --nemo model.nemo --out model.gguf
"""
from __future__ import annotations

import argparse
import io
import re
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# GGUF constants (inline to avoid external dependency on gguf package at
# import time -- we only need the writer, which we implement here directly
# for maximum portability and control).
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML quantization types (subset we use)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


# ---------------------------------------------------------------------------
# Minimal GGUF writer
# ---------------------------------------------------------------------------
class GGUFWriter:
    """Minimal GGUF v3 writer -- just enough for SortFormer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.kv: list[tuple[str, int, bytes]] = []  # (key, vtype, packed_value)
        self.tensors: list[tuple[str, np.ndarray, int]] = []  # (name, data, ggml_type)

    # -- metadata helpers ---------------------------------------------------
    def _pack_string(self, s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    def add_uint32(self, key: str, val: int) -> None:
        self.kv.append((key, GGUF_TYPE_UINT32, struct.pack("<I", val)))

    def add_int32(self, key: str, val: int) -> None:
        self.kv.append((key, GGUF_TYPE_INT32, struct.pack("<i", val)))

    def add_float32(self, key: str, val: float) -> None:
        self.kv.append((key, GGUF_TYPE_FLOAT32, struct.pack("<f", val)))

    def add_string(self, key: str, val: str) -> None:
        self.kv.append((key, GGUF_TYPE_STRING, self._pack_string(val)))

    def add_bool(self, key: str, val: bool) -> None:
        self.kv.append((key, GGUF_TYPE_BOOL, struct.pack("<?", val)))

    # -- tensor registration ------------------------------------------------
    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int | None = None) -> None:
        """Register a tensor. *data* must be contiguous C-order ndarray."""
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        if ggml_type is None:
            if data.dtype == np.float16:
                ggml_type = GGML_TYPE_F16
            elif data.dtype == np.float32:
                ggml_type = GGML_TYPE_F32
            else:
                raise ValueError(f"Unsupported dtype {data.dtype} for tensor {name}")
        self.tensors.append((name, data, ggml_type))

    # -- serialisation ------------------------------------------------------
    @staticmethod
    def _pad(n: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
        return ((n + alignment - 1) // alignment) * alignment

    def write(self) -> None:
        """Serialise everything to *self.path*."""
        alignment = GGUF_DEFAULT_ALIGNMENT

        with open(self.path, "wb") as f:
            # ---- header --------------------------------------------------
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv)))

            # ---- KV data ------------------------------------------------
            for key, vtype, packed_val in self.kv:
                f.write(self._pack_string(key))
                f.write(struct.pack("<I", vtype))
                f.write(packed_val)

            # ---- tensor info ---------------------------------------------
            offset = 0
            tensor_offsets: list[int] = []
            for name, data, ggml_type in self.tensors:
                tensor_offsets.append(offset)
                f.write(self._pack_string(name))
                ndim = len(data.shape)
                f.write(struct.pack("<I", ndim))
                # GGUF stores dimensions in reverse order (innermost first)
                for d in reversed(data.shape):
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", ggml_type))
                f.write(struct.pack("<Q", offset))
                offset += self._pad(data.nbytes, alignment)

            # ---- tensor data (padded to alignment) -----------------------
            # Pad from current position to alignment boundary
            pos = f.tell()
            pad_bytes = self._pad(pos, alignment) - pos
            if pad_bytes:
                f.write(b"\x00" * pad_bytes)

            for i, (name, data, _ggml_type) in enumerate(self.tensors):
                data.tofile(f)
                # Pad after each tensor
                pad_bytes = self._pad(data.nbytes, alignment) - data.nbytes
                if pad_bytes:
                    f.write(b"\x00" * pad_bytes)

        print(f"Wrote {self.path} ({self.path.stat().st_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# NeMo loading helpers
# ---------------------------------------------------------------------------
def load_nemo(nemo_path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load state_dict and config from a .nemo archive (plain tar)."""
    nemo_path = Path(nemo_path)
    state_dict = None
    config = None

    with tarfile.open(nemo_path, "r") as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename == "model_weights.ckpt":
                f = tar.extractfile(member)
                assert f is not None
                buf = io.BytesIO(f.read())
                state_dict = torch.load(buf, map_location="cpu", weights_only=True)
            elif basename == "model_config.yaml":
                f = tar.extractfile(member)
                assert f is not None
                config = yaml.safe_load(f.read())

    if state_dict is None:
        raise RuntimeError("model_weights.ckpt not found in .nemo archive")
    if config is None:
        raise RuntimeError("model_config.yaml not found in .nemo archive")
    return state_dict, config


# ---------------------------------------------------------------------------
# BatchNorm fusion
# ---------------------------------------------------------------------------
def fuse_batchnorm(
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BatchNorm into depthwise conv for conformer layer *layer_idx*.

    Returns (fused_weight, fused_bias) as float32 numpy arrays.
    """
    prefix = f"encoder.layers.{layer_idx}.conv"

    # BatchNorm parameters
    gamma = state_dict[f"{prefix}.batch_norm.weight"].float()          # [512]
    beta = state_dict[f"{prefix}.batch_norm.bias"].float()             # [512]
    running_mean = state_dict[f"{prefix}.batch_norm.running_mean"].float()  # [512]
    running_var = state_dict[f"{prefix}.batch_norm.running_var"].float()    # [512]

    # Depthwise conv parameters
    dw_weight = state_dict[f"{prefix}.depthwise_conv.weight"].float()  # [512, 1, 9]
    dw_bias = state_dict[f"{prefix}.depthwise_conv.bias"].float()      # [512]

    # Compute fused parameters in F32
    # scale[c] = gamma[c] / sqrt(running_var[c] + eps)
    inv_std = torch.rsqrt(running_var + eps)
    scale = gamma * inv_std  # [512]

    # fused_weight[c, 1, k] = dw_weight[c, 1, k] * scale[c]
    fused_weight = dw_weight * scale.view(-1, 1, 1)

    # fused_bias[c] = (dw_bias[c] - running_mean[c]) * scale[c] + beta[c]
    fused_bias = (dw_bias - running_mean) * scale + beta

    return fused_weight.numpy(), fused_bias.numpy()


# ---------------------------------------------------------------------------
# Tensor filtering
# ---------------------------------------------------------------------------
# Keys to SKIP entirely
SKIP_PATTERNS = [
    re.compile(r".*\.batch_norm\.num_batches_tracked$"),
    re.compile(r".*\.batch_norm\.weight$"),
    re.compile(r".*\.batch_norm\.bias$"),
    re.compile(r".*\.batch_norm\.running_mean$"),
    re.compile(r".*\.batch_norm\.running_var$"),
    re.compile(r"^sortformer_modules\.hidden_to_spks\."),
]

# Keys that must stay F32
F32_KEYS = {
    "preprocessor.featurizer.fb",
    "preprocessor.featurizer.window",
}


def should_skip(name: str) -> bool:
    return any(p.match(name) for p in SKIP_PATTERNS)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def convert(nemo_path: str, out_path: str) -> None:
    print(f"Loading NeMo model from {nemo_path} ...")
    state_dict, config = load_nemo(nemo_path)
    print(f"  Loaded {len(state_dict)} state_dict keys")

    writer = GGUFWriter(out_path)

    # ---- Metadata --------------------------------------------------------
    writer.add_string("general.architecture", "sortformer")

    # Mel spectrogram
    writer.add_uint32("sortformer.mel.n_mels", 128)
    writer.add_uint32("sortformer.mel.n_fft", 512)
    writer.add_uint32("sortformer.mel.hop_length", 160)  # window_stride=0.01 * sr=16000
    writer.add_uint32("sortformer.mel.win_length", 400)  # window_size=0.025 * sr=16000
    writer.add_uint32("sortformer.mel.sample_rate", 16000)
    writer.add_float32("sortformer.mel.dither", 1e-5)

    # Conformer encoder
    enc_cfg = config.get("encoder", {})
    writer.add_uint32("sortformer.encoder.n_layers", enc_cfg.get("n_layers", 17))
    writer.add_uint32("sortformer.encoder.d_model", enc_cfg.get("d_model", 512))
    writer.add_uint32("sortformer.encoder.n_heads", enc_cfg.get("n_heads", 8))
    writer.add_uint32("sortformer.encoder.conv_kernel_size", enc_cfg.get("conv_kernel_size", 9))
    writer.add_uint32("sortformer.encoder.ff_expansion", enc_cfg.get("ff_expansion_factor", 4))
    writer.add_uint32("sortformer.encoder.subsampling_factor", enc_cfg.get("subsampling_factor", 8))
    writer.add_uint32("sortformer.encoder.subsampling_conv_channels", enc_cfg.get("subsampling_conv_channels", 256))
    writer.add_uint32("sortformer.encoder.pos_emb_max_len", enc_cfg.get("pos_emb_max_len", 5000))

    # Transformer encoder
    tf_cfg = config.get("transformer_encoder", {})
    writer.add_uint32("sortformer.transformer.n_layers", tf_cfg.get("num_layers", 18))
    writer.add_uint32("sortformer.transformer.d_model", tf_cfg.get("hidden_size", 192))
    writer.add_uint32("sortformer.transformer.n_heads", tf_cfg.get("num_attention_heads", 8))
    writer.add_uint32("sortformer.transformer.ff_inner", tf_cfg.get("inner_size", 768))

    # Speakers
    writer.add_uint32("sortformer.n_speakers", config.get("max_num_of_spks", 4))

    # ---- Fuse BatchNorm for all 17 conformer layers ----------------------
    n_conformer_layers = enc_cfg.get("n_layers", 17)
    fused_weights: dict[str, np.ndarray] = {}
    fused_biases: dict[str, np.ndarray] = {}

    print(f"Fusing BatchNorm for {n_conformer_layers} conformer layers ...")
    for i in range(n_conformer_layers):
        fw, fb = fuse_batchnorm(state_dict, i)
        fused_weights[f"encoder.layers.{i}.conv.depthwise_conv.weight"] = fw
        fused_biases[f"encoder.layers.{i}.conv.depthwise_conv.bias"] = fb

    # ---- Write tensors ---------------------------------------------------
    n_written = 0
    n_skipped = 0
    n_fused = 0

    for name in sorted(state_dict.keys()):
        # Skip filtered keys
        if should_skip(name):
            n_skipped += 1
            continue

        tensor = state_dict[name]

        # Handle special cases
        if name == "preprocessor.featurizer.fb":
            # Squeeze [1, 128, 257] -> [128, 257], keep F32
            data = tensor.squeeze(0).float().numpy()
            writer.add_tensor(name, data, GGML_TYPE_F32)
            n_written += 1
            print(f"  [F32] {name}: {list(data.shape)}")
            continue

        if name == "preprocessor.featurizer.window":
            # Keep F32
            data = tensor.float().numpy()
            writer.add_tensor(name, data, GGML_TYPE_F32)
            n_written += 1
            print(f"  [F32] {name}: {list(data.shape)}")
            continue

        # Use fused weights/biases for depthwise conv
        if name in fused_weights:
            # Fused weight: compute in F32, then convert to F16
            data = fused_weights[name].astype(np.float16)
            writer.add_tensor(name, data, GGML_TYPE_F16)
            n_written += 1
            n_fused += 1
            print(f"  [F16/fused] {name}: {list(data.shape)}")
            continue

        if name in fused_biases:
            data = fused_biases[name].astype(np.float16)
            writer.add_tensor(name, data, GGML_TYPE_F16)
            n_written += 1
            n_fused += 1
            print(f"  [F16/fused] {name}: {list(data.shape)}")
            continue

        # Everything else: convert to F16
        data = tensor.float().numpy().astype(np.float16)
        writer.add_tensor(name, data, GGML_TYPE_F16)
        n_written += 1

    print(f"\nTensor summary:")
    print(f"  Written: {n_written}")
    print(f"  Skipped: {n_skipped}")
    print(f"  Fused (BN->DWConv): {n_fused}")
    print(f"  Metadata keys: {len(writer.kv)}")

    # ---- Write GGUF file -------------------------------------------------
    print(f"\nWriting GGUF to {out_path} ...")
    writer.write()
    print("Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NeMo SortFormer model to GGUF format"
    )
    parser.add_argument(
        "--nemo", required=True, help="Path to input .nemo file"
    )
    parser.add_argument(
        "--out", required=True, help="Path to output .gguf file"
    )
    args = parser.parse_args()
    convert(args.nemo, args.out)


if __name__ == "__main__":
    main()
