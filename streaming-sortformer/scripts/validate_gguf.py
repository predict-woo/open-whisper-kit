#!/usr/bin/env python3
"""Validate a GGUF SortFormer model against the original NeMo weights.

Usage:
    uv run python scripts/validate_gguf.py --nemo model.nemo --gguf model.gguf
    uv run python scripts/validate_gguf.py --nemo model.nemo --gguf model.gguf --check-bn-fusion
"""
from __future__ import annotations

import argparse
import io
import struct
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# GGUF constants (mirrors convert_to_gguf.py)
# ---------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

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

# Size in bytes per element for GGML types
GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
}

GGML_TYPE_NUMPY = {
    GGML_TYPE_F32: np.float32,
    GGML_TYPE_F16: np.float16,
}


# ---------------------------------------------------------------------------
# Minimal GGUF reader
# ---------------------------------------------------------------------------
class GGUFTensor:
    __slots__ = ("name", "shape", "ggml_type", "offset", "data")

    def __init__(self, name: str, shape: tuple[int, ...], ggml_type: int, offset: int):
        self.name = name
        self.shape = shape
        self.ggml_type = ggml_type
        self.offset = offset
        self.data: np.ndarray | None = None


def _read_string(f) -> str:  # type: ignore[no-untyped-def]
    (length,) = struct.unpack("<Q", f.read(8))
    return f.read(length).decode("utf-8")


def _skip_kv_value(f, vtype: int) -> None:  # type: ignore[no-untyped-def]
    """Skip over a KV value of the given type."""
    fixed_sizes = {
        GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
        GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4, GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }
    if vtype in fixed_sizes:
        f.read(fixed_sizes[vtype])
    elif vtype == GGUF_TYPE_STRING:
        _read_string(f)
    elif vtype == GGUF_TYPE_ARRAY:
        (sub_type,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _skip_kv_value(f, sub_type)
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


def _read_kv_value(f, vtype: int):  # type: ignore[no-untyped-def]
    """Read and return a KV value of the given type."""
    if vtype == GGUF_TYPE_UINT32:
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == GGUF_TYPE_INT32:
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == GGUF_TYPE_BOOL:
        return struct.unpack("<?", f.read(1))[0]
    elif vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    elif vtype == GGUF_TYPE_UINT64:
        return struct.unpack("<Q", f.read(8))[0]
    else:
        _skip_kv_value(f, vtype)
        return None


def read_gguf(path: str | Path) -> tuple[dict[str, object], dict[str, GGUFTensor]]:
    """Read GGUF file, returning (metadata_dict, tensor_dict)."""
    path = Path(path)
    metadata: dict[str, object] = {}
    tensors: dict[str, GGUFTensor] = {}

    with open(path, "rb") as f:
        # Header
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic:#x})")

        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        # KV pairs
        for _ in range(n_kv):
            key = _read_string(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            val = _read_kv_value(f, vtype)
            metadata[key] = val

        # Tensor info
        for _ in range(n_tensors):
            name = _read_string(f)
            (ndim,) = struct.unpack("<I", f.read(4))
            # Dimensions are stored in reverse order (innermost first)
            dims_reversed = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            shape = tuple(reversed(dims_reversed))
            (ggml_type,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors[name] = GGUFTensor(name, shape, ggml_type, offset)

        # Data section starts at next alignment boundary
        data_start = _pad(f.tell(), GGUF_DEFAULT_ALIGNMENT)

        # Read tensor data
        for ti in tensors.values():
            abs_offset = data_start + ti.offset
            f.seek(abs_offset)
            dtype = GGML_TYPE_NUMPY[ti.ggml_type]
            n_elements = 1
            for d in ti.shape:
                n_elements *= d
            ti.data = np.frombuffer(f.read(n_elements * np.dtype(dtype).itemsize), dtype=dtype).reshape(ti.shape)

    return metadata, tensors


def _pad(n: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return ((n + alignment - 1) // alignment) * alignment


# ---------------------------------------------------------------------------
# NeMo loading (same as convert_to_gguf.py)
# ---------------------------------------------------------------------------
def load_nemo(nemo_path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    nemo_path = Path(nemo_path)
    state_dict = None
    config = None

    with tarfile.open(nemo_path, "r") as tar:
        for member in tar.getmembers():
            basename = Path(member.name).name
            if basename == "model_weights.ckpt":
                fobj = tar.extractfile(member)
                assert fobj is not None
                buf = io.BytesIO(fobj.read())
                state_dict = torch.load(buf, map_location="cpu", weights_only=True)
            elif basename == "model_config.yaml":
                fobj = tar.extractfile(member)
                assert fobj is not None
                config = yaml.safe_load(fobj.read())

    if state_dict is None:
        raise RuntimeError("model_weights.ckpt not found in .nemo archive")
    if config is None:
        raise RuntimeError("model_config.yaml not found in .nemo archive")
    return state_dict, config


# ---------------------------------------------------------------------------
# BatchNorm fusion (same as convert_to_gguf.py)
# ---------------------------------------------------------------------------
def fuse_batchnorm(
    state_dict: dict[str, torch.Tensor],
    layer_idx: int,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"encoder.layers.{layer_idx}.conv"

    gamma = state_dict[f"{prefix}.batch_norm.weight"].float()
    beta = state_dict[f"{prefix}.batch_norm.bias"].float()
    running_mean = state_dict[f"{prefix}.batch_norm.running_mean"].float()
    running_var = state_dict[f"{prefix}.batch_norm.running_var"].float()

    dw_weight = state_dict[f"{prefix}.depthwise_conv.weight"].float()
    dw_bias = state_dict[f"{prefix}.depthwise_conv.bias"].float()

    inv_std = torch.rsqrt(running_var + eps)
    scale = gamma * inv_std

    fused_weight = dw_weight * scale.view(-1, 1, 1)
    fused_bias = (dw_bias - running_mean) * scale + beta

    return fused_weight.numpy(), fused_bias.numpy()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
import re  # noqa: E402

SKIP_PATTERNS = [
    re.compile(r".*\.batch_norm\.num_batches_tracked$"),
    re.compile(r".*\.batch_norm\.weight$"),
    re.compile(r".*\.batch_norm\.bias$"),
    re.compile(r".*\.batch_norm\.running_mean$"),
    re.compile(r".*\.batch_norm\.running_var$"),
    re.compile(r"^sortformer_modules\.hidden_to_spks\."),
]


def should_skip(name: str) -> bool:
    return any(p.match(name) for p in SKIP_PATTERNS)


def validate(
    nemo_path: str,
    gguf_path: str,
    check_bn_fusion: bool = False,
    tolerance: float = 0.0,
) -> bool:
    print(f"Loading NeMo model from {nemo_path} ...")
    state_dict, config = load_nemo(nemo_path)

    print(f"Loading GGUF from {gguf_path} ...")
    metadata, gguf_tensors = read_gguf(gguf_path)

    # --- Metadata validation ---
    print("\n=== Metadata ===")
    for k, v in sorted(metadata.items()):
        print(f"  {k} = {v}")

    expected_meta = {
        "general.architecture": "sortformer",
        "sortformer.mel.n_mels": 128,
        "sortformer.mel.n_fft": 512,
        "sortformer.mel.hop_length": 160,
        "sortformer.mel.win_length": 400,
        "sortformer.mel.sample_rate": 16000,
        "sortformer.encoder.n_layers": 17,
        "sortformer.encoder.d_model": 512,
        "sortformer.encoder.n_heads": 8,
        "sortformer.transformer.n_layers": 18,
        "sortformer.transformer.d_model": 192,
        "sortformer.transformer.n_heads": 8,
        "sortformer.transformer.ff_inner": 768,
        "sortformer.n_speakers": 4,
    }
    meta_ok = True
    for k, expected in expected_meta.items():
        actual = metadata.get(k)
        if actual != expected:
            print(f"  FAIL: {k} expected={expected} actual={actual}")
            meta_ok = False
    if meta_ok:
        print("  All metadata checks PASSED")

    # --- Compute expected fused tensors ---
    n_conformer_layers = config.get("encoder", {}).get("n_layers", 17)
    fused_data: dict[str, np.ndarray] = {}
    if check_bn_fusion:
        print(f"\nComputing BatchNorm fusion for {n_conformer_layers} layers ...")
        for i in range(n_conformer_layers):
            fw, fb = fuse_batchnorm(state_dict, i)
            fused_data[f"encoder.layers.{i}.conv.depthwise_conv.weight"] = fw
            fused_data[f"encoder.layers.{i}.conv.depthwise_conv.bias"] = fb

    # --- Per-tensor validation ---
    print("\n=== Tensor Validation ===")

    # Build expected tensor list
    expected_tensors: dict[str, np.ndarray] = {}
    for name in sorted(state_dict.keys()):
        if should_skip(name):
            continue
        tensor = state_dict[name]
        if name == "preprocessor.featurizer.fb":
            expected_tensors[name] = tensor.squeeze(0).float().numpy()
        elif name == "preprocessor.featurizer.window":
            expected_tensors[name] = tensor.float().numpy()
        elif check_bn_fusion and name in fused_data:
            expected_tensors[name] = fused_data[name]
        else:
            expected_tensors[name] = tensor.float().numpy()

    # Check all expected tensors are in GGUF
    missing = set(expected_tensors.keys()) - set(gguf_tensors.keys())
    extra = set(gguf_tensors.keys()) - set(expected_tensors.keys())
    if missing:
        print(f"  MISSING from GGUF: {len(missing)} tensors")
        for m in sorted(missing):
            print(f"    - {m}")
    if extra:
        print(f"  EXTRA in GGUF (not in NeMo): {len(extra)} tensors")
        for e in sorted(extra):
            print(f"    - {e}")

    # Compare tensors
    all_pass = True
    max_errors: list[tuple[str, float]] = []
    n_compared = 0

    for name in sorted(expected_tensors.keys()):
        if name not in gguf_tensors:
            continue
        n_compared += 1

        expected = expected_tensors[name]
        gguf_t = gguf_tensors[name]
        assert gguf_t.data is not None

        # Upcast GGUF data to F32 for comparison
        actual = gguf_t.data.astype(np.float32)

        # Check shape
        if expected.shape != actual.shape:
            print(f"  FAIL [{name}]: shape mismatch expected={expected.shape} actual={actual.shape}")
            all_pass = False
            continue

        # For F16 tensors, expected was F32 -> convert expected to F16 -> F32
        # to get the same quantization rounding
        if gguf_t.ggml_type == GGML_TYPE_F16:
            if name not in fused_data:
                # Regular tensor: expected -> F16 -> F32
                expected_quantized = expected.astype(np.float16).astype(np.float32)
            else:
                # Fused tensor: fusion was done in F32, then converted to F16
                expected_quantized = expected.astype(np.float16).astype(np.float32)
        else:
            expected_quantized = expected

        max_abs_err = float(np.max(np.abs(actual - expected_quantized)))
        max_errors.append((name, max_abs_err))

        # F16 quantization should give exact match (0 error) when we apply
        # the same F32->F16->F32 roundtrip
        if max_abs_err > tolerance:
            status = "WARN" if max_abs_err < 1e-2 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {status} [{name}]: max_abs_error={max_abs_err:.6e}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"  NeMo keys: {len(state_dict)}")
    print(f"  GGUF tensors: {len(gguf_tensors)}")
    print(f"  Expected tensors (after skip): {len(expected_tensors)}")
    print(f"  Compared: {n_compared}")
    print(f"  Missing: {len(missing)}")
    print(f"  Extra: {len(extra)}")

    if max_errors:
        worst = max(max_errors, key=lambda x: x[1])
        print(f"  Worst max_abs_error: {worst[1]:.6e} ({worst[0]})")

        # Histogram of errors
        thresholds = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        for i in range(len(thresholds) - 1):
            lo, hi = thresholds[i], thresholds[i + 1]
            count = sum(1 for _, e in max_errors if lo <= e < hi)
            if count > 0:
                print(f"  [{lo:.0e}, {hi:.0e}): {count} tensors")
        count_exact = sum(1 for _, e in max_errors if e == 0.0)
        print(f"  Exact match (0.0): {count_exact} tensors")

    if all_pass and not missing:
        print("\n  ALL CHECKS PASSED!")
        return True
    else:
        print("\n  SOME CHECKS FAILED!")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GGUF SortFormer model against NeMo weights"
    )
    parser.add_argument(
        "--nemo", required=True, help="Path to input .nemo file"
    )
    parser.add_argument(
        "--gguf", required=True, help="Path to GGUF file to validate"
    )
    parser.add_argument(
        "--check-bn-fusion",
        action="store_true",
        help="Verify BatchNorm fusion correctness (compares fused values)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Maximum allowed max_abs_error per tensor (default: 0.0 for exact F16 roundtrip)",
    )
    args = parser.parse_args()

    ok = validate(args.nemo, args.gguf, args.check_bn_fusion, args.tolerance)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
