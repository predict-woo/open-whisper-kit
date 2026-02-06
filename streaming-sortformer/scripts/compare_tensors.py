#!/usr/bin/env python3
"""Compare two tensor files (reference .npy vs test .npy or raw float32 binary).

Usage:
    python scripts/compare_tensors.py ref_tensors/mel.npy test_tensors/mel.npy
    python scripts/compare_tensors.py ref_tensors/mel.npy test_output/mel.bin --shape 1,2000,128 --tolerance 1e-2

Exit code: 0 = PASS, 1 = FAIL
"""

import argparse
import os
import sys

import numpy as np


def load_tensor(path: str, shape: tuple = None) -> np.ndarray:
    """Load tensor from .npy or raw float32 binary file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    else:
        # Treat as raw float32 binary
        raw = np.fromfile(path, dtype=np.float32)
        if shape is not None:
            arr = raw.reshape(shape)
        else:
            arr = raw
        return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between flattened arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a < 1e-12 or norm_b < 1e-12:
        # Both near-zero → consider identical
        if norm_a < 1e-12 and norm_b < 1e-12:
            return 1.0
        return 0.0

    return float(dot / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(description="Compare two tensor files")
    parser.add_argument("reference", help="Reference tensor file (.npy)")
    parser.add_argument("test", help="Test tensor file (.npy or raw float32 binary)")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Max absolute error tolerance (default: 1e-3)")
    parser.add_argument("--shape", type=str, default=None,
                        help="Reshape raw binary to this shape (e.g., '1,250,4')")
    args = parser.parse_args()

    # Parse shape
    shape = None
    if args.shape:
        shape = tuple(int(x) for x in args.shape.split(","))

    # Load tensors
    print(f"Reference: {args.reference}")
    ref = load_tensor(args.reference)
    print(f"  Shape: {ref.shape}, Dtype: {ref.dtype}")

    print(f"Test:      {args.test}")
    test = load_tensor(args.test, shape=shape)
    print(f"  Shape: {test.shape}, Dtype: {test.dtype}")

    # Shape check
    if ref.shape != test.shape:
        print(f"\nFAIL: Shape mismatch — reference {ref.shape} vs test {test.shape}")
        sys.exit(1)

    # Cast to float64 for comparison
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)

    # Compute metrics
    abs_diff = np.abs(ref_f - test_f)
    max_abs_error = float(abs_diff.max())
    mean_abs_error = float(abs_diff.mean())

    cos_sim = cosine_similarity(ref_f, test_f)

    # Relative error (avoid division by zero)
    ref_abs = np.abs(ref_f)
    mask = ref_abs > 1e-8
    if mask.any():
        rel_error = float((abs_diff[mask] / ref_abs[mask]).max())
    else:
        rel_error = 0.0 if max_abs_error < 1e-8 else float('inf')

    # Print results
    print(f"\n{'='*50}")
    print(f"  Max Absolute Error:  {max_abs_error:.6e}")
    print(f"  Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"  Cosine Similarity:   {cos_sim:.10f}")
    print(f"  Max Relative Error:  {rel_error:.6e}")
    print(f"  Tolerance:           {args.tolerance:.6e}")
    print(f"{'='*50}")

    # PASS/FAIL
    passed = (max_abs_error <= args.tolerance) and (cos_sim > 0.99)
    if passed:
        print("PASS")
        sys.exit(0)
    else:
        reasons = []
        if max_abs_error > args.tolerance:
            reasons.append(f"max_abs_error ({max_abs_error:.6e}) > tolerance ({args.tolerance:.6e})")
        if cos_sim <= 0.99:
            reasons.append(f"cosine_sim ({cos_sim:.6f}) <= 0.99")
        print(f"FAIL: {'; '.join(reasons)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
