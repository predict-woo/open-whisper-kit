# AGENTS.md

Guidance for AI coding agents working on this project.

## Project Overview

**sortformer-ggml** is a native C/C++ implementation of NVIDIA's SortFormer streaming speaker diarization model using the GGML tensor library. It converts audio into per-frame speaker activity probabilities for up to 4 speakers, outputting standard RTTM diarization format.

- **Language**: C++17 (core library), C API (public header), Python (scripts)
- **Build system**: CMake >= 3.14
- **Dependencies**: GGML (git submodule at `ggml/`, pinned to commit `a8db410a`)
- **No external runtime dependencies** (no PyTorch, no ONNX, no protobuf)

## Repository Structure

```
sortformer-ggml/
  ggml/                    # GGML submodule (tensor computation library)
  src/
    sortformer.h           # Public C API (136 lines)
    sortformer.cpp         # Core inference implementation (~2200 lines)
    sortformer-cli.cpp     # CLI binary (~830 lines)
  scripts/
    convert_to_gguf.py     # NeMo -> GGUF model converter
    validate_gguf.py       # GGUF file validator
    compare_tensors.py     # Tensor comparison (cosine, max error)
    compare_rttm.py        # RTTM/DER comparison
    dump_nemo_intermediate.py  # NeMo intermediate tensor dumper
    run_nemo_diarize.py    # Full NeMo reference inference
    dump_conformer_submodules.py  # Per-submodule conformer debugging
  CMakeLists.txt           # Build configuration
  README.md                # User-facing documentation
```

## Build Instructions

```bash
cmake -B build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF
cmake --build build -j$(nproc)
```

Produces `build/libsortformer.a` (static library) and `build/sortformer-diarize` (CLI).

## Architecture

The inference pipeline has six sequential stages:

1. **Mel Spectrogram** (`sortformer_compute_mel`) — 128 mel bins, 16kHz, hop=160, win=400, n_fft=512
2. **Pre-Encoder** (`sortformer_compute_preenc`) — Conv2D stack with 8x subsampling, output dim 512
3. **Fast-Conformer** (`sortformer_compute_conformer`) — 17 layers, d_model=512, 8 heads, relative position encoding (interleaved sin/cos), depthwise conv with fused BatchNorm
4. **Projection** (`sortformer_compute_projection`) — Linear 512 -> 192
5. **Transformer Encoder** (`sortformer_compute_transformer`) — 18 layers, d_model=192, 8 heads, post-layer-norm, ReLU activation
6. **Prediction Head** (`sortformer_compute_prediction`) — ReLU -> Linear(192,192) -> ReLU -> Linear(192,4) -> Sigmoid

### Streaming Pipeline

`sortformer_diarize()` implements the full streaming pipeline:
- Audio is chunked into mel segments (default chunk_len=188 frames)
- Each chunk is pre-encoded independently
- Conformer/transformer/prediction process concatenated [spkcache, fifo, chunk]
- AOSC (Adaptive Online Speaker Cache) compression maintains speaker context
- FIFO buffer length is 0 by default (disabled)

## Code Conventions

### C++ Style
- C++17 standard, no exceptions, no RTTI
- All public API functions use C linkage (`extern "C"`)
- Memory: caller-frees pattern (functions allocate via `malloc`, caller calls `free`)
- Error handling: return -1 on error, print to stderr via `fprintf(stderr, ...)`
- All GGML weights are F16 in the GGUF file; cast to F32 before computation
- Prefix all public symbols with `sortformer_`
- Internal functions are `static`

### GGML Patterns
- `ggml_mul_mat(weight, x)` computes `x @ weight^T` (weight is first argument)
- Bias broadcasting: reshape bias to match tensor dimensions before `ggml_add`
- Conv2D kernel layout: `ne[0]=KW, ne[1]=KH, ne[2]=IC, ne[3]=OC`
- Graph allocation: `ggml_new_graph_custom(ctx, node_count, false)`
- Backend: CPU only (`ggml_backend_cpu_init`)

### Naming
- Conformer layer weights: `encoder.layers.{N}.{submodule}.{param}`
- Transformer layer weights: `transformer_encoder.layers.{N}.{submodule}.{param}`
- Pre-encoder weights: `encoder.pre_encode.conv.{N}.{param}`

## Key Technical Details

### Position Embeddings
NeMo uses **interleaved** sinusoidal position encoding: `[sin(f0), cos(f0), sin(f1), cos(f1), ...]`. This is NOT the concatenated layout used by many other implementations.

### BatchNorm Fusion
All conformer depthwise conv BatchNorm parameters are fused into conv weights during GGUF conversion:
- `fused_w = orig_w * gamma / sqrt(running_var + eps)`
- `fused_b = (orig_b - running_mean) * gamma / sqrt(running_var + eps) + beta`

### Streaming State
- `spkcache`: speaker cache embeddings (up to spkcache_len frames of d_model=512)
- `spkcache_preds`: corresponding predictions for AOSC compression
- `fifo`: FIFO buffer (length 0 by default, effectively disabled)
- AOSC compression fires when spkcache exceeds its limit, keeping strong/weak speaker frames

### Precision
- All weights stored as F16 in GGUF
- Computation in F32
- Cosine similarity vs NeMo reference: 0.999+ at every pipeline stage
- Max absolute error from F16 quantization: ~4.37 (pre-encoder), ~4.8e-5 (prediction)

## Testing and Validation

There is no formal test framework. Validation is done via comparison scripts:

```bash
# Dump NeMo reference tensors
python scripts/dump_nemo_intermediate.py --nemo model.nemo --audio test.wav --stage mel

# Dump C++ tensors
./build/sortformer-diarize -m model.gguf -f test.wav --dump-mel

# Compare
python scripts/compare_tensors.py ref_tensors/mel.npy cpp_mel.raw
```

Expected cosine similarities: mel=1.0, pre_encoder=0.9998, conformer=0.9991, projection=0.9990, transformer=0.9998, prediction=0.9979, streaming_e2e=0.9999.

### Streaming End-to-End Validation

A 120-second test audio (`test.wav`) with reference RTTM outputs is included in the repo:

```bash
# C++ streaming
./build/sortformer-diarize -m model.gguf -f test.wav --streaming -o output.rttm

# NeMo reference streaming (requires .venv with NeMo)
.venv/bin/python scripts/run_nemo_diarize.py --audio test.wav --out nemo_streaming.rttm --model model.nemo --streaming

# Compare (DER)
.venv/bin/python scripts/compare_rttm.py nemo_streaming.rttm output.rttm
```

Results: **0.00% DER** at standard 0.25s collar, 7.53% at 0s collar (boundary timing only, zero speaker confusion). The streaming pipeline processes 120s audio in 8 chunks (~29s wall time, 4.2x real-time). Memory plateaus after chunk 0 — AOSC compression keeps spkcache bounded at 188 frames regardless of audio length.

## Common Pitfalls

1. **Do not use `git add .`** — many large generated files (model, tensors, build) must stay untracked
2. **GGML graph size** — conformer needs ~2126 nodes, transformer ~954 nodes; use `ggml_new_graph_custom` with sufficient capacity
3. **Mel seq_len vs padded length** — mel is padded to multiple of 16, but streaming must use `seq_len` (unpadded valid frames) to get correct frame counts
4. **F16 weight casting** — always cast F16 weights to F32 before use; `ggml_conv_2d_dw_direct` assumes F32 kernels
5. **Python scripts require NeMo** — the comparison/dump scripts need the full NeMo environment (torch, nemo, soundfile); use the `.venv` with `uv`
6. **GGML context memory must scale with T²** — conformer and transformer allocate `3GB + T²×N` bytes; hardcoded sizes fail when streaming concatenates spkcache (188) + chunk (~190) = T=378
7. **NeMo `forward_streaming()` takes mel features, not raw audio** — call `model.preprocessor(input_signal, length)` first, then pass `processed_signal` to `forward_streaming()`
