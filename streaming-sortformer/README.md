# sortformer-ggml

Native C/C++ implementation of NVIDIA's **SortFormer** streaming speaker diarization model using the [GGML](https://github.com/ggml-org/ggml) tensor library.

Based on [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_streaming_sortformer_4spk-v2) (117M parameters). No PyTorch, no ONNX runtime, no large dependencies — just GGML.

## Features

- **Streaming inference** with configurable latency parameters
- **Up to 4 speakers** detected simultaneously
- **CoreML/ANE acceleration** on Apple Silicon (~110x real-time)
- **F16 weights, F32 computation** — 235 MB model file (or 97 MB with Q4_K quantization)
- **RTTM output** (standard diarization format) or raw frame-level probabilities
- **Debug dump modes** for intermediate tensors at every pipeline stage
- **C library API** (`libsortformer.a`) for embedding in other applications

## Performance

| Platform | Model | Speed | Memory |
|----------|-------|-------|--------|
| Apple Silicon (CoreML/ANE) | F16 | **~110x real-time** | ~300 MB |
| Apple Silicon (CPU only) | Q4_K | ~13x real-time | ~240 MB |
| Apple Silicon (CPU only) | F16 | ~10x real-time | ~380 MB |
| 4-core x86 CPU | Q4_K | ~8x real-time | ~240 MB |

*Tested on M3 MacBook Pro. 45-minute audio processed in 24 seconds with CoreML/ANE.*

## Build

Requirements: CMake >= 3.14, C++17 compiler.

```bash
git clone --recursive https://github.com/predict-woo/streaming-sortformer-ggml.git
cd sortformer-ggml
cmake -B build -DSORTFORMER_COREML=ON   # Enable CoreML (macOS only)
cmake --build build -j$(nproc)
```

Build options:
- `-DSORTFORMER_COREML=ON` — Enable CoreML/ANE acceleration (macOS only, recommended)
- `-DSORTFORMER_COREML=OFF` — CPU-only build (default, works on all platforms)

This produces:
- `build/libsortformer.a` — static library
- `build/sortformer-diarize` — CLI binary

## Model Conversion

### GGUF Model (Required)

Download the NeMo model from [NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_streaming_sortformer_4spk-v2) and convert it to GGUF:

```bash
python scripts/convert_to_gguf.py --nemo model.nemo --out model.gguf
```

The conversion script requires Python with `torch`, `numpy`, and `pyyaml`. It:
- Extracts weights from the `.nemo` archive (plain tar)
- Converts all weights to F16 (mel filterbank and Hann window stay F32)
- Fuses BatchNorm parameters into depthwise convolution weights
- Writes a GGUF v3 file with 903 tensors and 20 metadata keys (~235 MB)

### CoreML Model (Optional, for ANE acceleration)

To enable CoreML/ANE acceleration, convert the head (conformer + transformer + prediction) to CoreML:

```bash
# Install dependencies
pip install torch coremltools nemo_toolkit[asr]

# Export CoreML model
python scripts/convert_head_to_coreml.py --model model.nemo --output model-coreml-head.mlpackage --precision fp16

# Compile for deployment
xcrun coremlcompiler compile model-coreml-head.mlpackage .
```

This creates `model-coreml-head.mlmodelc/` which the CLI will automatically load when present.

**Architecture with CoreML:**
```
Audio -> [GGML: Mel + Pre-Encoder] -> [CoreML/ANE: Conformer + Transformer + Prediction]
              ~620ms (6%)                              ~450ms (94% on ANE)
```

The heavy compute (conformer + transformer) runs on the Apple Neural Engine, achieving 11x speedup over CPU.

## Model Quantization

For faster CPU inference and lower memory usage, quantize the F16 model:

```bash
./build/sortformer-quantize model.gguf model_q4k.gguf q4_k
```

Supported quantization types:
- `q4_k` — 4-bit K-quant (recommended for CPU, ~97 MB)
- `q5_k` — 5-bit K-quant (~120 MB)
- `q8_0` — 8-bit uniform quant (~180 MB)

*Note: Quantization is for CPU inference only. CoreML uses F16 and is faster than quantized CPU.*

## Usage

```bash
# Streaming diarization, RTTM to stdout
./build/sortformer-diarize -m model.gguf -f audio.wav --streaming

# RTTM to file
./build/sortformer-diarize -m model.gguf -f audio.wav --streaming -o output.rttm

# Raw frame-level probabilities (T lines, 4 floats each)
./build/sortformer-diarize -m model.gguf -f audio.wav --streaming --probs

# Custom speaker activity threshold
./build/sortformer-diarize -m model.gguf -f audio.wav --streaming --threshold 0.3

# Multi-threaded (CPU only)
./build/sortformer-diarize -m model.gguf -f audio.wav --streaming --threads 8
```

Input audio must be **16 kHz mono WAV** (16-bit PCM).

The CLI automatically uses CoreML if `model-coreml-head.mlmodelc` is present alongside `model.gguf`.

## Streaming Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-len` | 188 | Chunk length in frames |
| `--right-context` | 1 | Right context frames per chunk |
| `--fifo-len` | 0 | FIFO buffer length (0 = disabled) |
| `--spkcache-len` | 188 | Speaker cache length in frames |
| `--threshold` | 0.5 | Speaker activity detection threshold |
| `--median-filter` | 11 | Median filter window for smoothing |
| `--threads` | 4 | Number of compute threads |

## Validation

A 120-second test audio file (`test.wav`) and reference RTTM output are included for validation.

```bash
# Run C++ streaming diarization
./build/sortformer-diarize -m model.gguf -f test.wav --streaming -o output.rttm

# Compare with NeMo reference
python scripts/compare_rttm.py nemo_streaming.rttm output.rttm
```

Results (C++ vs NeMo streaming):

| Collar | DER | Missed | False Alarm | Confusion |
|--------|-----|--------|-------------|-----------|
| 0.25s (standard) | **0.00%** | 0 | 0 | 0 |
| 0.00s (strictest) | 7.53% | 440 frames | 176 frames | 0 |

Zero speaker confusion at all collar values. The 7.53% DER at zero collar is entirely from minor boundary timing differences due to F16 quantization.

## Architecture

The SortFormer pipeline processes audio through six stages:

```
Audio (16kHz) -> Mel Spectrogram (128 bins)
             -> Pre-Encoder (Conv2D, 8x subsampling)
             -> Fast-Conformer (17 layers, d=512, 8 heads)
             -> Projection (512 -> 192)
             -> Transformer Encoder (18 layers, d=192, 8 heads)
             -> Prediction Head (4 speaker outputs)
```

In streaming mode, audio is processed in chunks. A speaker cache (AOSC compression) maintains context across chunks, enabling the model to track speakers over long recordings. Memory usage plateaus after the first chunk regardless of audio length.

## Debug Dump Modes

Dump intermediate tensors as raw float32 binary files:

```bash
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-mel           # -> cpp_mel.raw
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-preenc        # -> cpp_preenc.raw
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-conformer 16  # -> cpp_conf16.raw
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-projection    # -> cpp_proj.raw
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-transformer 17 # -> cpp_trans17.raw
./build/sortformer-diarize -m model.gguf -f audio.wav --dump-prediction    # -> cpp_pred.raw
```

## RTTM Output Format

The output follows the standard [RTTM](https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf) format:

```
SPEAKER <file> 1 <start> <duration> <NA> <NA> speaker_<id> <NA> <NA>
```

Each line represents a contiguous speech segment for one speaker.

## Library API

The C API is defined in `src/sortformer.h`. Key functions:

```c
struct sortformer_context * sortformer_init(const char * model_path, struct sortformer_params params);
void sortformer_free(struct sortformer_context * ctx);

int sortformer_diarize(struct sortformer_context * ctx,
                       const float * audio_samples, int n_samples,
                       float * probs_out, int n_frames_max);

int sortformer_to_rttm(const float * probs, int n_frames,
                       float threshold, int median_filter,
                       const char * filename,
                       char * rttm_out, int rttm_out_size);
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/convert_to_gguf.py` | Convert NeMo model to GGUF |
| `scripts/convert_head_to_coreml.py` | Convert head to CoreML for ANE acceleration |
| `scripts/compare_tensors.py` | Compare intermediate tensors |
| `scripts/compare_rttm.py` | Compute Diarization Error Rate |
| `scripts/dump_nemo_intermediate.py` | Dump NeMo tensors for comparison |
| `scripts/run_nemo_diarize.py` | Run NeMo reference inference |
| `scripts/validate_gguf.py` | Validate GGUF file integrity |

## Acknowledgments

- [GGML](https://github.com/ggml-org/ggml) — MIT License
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — Apache 2.0
- SortFormer model: [Park et al., "SortFormer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens"](https://arxiv.org/abs/2409.06656)
