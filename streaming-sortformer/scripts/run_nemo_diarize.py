#!/usr/bin/env python3
"""Run NeMo SortformerEncLabelModel inference and produce RTTM output.

Supports both offline and streaming modes.

Usage:
    python scripts/run_nemo_diarize.py --audio test_audio.wav --out test.rttm
    python scripts/run_nemo_diarize.py --audio test_audio.wav --out test_streaming.rttm --streaming

RTTM format: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch


# Model config constants
SAMPLE_RATE = 16000
FRAME_SHIFT_SAMPLES = 1280  # 80ms = 0.01s window_stride * subsampling_factor(8) * 16000
FRAME_SHIFT_SEC = FRAME_SHIFT_SAMPLES / SAMPLE_RATE  # 0.08s per frame


def load_audio(audio_path: str):
    """Load audio file, return (waveform [1, samples], length tensor)."""
    data, sr = sf.read(audio_path, dtype='float32')
    # Ensure mono
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Simple resample if not 16kHz
    if sr != SAMPLE_RATE:
        import warnings
        warnings.warn(f"Audio sample rate is {sr}, expected {SAMPLE_RATE}. Basic resampling applied.")
        ratio = SAMPLE_RATE / sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        indices = np.clip(indices, 0, len(data) - 1)
        data = data[indices]
    # Convert to torch: [1, samples]
    waveform = torch.from_numpy(data).unsqueeze(0)
    audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long)
    return waveform, audio_length


def preds_to_rttm(preds: np.ndarray, threshold: float, file_id: str) -> list:
    """Convert prediction matrix to RTTM lines.

    Args:
        preds: (T, num_spks) sigmoid predictions
        threshold: activity threshold
        file_id: file identifier for RTTM

    Returns:
        List of RTTM line strings
    """
    num_frames, num_spks = preds.shape
    rttm_lines = []

    for spk_idx in range(num_spks):
        spk_label = f"speaker_{spk_idx}"
        in_segment = False
        seg_start = 0

        for t in range(num_frames):
            active = preds[t, spk_idx] >= threshold

            if active and not in_segment:
                # Start new segment
                seg_start = t
                in_segment = True
            elif not active and in_segment:
                # End segment
                start_sec = seg_start * FRAME_SHIFT_SEC
                dur_sec = (t - seg_start) * FRAME_SHIFT_SEC
                if dur_sec > 0:
                    rttm_lines.append(
                        f"SPEAKER {file_id} 1 {start_sec:.3f} {dur_sec:.3f} "
                        f"<NA> <NA> {spk_label} <NA> <NA>"
                    )
                in_segment = False

        # Close any open segment at end
        if in_segment:
            start_sec = seg_start * FRAME_SHIFT_SEC
            dur_sec = (num_frames - seg_start) * FRAME_SHIFT_SEC
            if dur_sec > 0:
                rttm_lines.append(
                    f"SPEAKER {file_id} 1 {start_sec:.3f} {dur_sec:.3f} "
                    f"<NA> <NA> {spk_label} <NA> <NA>"
                )

    # Sort by start time
    def sort_key(line):
        parts = line.split()
        return float(parts[3])

    rttm_lines.sort(key=sort_key)
    return rttm_lines


def main():
    parser = argparse.ArgumentParser(
        description="Run NeMo SortformerEncLabelModel diarization"
    )
    parser.add_argument("--audio", required=True, help="Input audio file (WAV)")
    parser.add_argument("--out", required=True, help="Output RTTM file")
    parser.add_argument("--model", default="model.nemo", help="Path to .nemo model")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Speaker activity threshold (default: 0.5)")
    # Streaming parameters (match model config defaults)
    parser.add_argument("--chunk-len", type=int, default=None,
                        help="Streaming chunk length in frames (default: from model config)")
    parser.add_argument("--right-context", type=int, default=None,
                        help="Right context frames (default: from model config)")
    parser.add_argument("--fifo-len", type=int, default=None,
                        help="FIFO buffer length (default: from model config)")
    parser.add_argument("--spkcache-len", type=int, default=None,
                        help="Speaker cache length (default: from model config)")
    args = parser.parse_args()

    # Load audio
    print(f"Loading audio: {args.audio}")
    waveform, audio_length = load_audio(args.audio)
    duration_sec = waveform.shape[1] / SAMPLE_RATE
    print(f"  Duration: {duration_sec:.2f}s, Samples: {waveform.shape[1]}")

    # Load model
    print(f"Loading model: {args.model}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(args.model, map_location='cpu')
    model.eval()
    print("  Model loaded and set to eval mode")

    # Override streaming params if provided
    if args.chunk_len is not None:
        model.sortformer_modules.chunk_len = args.chunk_len
    if args.right_context is not None:
        model.sortformer_modules.chunk_right_context = args.right_context
    if args.fifo_len is not None:
        model.sortformer_modules.fifo_len = args.fifo_len
    if args.spkcache_len is not None:
        model.sortformer_modules.spkcache_len = args.spkcache_len

    # Run inference
    mode_str = "streaming" if args.streaming else "offline"
    print(f"Running {mode_str} inference...")

    with torch.no_grad():
        if args.streaming:
            preds = model.forward_streaming(
                audio_signal=waveform,
                audio_signal_length=audio_length,
            )
        else:
            preds = model.forward(
                audio_signal=waveform,
                audio_signal_length=audio_length,
            )

    # Convert to numpy: (batch, T, num_spks) → (T, num_spks)
    preds_np = preds[0].cpu().numpy()
    print(f"  Predictions shape: {preds_np.shape}")
    print(f"  Value range: [{preds_np.min():.4f}, {preds_np.max():.4f}]")

    # Generate RTTM
    file_id = os.path.splitext(os.path.basename(args.audio))[0]
    rttm_lines = preds_to_rttm(preds_np, args.threshold, file_id)

    # Write RTTM
    with open(args.out, 'w') as f:
        for line in rttm_lines:
            f.write(line + "\n")

    print(f"\nOutput: {args.out}")
    print(f"  Total RTTM segments: {len(rttm_lines)}")

    # Print summary per speaker
    from collections import defaultdict
    spk_dur = defaultdict(float)
    spk_segs = defaultdict(int)
    for line in rttm_lines:
        parts = line.split()
        spk = parts[7]
        dur = float(parts[4])
        spk_dur[spk] += dur
        spk_segs[spk] += 1

    speakers_detected = len(spk_dur)
    print(f"\n  Speakers detected: {speakers_detected}")
    print(f"  Threshold: {args.threshold}")
    for spk in sorted(spk_dur.keys()):
        print(f"    {spk}: {spk_dur[spk]:.2f}s ({spk_segs[spk]} segments)")

    if not rttm_lines:
        print(f"\n  NOTE: No speech detected above threshold {args.threshold}.")
        print(f"  Try lowering --threshold (current predictions range: "
              f"[{preds_np.min():.4f}, {preds_np.max():.4f}])")


if __name__ == "__main__":
    main()
