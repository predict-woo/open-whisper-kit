#!/usr/bin/env python3
"""Dump per-submodule intermediate tensors from a single conformer layer.

Dumps the residual state after each sub-module:
  - after_ffn1: residual after FFN1 (Macaron half-step)
  - after_mhsa: residual after MHSA
  - after_conv: residual after Conv module
  - after_ffn2: residual after FFN2 (Macaron half-step)
  - after_norm: output after final LayerNorm (= layer output)

Also dumps the input to the layer (after xscaling for layer 0).

Usage:
    python scripts/dump_conformer_submodules.py --layer 0 --audio test_audio.wav --outdir ref_tensors/
"""

import argparse
import sys
import numpy as np
import soundfile as sf
import torch


def load_audio(audio_path: str):
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        indices = np.clip(indices, 0, len(data) - 1)
        data = data[indices]
    waveform = torch.from_numpy(data).unsqueeze(0)
    audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long)
    return waveform, audio_length


def main():
    parser = argparse.ArgumentParser(description="Dump conformer sub-module intermediates")
    parser.add_argument("--layer", type=int, required=True, help="Conformer layer index (0-16)")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--outdir", default="ref_tensors", help="Output directory")
    parser.add_argument("--model", default="model.nemo", help="Path to .nemo model file")
    args = parser.parse_args()

    layer_idx = args.layer
    assert 0 <= layer_idx <= 16, f"Layer must be 0-16, got {layer_idx}"

    print(f"Loading audio: {args.audio}")
    waveform, audio_length = load_audio(args.audio)
    print(f"  Waveform: {waveform.shape}, length: {audio_length.item()}")

    print(f"Loading model: {args.model}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(args.model, map_location='cpu')
    model.eval()
    model.streaming_mode = False

    # Monkey-patch the target conformer layer's forward to capture intermediates
    target_layer = model.encoder.layers[layer_idx]
    captured = {}

    original_forward = target_layer.forward

    def patched_forward(x, att_mask=None, pos_emb=None, pad_mask=None,
                        cache_last_channel=None, cache_last_time=None):
        # Save input
        captured['input'] = x.detach().cpu().numpy()

        # FFN1
        residual = x
        fx = target_layer.norm_feed_forward1(x)
        fx = target_layer.feed_forward1(fx)
        residual = residual + fx * target_layer.fc_factor  # no dropout in eval
        captured['after_ffn1'] = residual.detach().cpu().numpy()

        # MHSA
        sx = target_layer.norm_self_att(residual)
        sx = target_layer.self_attn(query=sx, key=sx, value=sx, mask=att_mask, pos_emb=pos_emb)
        residual = residual + sx  # no dropout in eval
        captured['after_mhsa'] = residual.detach().cpu().numpy()

        # Conv
        cx = target_layer.norm_conv(residual)
        cx = target_layer.conv(cx, pad_mask=pad_mask)
        residual = residual + cx  # no dropout in eval
        captured['after_conv'] = residual.detach().cpu().numpy()

        # FFN2
        f2x = target_layer.norm_feed_forward2(residual)
        f2x = target_layer.feed_forward2(f2x)
        residual = residual + f2x * target_layer.fc_factor  # no dropout in eval
        captured['after_ffn2'] = residual.detach().cpu().numpy()

        # Final LayerNorm
        out = target_layer.norm_out(residual)
        captured['after_norm'] = out.detach().cpu().numpy()

        return out

    target_layer.forward = patched_forward

    print(f"Running forward pass...")
    with torch.no_grad():
        preds = model.forward(audio_signal=waveform, audio_signal_length=audio_length)

    import os
    os.makedirs(args.outdir, exist_ok=True)

    for name, arr in captured.items():
        outpath = os.path.join(args.outdir, f"conf{layer_idx}_{name}.npy")
        np.save(outpath, arr)
        print(f"  Saved {outpath}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")

    print("Done!")


if __name__ == "__main__":
    main()
