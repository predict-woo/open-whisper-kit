#!/usr/bin/env python3
"""Dump intermediate tensors from NeMo SortformerEncLabelModel at various stages.

Usage:
    python scripts/dump_nemo_intermediate.py --stage mel --audio test_audio.wav --out ref_tensors/mel.npy
    python scripts/dump_nemo_intermediate.py --stage conformer_0 --audio test_audio.wav --out ref_tensors/conformer_0.npy
    python scripts/dump_nemo_intermediate.py --stage prediction --audio test_audio.wav --out ref_tensors/prediction.npy

Supported stages:
    mel                  - Mel spectrogram from preprocessor
    pre_encoder          - Output of encoder.pre_encode (subsampled)
    conformer_N          - Output of encoder.layers[N] (N=0..16)
    projection           - Output of sortformer_modules.encoder_proj
    transformer_N        - Output of transformer_encoder.layers[N] (N=0..17)
    prediction           - Final offline prediction (sigmoid output)
    streaming_prediction - Streaming prediction output
"""

import argparse
import sys
import re

import numpy as np
import soundfile as sf
import torch


def parse_stage(stage_str: str):
    """Parse stage string, return (stage_type, layer_index_or_none)."""
    if stage_str == "mel":
        return ("mel", None)
    if stage_str == "pre_encoder":
        return ("pre_encoder", None)
    if stage_str == "projection":
        return ("projection", None)
    if stage_str == "prediction":
        return ("prediction", None)
    if stage_str == "streaming_prediction":
        return ("streaming_prediction", None)

    m = re.match(r"^conformer_(\d+)$", stage_str)
    if m:
        idx = int(m.group(1))
        if idx < 0 or idx > 16:
            print(f"ERROR: conformer layer index must be 0..16, got {idx}", file=sys.stderr)
            sys.exit(1)
        return ("conformer", idx)

    m = re.match(r"^transformer_(\d+)$", stage_str)
    if m:
        idx = int(m.group(1))
        if idx < 0 or idx > 17:
            print(f"ERROR: transformer layer index must be 0..17, got {idx}", file=sys.stderr)
            sys.exit(1)
        return ("transformer", idx)

    print(f"ERROR: unknown stage '{stage_str}'", file=sys.stderr)
    print("Valid stages: mel, pre_encoder, conformer_N (0-16), projection, "
          "transformer_N (0-17), prediction, streaming_prediction", file=sys.stderr)
    sys.exit(1)


def load_audio(audio_path: str):
    """Load audio file, return (waveform [1, samples], length tensor)."""
    data, sr = sf.read(audio_path, dtype='float32')
    # Ensure mono: if stereo, average channels
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Simple resample if not 16kHz (nearest-neighbor, sufficient for testing)
    if sr != 16000:
        import warnings
        warnings.warn(f"Audio sample rate is {sr}, expected 16000. Basic resampling applied.")
        ratio = 16000 / sr
        new_len = int(len(data) * ratio)
        indices = (np.arange(new_len) / ratio).astype(int)
        indices = np.clip(indices, 0, len(data) - 1)
        data = data[indices]
    # Convert to torch: [1, samples]
    waveform = torch.from_numpy(data).unsqueeze(0)
    audio_length = torch.tensor([waveform.shape[1]], dtype=torch.long)
    return waveform, audio_length


def main():
    parser = argparse.ArgumentParser(
        description="Dump intermediate tensors from NeMo SortformerEncLabelModel"
    )
    parser.add_argument("--stage", required=True, help="Stage to capture (see --help for list)")
    parser.add_argument("--audio", required=True, help="Path to input audio file (WAV)")
    parser.add_argument("--out", required=True, help="Output .npy file path")
    parser.add_argument("--model", default="model.nemo", help="Path to .nemo model file")
    args = parser.parse_args()

    stage_type, layer_idx = parse_stage(args.stage)

    # Load audio
    print(f"Loading audio: {args.audio}")
    waveform, audio_length = load_audio(args.audio)
    print(f"  Waveform shape: {waveform.shape}, length: {audio_length.item()} samples")

    # Load model
    print(f"Loading model: {args.model}")
    from nemo.collections.asr.models import SortformerEncLabelModel
    model = SortformerEncLabelModel.restore_from(args.model, map_location='cpu')
    model.eval()
    print("  Model loaded and set to eval mode")

    captured = {}

    def make_hook(name):
        """Create a forward hook that captures the output tensor."""
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Take the first element (the tensor), ignore masks/lengths
                captured[name] = output[0].detach().cpu()
            else:
                captured[name] = output.detach().cpu()
        return hook_fn

    # Register hooks based on stage
    hooks = []

    if stage_type == "mel":
        h = model.preprocessor.register_forward_hook(make_hook("mel"))
        hooks.append(h)

    elif stage_type == "pre_encoder":
        h = model.encoder.pre_encode.register_forward_hook(make_hook("pre_encoder"))
        hooks.append(h)

    elif stage_type == "conformer":
        h = model.encoder.layers[layer_idx].register_forward_hook(make_hook(f"conformer_{layer_idx}"))
        hooks.append(h)

    elif stage_type == "projection":
        h = model.sortformer_modules.encoder_proj.register_forward_hook(make_hook("projection"))
        hooks.append(h)

    elif stage_type == "transformer":
        h = model.transformer_encoder.layers[layer_idx].register_forward_hook(make_hook(f"transformer_{layer_idx}"))
        hooks.append(h)

    # For prediction/streaming_prediction, no hooks needed — we capture the output directly

    # Run forward pass
    # For non-streaming stages, force offline mode to get full-audio intermediate tensors
    # (NeMo's streaming_mode=True causes model.forward() to use streaming internally,
    #  which processes chunks and hooks only capture the last chunk's output)
    if stage_type != "streaming_prediction":
        model.streaming_mode = False
        print("  Forced offline mode (streaming_mode=False) for deterministic intermediate capture")

    print(f"Running forward pass (stage: {args.stage})...")
    with torch.no_grad():
        if stage_type == "streaming_prediction":
            # Use streaming forward
            preds = model.forward_streaming(
                audio_signal=waveform,
                audio_signal_length=audio_length,
            )
            captured["streaming_prediction"] = preds.detach().cpu()
        else:
            # Use standard offline forward
            preds = model.forward(
                audio_signal=waveform,
                audio_signal_length=audio_length,
            )
            if stage_type == "prediction":
                captured["prediction"] = preds.detach().cpu()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Get the captured tensor
    key = args.stage
    if key not in captured:
        print(f"ERROR: Failed to capture tensor for stage '{args.stage}'", file=sys.stderr)
        print(f"  Available captures: {list(captured.keys())}", file=sys.stderr)
        sys.exit(1)

    tensor = captured[key]
    arr = tensor.numpy()

    # Save
    np.save(args.out, arr)
    print(f"\nSaved: {args.out}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min: {arr.min():.6f}, Max: {arr.max():.6f}, Mean: {arr.mean():.6f}")


if __name__ == "__main__":
    main()
