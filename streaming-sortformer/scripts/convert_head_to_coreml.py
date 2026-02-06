#!/usr/bin/env python3
"""
Convert SortFormer head (conformer + transformer + prediction) to CoreML.

This exports ONLY the head, not the pre-encoder. The GGML code handles
mel spectrogram and pre-encoder computation, then passes pre-encoder
embeddings to this CoreML model for the heavy compute.

Input:  pre_encoder_embs [1, T, 512] - concatenated spkcache + fifo + chunk embeddings
        pre_encoder_lengths [1] - valid length
Output: speaker_preds [1, T, 4] - per-frame speaker probabilities
"""

import os
import argparse
import torch
import numpy as np
import coremltools as ct

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from nemo.collections.asr.models import SortformerEncLabelModel


class HeadOnlyWrapper(torch.nn.Module):
    """
    Wraps conformer encoder + transformer encoder + prediction head.
    
    Takes pre-encoder embeddings (already concatenated spkcache + fifo + chunk)
    and returns speaker predictions.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pre_encoder_embs, pre_encoder_lengths):
        # Run conformer encoder (bypass_pre_encode=True skips the pre-encoder)
        fc_encoder_embs, fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=pre_encoder_embs,
            processed_signal_length=pre_encoder_lengths,
            bypass_pre_encode=True,
        )
        
        # Run transformer encoder + prediction head
        preds = self.model.forward_infer(fc_encoder_embs, fc_encoder_lengths)
        
        return preds


def export_head_only(
    model_path: str,
    output_path: str,
    max_seq_len: int = 400,
    precision: str = "fp16"
):
    """
    Export head-only CoreML model.
    
    Args:
        model_path: Path to .nemo model or HuggingFace model name
        output_path: Output .mlpackage path
        max_seq_len: Maximum sequence length (spkcache + fifo + chunk)
                     Default 378 = 188 (spkcache) + 0 (fifo) + 190 (chunk)
        precision: "fp16" or "fp32"
    """
    print("=" * 60)
    print("Exporting SortFormer Head-Only Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Max seq len: {max_seq_len}")
    print(f"Precision: {precision}")
    print("=" * 60)
    
    # Load model
    print("\nLoading NeMo model...")
    if os.path.exists(model_path):
        model = SortformerEncLabelModel.restore_from(model_path, map_location="cpu")
    else:
        model = SortformerEncLabelModel.from_pretrained(model_path, map_location="cpu")
    model.eval()
    
    # Get dimensions
    fc_d_model = model.sortformer_modules.fc_d_model  # 512
    n_speakers = 4
    
    print(f"fc_d_model: {fc_d_model}")
    print(f"n_speakers: {n_speakers}")
    
    # Create wrapper
    wrapper = HeadOnlyWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    dummy_embs = torch.randn(1, max_seq_len, fc_d_model)
    dummy_lengths = torch.tensor([max_seq_len], dtype=torch.long)
    
    # Trace
    print("\nTracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_embs, dummy_lengths))
    
    # Convert to CoreML
    print("\nConverting to CoreML...")
    compute_precision = ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="pre_encoder_embs",
                shape=(1, max_seq_len, fc_d_model),
                dtype=np.float32
            ),
            ct.TensorType(
                name="pre_encoder_lengths", 
                shape=(1,),
                dtype=np.int32
            ),
        ],
        outputs=[
            ct.TensorType(name="speaker_preds", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Add metadata
    mlmodel.author = "sortformer-ggml"
    mlmodel.license = "Apache-2.0"
    mlmodel.version = "1.0"
    mlmodel.short_description = "SortFormer head (conformer + transformer + prediction) for speaker diarization"
    
    mlmodel.user_defined_metadata["max_seq_len"] = str(max_seq_len)
    mlmodel.user_defined_metadata["fc_d_model"] = str(fc_d_model)
    mlmodel.user_defined_metadata["n_speakers"] = str(n_speakers)
    mlmodel.user_defined_metadata["precision"] = precision
    
    mlmodel.input_description["pre_encoder_embs"] = f"Pre-encoder embeddings [1, T, {fc_d_model}], T <= {max_seq_len}"
    mlmodel.input_description["pre_encoder_lengths"] = "Valid sequence length"
    mlmodel.output_description["speaker_preds"] = "Speaker probabilities [1, T, 4]"
    
    # Save
    print(f"\nSaving to {output_path}...")
    mlmodel.save(output_path)
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nTo compile for deployment:")
    print(f"  xcrun coremlcompiler compile {output_path} .")
    print(f"\nThen rename the output directory to model-coreml-head.mlmodelc")
    
    return mlmodel


def verify_model(mlmodel_path: str, nemo_model_path: str, seq_len: int = 378):
    """Verify CoreML model against NeMo reference."""
    print("\n" + "=" * 60)
    print("Verifying CoreML model...")
    print("=" * 60)
    
    # Load models
    mlmodel = ct.models.MLModel(mlmodel_path)
    
    if os.path.exists(nemo_model_path):
        nemo_model = SortformerEncLabelModel.restore_from(nemo_model_path, map_location="cpu")
    else:
        nemo_model = SortformerEncLabelModel.from_pretrained(nemo_model_path, map_location="cpu")
    nemo_model.eval()
    
    wrapper = HeadOnlyWrapper(nemo_model)
    wrapper.eval()
    
    # Create test input
    fc_d_model = nemo_model.sortformer_modules.fc_d_model
    test_embs = torch.randn(1, seq_len, fc_d_model)
    test_lengths = torch.tensor([seq_len], dtype=torch.long)
    
    # Run NeMo
    with torch.no_grad():
        nemo_out = wrapper(test_embs, test_lengths).numpy()
    
    # Run CoreML
    coreml_out = mlmodel.predict({
        "pre_encoder_embs": test_embs.numpy(),
        "pre_encoder_lengths": np.array([seq_len], dtype=np.int32)
    })["speaker_preds"]
    
    # Compare
    cosine_sim = np.dot(nemo_out.flatten(), coreml_out.flatten()) / (
        np.linalg.norm(nemo_out) * np.linalg.norm(coreml_out)
    )
    max_abs_err = np.max(np.abs(nemo_out - coreml_out))
    
    print(f"Cosine similarity: {cosine_sim:.6f}")
    print(f"Max absolute error: {max_abs_err:.6f}")
    print(f"Output shape: {coreml_out.shape}")
    
    if cosine_sim > 0.99:
        print("\n✓ Verification PASSED")
    else:
        print("\n✗ Verification FAILED - cosine similarity too low")
    
    return cosine_sim, max_abs_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SortFormer head to CoreML")
    parser.add_argument("--model", default="model.nemo",
                        help="NeMo model path or HuggingFace name")
    parser.add_argument("--output", default="model-coreml-head.mlpackage",
                        help="Output .mlpackage path")
    parser.add_argument("--max-seq-len", type=int, default=400,
                        help="Max sequence length (default: 400 for low-latency support)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16",
                        help="Compute precision")
    parser.add_argument("--verify", action="store_true",
                        help="Verify against NeMo after export")
    
    args = parser.parse_args()
    
    mlmodel = export_head_only(
        model_path=args.model,
        output_path=args.output,
        max_seq_len=args.max_seq_len,
        precision=args.precision
    )
    
    if args.verify:
        verify_model(args.output, args.model, args.max_seq_len)
