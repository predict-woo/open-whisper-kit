# WISDOM.md — Converting Python AI Pipelines to GGML/C++

Hard-won lessons from converting NVIDIA's SortFormer (117M params, NeMo/PyTorch) into a native C++ inference pipeline using GGML. This guide is for AI agents and developers doing similar conversions.

---

## The Overall Process

The conversion follows a strict pipeline: **Understand → Extract → Convert → Implement → Verify**, one stage at a time. Never skip ahead. Never implement two stages without verifying the first.

### Phase 1: Understand the Source Model

**Before writing any C++, you must fully understand the Python model.**

1. **Load the model and dump everything**:

   - All state_dict keys with shapes and dtypes
   - The full model config (YAML, JSON, whatever)
   - A forward pass on test input to get the output shape

2. **Never trust documentation or papers for weight names**. The actual state_dict keys WILL differ from what you expect. In SortFormer:

   - Plan predicted `feed_forward1.w_1` → actual was `feed_forward1.linear1`
   - Plan predicted `self_attention.query` → actual was `first_sub_layer.query_net`
   - Plan predicted `norm_final` → actual was `norm_out`
   - Plan predicted `out.0.weight` → actual was `out.weight`
   - Plan predicted a `final_layer_norm` → it didn't exist in the state_dict at all

3. **Dump intermediate tensors at every pipeline stage** using `register_forward_hook`. This is your ground truth for the entire project. For each stage, save the output as `.npy` files. You will compare against these hundreds of times.

4. **Record the exact tensor shapes at every stage**. Build a shape map:
   ```
   mel:          (1, 128, 2016)
   pre_encoder:  (1, 63, 512)
   conformer_0:  (1, 251, 512)   ← shape CHANGED from pre_encoder!
   projection:   (1, 251, 192)
   transformer:  (1, 251, 192)
   prediction:   (1, 250, 4)     ← dropped 1 frame!
   ```
   Shape changes between stages are where bugs hide.

### Phase 2: Weight Conversion (Python → GGUF)

5. **Write your own minimal GGUF writer** instead of depending on the `gguf` Python package. It's ~200 lines of struct packing. You get full control and zero dependency issues.

6. **Keep the original tensor names**. Don't rename weights during conversion. It makes debugging much easier when the GGUF tensor name matches the PyTorch state_dict key exactly.

7. **Fuse BatchNorm during conversion, not at runtime**. The formula:

   ```
   scale = gamma / sqrt(running_var + eps)
   fused_w = orig_w * scale
   fused_b = (orig_b - running_mean) * scale + beta
   ```

   Do the fusion in F32, then convert to F16. Write a validation script that runs the original BN forward pass and compares with the fused result.

8. **Store weights as F16, but keep special tensors as F32**. Mel filterbanks, window functions, and any small lookup tables should stay F32. The precision loss from F16 on these is unnecessary.

9. **Write all model hyperparameters as GGUF metadata**. The C++ code should read dimensions, layer counts, etc. from the GGUF file, not hardcode them (though for a single model, hardcoding is acceptable for v1).

10. **GGUF dimensions are stored reversed** (innermost first). A PyTorch tensor of shape `(512, 256)` becomes GGUF dimensions `[256, 512]`. The actual memory layout is identical — it's just the dimension ordering convention that differs.

### Phase 3: C++ Implementation with GGML

11. **Set up the project skeleton first**, with stubs that return -1. Verify the build works before writing any real code. Use GGML as a git submodule pinned to a specific commit.

12. **CMake pattern** (from whisper.cpp):

    ```cmake
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    add_subdirectory(ggml)
    target_link_libraries(your_lib PUBLIC ggml Threads::Threads)
    ```

    That's it. GGML handles the rest.

13. **Implement stages sequentially, verify each one before moving on**. The order matters:
    - Mel spectrogram (pure DSP, no GGML graphs)
    - Pre-encoder (first GGML graph — learn the patterns here)
    - Encoder layers (the hard part)
    - Projection (trivial linear)
    - Decoder/transformer layers
    - Output head
    - Streaming state management (if applicable)

### Phase 4: Verification

14. **Compare with cosine similarity, not absolute error**. F16 quantization introduces absolute errors of 1-10 in early layers, but cosine similarity stays > 0.999. Use graduated thresholds:

    - First stage: cos > 0.9999
    - Mid-pipeline: cos > 0.999
    - Full pipeline: cos > 0.99

15. **When a stage doesn't match, bisect within the stage**. For the conformer, we dumped per-submodule outputs (FFN1, MHSA, Conv, FFN2) to find that only MHSA was wrong. Then compared position embeddings directly to find the interleaved vs concatenated bug.

---

## GGML-Specific Wisdom

### The Computation Graph Pattern

Every GGML computation follows this pattern:

```cpp
// 1. Create a temporary context for the graph
size_t buf_size = 512 * 1024 * 1024;  // 512 MB
std::vector<uint8_t> buf(buf_size);
struct ggml_init_params gparams = { buf_size, buf.data(), true };
struct ggml_context * ctx0 = ggml_init(gparams);

// 2. Create input tensor and copy data in
struct ggml_tensor * input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, T);
memcpy(input->data, input_data, dim * T * sizeof(float));

// 3. Build the computation graph using GGML ops
struct ggml_tensor * x = ggml_mul_mat(ctx0, weight, input);  // weight @ input^T
x = ggml_add(ctx0, x, bias);
x = ggml_relu(ctx0, x);
// ... more ops ...

// 4. Allocate and run the graph
struct ggml_cgraph * graph = ggml_new_graph_custom(ctx0, max_nodes, false);
ggml_build_forward_expand(graph, output_tensor);
ggml_graph_compute_with_ctx(ctx0, graph, n_threads);

// 5. Copy output data out
memcpy(output_data, output_tensor->data, output_size);

// 6. Free the temporary context
ggml_free(ctx0);
```

### Critical GGML Gotchas

16. **`ggml_mul_mat(A, B)` computes `B @ A^T`**, not `A @ B`. The weight matrix goes FIRST. This is the #1 source of confusion. For a linear layer `y = x @ W^T + b`:

    ```cpp
    y = ggml_mul_mat(ctx, W, x);  // W is first argument
    y = ggml_add(ctx, y, b);
    ```

17. **F16 weights must be cast to F32 before computation**:

    ```cpp
    struct ggml_tensor * w_f32 = ggml_cast(ctx, w_f16, GGML_TYPE_F32);
    ```

    This is especially critical for `ggml_conv_2d_dw_direct` which assumes F32 kernels (it casts to `float*` directly). Other ops like `ggml_mul_mat` handle F16 internally, but casting everything to F32 is safer and the performance difference is negligible on CPU.

18. **Bias broadcasting requires reshaping**. GGML's `ggml_add` broadcasts when `a->ne[i] % b->ne[i] == 0`. A 1D bias `(dim,)` won't broadcast against a 2D tensor `(dim, T)` — you need:

    ```cpp
    // For 2D: reshape bias to (dim, 1)
    bias_2d = ggml_reshape_2d(ctx, bias, dim, 1);
    result = ggml_add(ctx, x, bias_2d);

    // For 4D conv output (W, H, C, N): reshape bias to (1, 1, C, 1)
    bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, C, 1);
    result = ggml_add(ctx, conv_out, bias_4d);
    ```

19. **Graph node limits**. The default `ggml_new_graph` creates a graph with limited capacity. For large models, use:

    ```cpp
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16384, false);
    ```

    A 17-layer conformer needs ~2126 nodes. An 18-layer transformer needs ~954 nodes.

20. **Conv2D kernel layout**: `ne[0]=KW, ne[1]=KH, ne[2]=IC, ne[3]=OC`. This matches PyTorch's C-contiguous memory layout, so no transposition is needed when loading from GGUF. The stride/padding parameters: `s0/p0` = width dimension, `s1/p1` = height dimension.

21. **Tensor data layout**. GGML stores tensors with `ne[0]` as the innermost (contiguous) dimension. A tensor with `ne = [512, 250]` has 250 rows of 512 elements each. This is the same as row-major `(250, 512)` in C. When you `memcpy` data out, you get row-major order naturally.

22. **Permute for multi-head attention**. The standard pattern:
    ```cpp
    // x shape: (d_model, T) = (512, 250)
    // Reshape to (d_head, n_heads, T)
    x = ggml_reshape_3d(ctx, x, d_head, n_heads, T);
    // Permute to (d_head, T, n_heads) for per-head operations
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    // Now Q @ K^T gives (T, T, n_heads) — correct for softmax over T
    ```

### Memory Management

23. **Two-context pattern**: Use a persistent `ggml_context` for model weights (loaded once, lives forever) and a temporary `ggml_context` for each computation graph (created and freed per forward pass). The persistent context is created during `model_init()` with `no_alloc=false` so GGML allocates memory for all weight tensors.

24. **Buffer sizing**. For the temporary computation context, allocate generously. A good rule of thumb: `max_sequence_length * d_model * n_layers * 4 bytes * 10`. For SortFormer with T=250, d=512, 17 layers: ~500MB was sufficient. You can always reduce later.

---

## Python-Side Wisdom

### Setting Up the Reference Pipeline

25. **Use `uv` for Python environment management**. It's fast, reliable, and doesn't pollute the system. Create a venv, install dependencies, and always run scripts through the venv.

26. **Force CPU-only PyTorch**. On machines without GPU, NeMo/PyTorch will still try to use CUDA. Force CPU:

    ```python
    model = Model.restore_from('model.nemo', map_location='cpu')
    ```

27. **Use `soundfile` instead of `torchaudio`** for loading audio. Recent torchaudio versions require `torchcodec` which may not be installed. `soundfile` is simpler and always works:

    ```python
    import soundfile as sf
    audio, sr = sf.read(path, dtype='float32')
    ```

28. **Use `register_forward_hook` for intermediate tensor extraction**:

    ```python
    outputs = {}
    def hook(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                outputs[name] = output[0].detach().cpu().numpy()
            else:
                outputs[name] = output.detach().cpu().numpy()
        return fn

    model.encoder.layers[0].register_forward_hook(hook('conformer_0'))
    ```

29. **Beware of streaming mode**. Some models run streaming internally even when you call `model.forward()`. Check the model config for `streaming_mode: true`. You may need to set `model.streaming_mode = False` to get clean offline intermediate tensors.

### Comparison Scripts

30. **Write a generic tensor comparison script** that handles both `.npy` and raw float32 binary files. Report: max absolute error, mean absolute error, cosine similarity, and PASS/FAIL against a threshold.

31. **Use `<=` not `<` for tolerance checks**. This handles the edge case of `--tolerance 0` for self-comparison (identical files should pass).

---

## Architecture-Specific Lessons

### Position Embeddings

32. **Always verify the sinusoidal position embedding layout**. There are two common conventions:

    - **Concatenated**: `[sin(f0), sin(f1), ..., sin(f_d/2), cos(f0), cos(f1), ..., cos(f_d/2)]`
    - **Interleaved**: `[sin(f0), cos(f0), sin(f1), cos(f1), ...]`

    NeMo uses interleaved. Many other implementations use concatenated. Getting this wrong gives cosine similarity of ~0.2 instead of ~0.999. **Always compare the raw position embeddings against the reference before debugging attention.**

### Layer Normalization

33. **Post-LN vs Pre-LN matters enormously**:

    - Pre-LN: `x = x + sublayer(layer_norm(x))`
    - Post-LN: `x = layer_norm(x + sublayer(x))`

    Check the model config carefully. SortFormer's transformer uses post-LN (`pre_ln=false`), which is less common in modern models but still used.

### Conformer-Specific

34. **The Conformer has 5 sub-modules per layer**, not 4:

    1. FFN1 (half-step, residual scaled by 0.5)
    2. Multi-Head Self-Attention (with relative position encoding)
    3. Convolution Module (pointwise → GLU → depthwise → BN → activation → pointwise)
    4. FFN2 (half-step, residual scaled by 0.5)
    5. Final LayerNorm

35. **xscaling**: When `xscaling=true`, the input to the conformer is multiplied by `sqrt(d_model)`. This is applied ONCE at the input, not per-layer.

36. **Relative position attention** uses `pos_bias_u` and `pos_bias_v` (untied biases per layer). The attention score is: `(Q + pos_bias_u) @ K^T + (Q + pos_bias_v) @ pos^T` with a relative shift on the position term.

### Streaming

37. **Implement offline first, then add streaming**. Even if the final product is streaming-only, getting the offline pipeline correct first gives you a verified reference for every stage.

38. **Use `seq_len` (valid frames), not padded length, for streaming chunk boundaries**. Mel spectrograms are often padded to a multiple of some number. Using the padded length gives wrong frame counts.

39. **The speaker cache (AOSC) is the hardest part of streaming**. It involves:
    - Tracking per-frame speaker predictions
    - Computing importance scores per speaker
    - Selecting top-K frames per speaker
    - Filling gaps with silence embeddings
    - Handling the cold-start case (empty cache on first chunk)

---

## Process Wisdom

### Debugging Strategy

40. **When something doesn't match, always bisect**. Don't stare at the full pipeline output. Dump intermediates at finer and finer granularity until you find the exact operation that diverges.

41. **Isolate F16 quantization error from implementation error**. Run the Python model with F16 weights (cast to F16 then back to F32) and compare against the F32 reference. This tells you the theoretical best your C++ implementation can achieve. If the Python-F16 cosine is 0.99999 but your C++ cosine is 0.95, you have a bug. If both are 0.999, you're fine.

42. **Create a per-submodule dump script** for the most complex layers. For the conformer, we created `dump_conformer_submodules.py` that dumps FFN1, MHSA, Conv, and FFN2 outputs separately. This immediately revealed that only MHSA was wrong.

### Project Structure

43. **Separate the library from the CLI**. Build a static library (`libfoo.a`) with a clean C API, and a separate CLI binary that links against it. This makes the library reusable and the CLI a thin wrapper.

44. **Add `--dump-*` flags from the start**. Every pipeline stage should have a dump flag that writes raw float32 binary. This is your primary debugging tool throughout the entire project.

45. **Keep all comparison scripts in `scripts/`**. You'll run them hundreds of times. Make them accept flexible input (both `.npy` and raw binary, configurable shapes and tolerances).

### What Takes the Most Time

In order of difficulty and time spent:

1. **Conformer encoder** (~40% of total effort) — Relative position attention, GLU gating, depthwise conv, multiple sub-modules per layer
2. **Streaming state management** (~20%) — FIFO, speaker cache, AOSC compression, cold start
3. **Pre-encoder** (~15%) — Conv2D with GGML, bias broadcasting, flatten/permute for linear
4. **Debugging position embeddings** (~10%) — Interleaved vs concatenated, comparing raw embeddings
5. **Everything else** (~15%) — Mel, transformer, prediction head, post-processing, CLI

### Common Failure Modes

- **Wrong weight names**: Always dump and verify actual state_dict keys
- **Wrong dimension ordering**: GGML vs PyTorch conventions differ
- **Wrong position embedding layout**: Interleaved vs concatenated
- **Wrong layer norm placement**: Pre-LN vs post-LN
- **F16 precision loss mistaken for bugs**: Always establish the F16 baseline first
- **Padded vs unpadded lengths**: Especially in streaming chunk boundaries
- **Missing bias terms**: Some layers have biases that aren't obvious from the architecture description
- **Broadcasting failures**: GGML requires explicit reshaping for bias addition

---

## Quick Reference: GGML Op Cheat Sheet

| PyTorch                 | GGML                                       | Notes                         |
| ----------------------- | ------------------------------------------ | ----------------------------- |
| `x @ W.T`               | `ggml_mul_mat(W, x)`                       | W is first arg                |
| `x + b`                 | `ggml_add(x, reshape(b))`                  | Reshape b for broadcasting    |
| `layer_norm(x)`         | `ggml_norm(x, eps)` then scale+shift       | Manual affine transform       |
| `relu(x)`               | `ggml_relu(x)`                             |                               |
| `silu(x)` / `swish(x)`  | `ggml_silu(x)`                             |                               |
| `sigmoid(x)`            | `ggml_sigmoid(x)`                          |                               |
| `softmax(x, dim=-1)`    | `ggml_soft_max(x)`                         | Always over dim 0 (innermost) |
| `conv2d(x, w, s, p)`    | `ggml_conv_2d(w, x, s0, s1, p0, p1, 1, 1)` | w first, then x               |
| `conv2d depthwise`      | `ggml_conv_2d_dw_direct(w, x, b, s, p, d)` | Assumes F32 kernel            |
| `x.reshape(...)`        | `ggml_reshape_Nd(x, ...)`                  |                               |
| `x.permute(...)`        | `ggml_permute(x, ...)`                     |                               |
| `x.contiguous()`        | `ggml_cont(x)`                             | After permute, before reshape |
| `torch.cat([a,b], dim)` | `ggml_concat(a, b, dim)`                   |                               |
| `F16 → F32`             | `ggml_cast(x, GGML_TYPE_F32)`              | Always do this for weights    |
