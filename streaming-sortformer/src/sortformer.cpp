#include "sortformer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef SORTFORMER_USE_COREML
#include "coreml/sortformer-coreml.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Conformer layer weights (17 layers × 34 tensors each)
// ============================================================================

static const int N_CONF_LAYERS = 17;
static const int N_HEADS       = 8;
static const int D_HEAD        = 64;
static const int CONV_KERNEL   = 9;

struct conformer_layer_weights {
    // FFN1 (Macaron half-step)
    struct ggml_tensor * norm_ff1_w;
    struct ggml_tensor * norm_ff1_b;
    struct ggml_tensor * ff1_up_w;    // linear1: (2048, 512)
    struct ggml_tensor * ff1_up_b;
    struct ggml_tensor * ff1_down_w;  // linear2: (512, 2048)
    struct ggml_tensor * ff1_down_b;

    // Self-Attention
    struct ggml_tensor * norm_sa_w;
    struct ggml_tensor * norm_sa_b;
    struct ggml_tensor * sa_q_w;
    struct ggml_tensor * sa_q_b;
    struct ggml_tensor * sa_k_w;
    struct ggml_tensor * sa_k_b;
    struct ggml_tensor * sa_v_w;
    struct ggml_tensor * sa_v_b;
    struct ggml_tensor * sa_out_w;
    struct ggml_tensor * sa_out_b;
    struct ggml_tensor * sa_pos_w;  // linear_pos.weight (no bias)
    struct ggml_tensor * pos_bias_u; // (8, 64) → GGML (64, 8)
    struct ggml_tensor * pos_bias_v;

    // Conv Module
    struct ggml_tensor * norm_conv_w;
    struct ggml_tensor * norm_conv_b;
    struct ggml_tensor * conv_pw1_w;  // pointwise_conv1: [1024, 512, 1]
    struct ggml_tensor * conv_pw1_b;
    struct ggml_tensor * conv_dw_w;   // depthwise_conv: [512, 1, 9] (BN fused)
    struct ggml_tensor * conv_dw_b;   // fused BN bias
    struct ggml_tensor * conv_pw2_w;  // pointwise_conv2: [512, 512, 1]
    struct ggml_tensor * conv_pw2_b;

    // FFN2
    struct ggml_tensor * norm_ff2_w;
    struct ggml_tensor * norm_ff2_b;
    struct ggml_tensor * ff2_up_w;
    struct ggml_tensor * ff2_up_b;
    struct ggml_tensor * ff2_down_w;
    struct ggml_tensor * ff2_down_b;

    // Final LayerNorm
    struct ggml_tensor * norm_out_w;
    struct ggml_tensor * norm_out_b;
};

// ============================================================================
// Transformer encoder layer weights (18 layers)
// ============================================================================

static const int N_TRANS_LAYERS = 18;
static const int TF_D_MODEL = 192;
static const int TF_N_HEADS = 8;
static const int TF_D_HEAD  = 24;  // 192/8
static const int TF_FF_DIM  = 768; // 4*192

struct transformer_layer_weights {
    // Self-attention (first_sub_layer)
    struct ggml_tensor * q_w, * q_b;     // query_net
    struct ggml_tensor * k_w, * k_b;     // key_net
    struct ggml_tensor * v_w, * v_b;     // value_net
    struct ggml_tensor * out_w, * out_b; // out_projection
    struct ggml_tensor * ln1_w, * ln1_b; // layer_norm_1

    // FFN (second_sub_layer)
    struct ggml_tensor * ff_up_w, * ff_up_b;     // dense_in (192→768)
    struct ggml_tensor * ff_down_w, * ff_down_b; // dense_out (768→192)
    struct ggml_tensor * ln2_w, * ln2_b;         // layer_norm_2
};

// ============================================================================
// Context
// ============================================================================

struct sortformer_context {
    sortformer_params params;

    // GGUF / ggml
    struct gguf_context * gguf_ctx;
    struct ggml_context * ggml_ctx;

    // Backend (persistent)
    ggml_backend_t backend;
    ggml_gallocr_t galloc;

    // Mel parameters (from GGUF metadata)
    int   n_mels;
    int   n_fft;
    int   hop_length;
    int   win_length;
    int   sample_rate;
    float preemph;
    float log_guard;
    int   pad_to;

    // Mel tensors (pointers into ggml_ctx memory)
    const float * mel_filterbank;
    const float * hann_window;

    // FFT cache
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    // Position embeddings cache
    std::vector<float> pos_emb_cache;
    int pos_emb_cache_n_pos;

    // Pre-encoder weights (pointers into ggml_ctx memory, F16)
    struct ggml_tensor * preenc_conv0_w;
    struct ggml_tensor * preenc_conv0_b;
    struct ggml_tensor * preenc_conv2_w;  // depthwise
    struct ggml_tensor * preenc_conv2_b;
    struct ggml_tensor * preenc_conv3_w;  // pointwise
    struct ggml_tensor * preenc_conv3_b;
    struct ggml_tensor * preenc_conv5_w;  // depthwise
    struct ggml_tensor * preenc_conv5_b;
    struct ggml_tensor * preenc_conv6_w;  // pointwise
    struct ggml_tensor * preenc_conv6_b;
    struct ggml_tensor * preenc_out_w;
    struct ggml_tensor * preenc_out_b;

    // Conformer encoder weights
    struct conformer_layer_weights conf_layers[N_CONF_LAYERS];

    // Projection layer (512 → 192)
    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    // Transformer encoder weights
    struct transformer_layer_weights trans_layers[N_TRANS_LAYERS];

    // Prediction head weights
    struct ggml_tensor * pred_hidden_w;
    struct ggml_tensor * pred_hidden_b;
    struct ggml_tensor * pred_spk_w;
    struct ggml_tensor * pred_spk_b;

    // Encoder metadata
    int d_model;
    int subsampling_factor;
    int n_conf_layers;
    int n_heads;
    int d_head;
    int conv_kernel_size;

#ifdef SORTFORMER_USE_COREML
    // CoreML context (nullptr if not using CoreML)
    struct sortformer_coreml_context * coreml_ctx;
#endif
};

// ============================================================================
// GGUF helpers
// ============================================================================

static int gguf_get_u32(const struct gguf_context * ctx, const char * key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        fprintf(stderr, "%s: key '%s' not found\n", __func__, key);
        return -1;
    }
    return (int)gguf_get_val_u32(ctx, idx);
}

static float gguf_get_f32(const struct gguf_context * ctx, const char * key) {
    int64_t idx = gguf_find_key(ctx, key);
    if (idx < 0) {
        fprintf(stderr, "%s: key '%s' not found\n", __func__, key);
        return -1.0f;
    }
    return gguf_get_val_f32(ctx, idx);
}

// ============================================================================
// FFT (Cooley-Tukey, based on whisper.cpp)
// ============================================================================

// Iterative radix-2 FFT in float precision (Cooley-Tukey, bit-reversal).
// Input: N real values in in_real[]. Output: N complex values in out[] as interleaved (re,im).
// N must be a power of 2.
static void fft_real_to_complex(const float * in_real, int N, float * out) {
    // Bit-reversal permutation
    for (int i = 0; i < N; i++) {
        int j = 0;
        for (int bit = 0; bit < 30; bit++) {
            if ((1 << bit) >= N) break;
            if (i & (1 << bit)) j |= (N >> (bit + 1));
        }
        out[2 * j + 0] = in_real[i];
        out[2 * j + 1] = 0.0f;
    }

    // Butterfly stages
    for (int stage_len = 2; stage_len <= N; stage_len *= 2) {
        float angle = -2.0f * (float)M_PI / (float)stage_len;
        float w_re = cosf(angle);
        float w_im = sinf(angle);

        for (int start = 0; start < N; start += stage_len) {
            float tw_re = 1.0f;
            float tw_im = 0.0f;
            int half = stage_len / 2;
            for (int k = 0; k < half; k++) {
                int idx_a = start + k;
                int idx_b = start + k + half;

                float a_re = out[2 * idx_a + 0];
                float a_im = out[2 * idx_a + 1];
                float b_re = out[2 * idx_b + 0];
                float b_im = out[2 * idx_b + 1];

                float tb_re = tw_re * b_re - tw_im * b_im;
                float tb_im = tw_re * b_im + tw_im * b_re;

                out[2 * idx_a + 0] = a_re + tb_re;
                out[2 * idx_a + 1] = a_im + tb_im;
                out[2 * idx_b + 0] = a_re - tb_re;
                out[2 * idx_b + 1] = a_im - tb_im;

                float new_tw_re = tw_re * w_re - tw_im * w_im;
                float new_tw_im = tw_re * w_im + tw_im * w_re;
                tw_re = new_tw_re;
                tw_im = new_tw_im;
            }
        }
    }
}

// ============================================================================
// Default params
// ============================================================================

struct sortformer_params sortformer_default_params(void) {
    struct sortformer_params params;
    params.chunk_len              = 188;
    params.right_context          = 1;
    params.fifo_len               = 0;
    params.spkcache_len           = 188;
    params.spkcache_update_period = 188;
    params.threshold              = 0.5f;
    params.median_filter          = 11;
    params.n_threads              = 4;
    params.chunk_left_context     = 1;
    return params;
}

// ============================================================================
// Init: Load GGUF model (mel-related tensors and metadata)
// ============================================================================

struct sortformer_context * sortformer_init(const char * model_path, struct sortformer_params params) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, model_path);

    // Open GGUF file with a ggml context to hold tensor data
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ggml_ctx,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: failed to open GGUF file\n", __func__);
        return nullptr;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
    const int64_t n_kv      = gguf_get_n_kv(gguf_ctx);
    fprintf(stderr, "%s: GGUF version %u, %lld tensors, %lld KV pairs\n",
            __func__, gguf_get_version(gguf_ctx),
            (long long)n_tensors, (long long)n_kv);

    // Read mel metadata
    int n_mels      = gguf_get_u32(gguf_ctx, "sortformer.mel.n_mels");
    int n_fft       = gguf_get_u32(gguf_ctx, "sortformer.mel.n_fft");
    int hop_length  = gguf_get_u32(gguf_ctx, "sortformer.mel.hop_length");
    int win_length  = gguf_get_u32(gguf_ctx, "sortformer.mel.win_length");
    int sample_rate = gguf_get_u32(gguf_ctx, "sortformer.mel.sample_rate");
    float dither    = gguf_get_f32(gguf_ctx, "sortformer.mel.dither");
    (void)dither; // ignored for now (NeMo disables dither in eval mode)

    if (n_mels < 0 || n_fft < 0 || hop_length < 0 || win_length < 0 || sample_rate < 0) {
        fprintf(stderr, "%s: failed to read mel metadata\n", __func__);
        gguf_free(gguf_ctx);
        if (ggml_ctx) ggml_free(ggml_ctx);
        return nullptr;
    }

    fprintf(stderr, "%s: mel params: n_mels=%d, n_fft=%d, hop=%d, win=%d, sr=%d\n",
            __func__, n_mels, n_fft, hop_length, win_length, sample_rate);

    // Load mel filterbank tensor: (n_mels, n_fft/2+1) = (128, 257)
    struct ggml_tensor * fb_tensor = ggml_get_tensor(ggml_ctx, "preprocessor.featurizer.fb");
    if (!fb_tensor) {
        fprintf(stderr, "%s: mel filterbank tensor not found\n", __func__);
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return nullptr;
    }
    // Verify shape: ne[0]=257 (n_fft/2+1), ne[1]=128 (n_mels)
    int n_fft_bins = n_fft / 2 + 1;
    if (fb_tensor->ne[0] != n_fft_bins || fb_tensor->ne[1] != n_mels) {
        fprintf(stderr, "%s: unexpected fb shape: [%lld, %lld], expected [%d, %d]\n",
                __func__,
                (long long)fb_tensor->ne[0], (long long)fb_tensor->ne[1],
                n_fft_bins, n_mels);
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return nullptr;
    }
    fprintf(stderr, "%s: mel filterbank: [%lld, %lld] F32\n",
            __func__, (long long)fb_tensor->ne[0], (long long)fb_tensor->ne[1]);

    // Load Hann window tensor: (win_length,) = (400,)
    struct ggml_tensor * win_tensor = ggml_get_tensor(ggml_ctx, "preprocessor.featurizer.window");
    if (!win_tensor) {
        fprintf(stderr, "%s: Hann window tensor not found\n", __func__);
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return nullptr;
    }
    if (win_tensor->ne[0] != win_length) {
        fprintf(stderr, "%s: unexpected window shape: [%lld], expected [%d]\n",
                __func__, (long long)win_tensor->ne[0], win_length);
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return nullptr;
    }
    fprintf(stderr, "%s: Hann window: [%lld] F32\n",
            __func__, (long long)win_tensor->ne[0]);

    // Build FFT sin/cos cache (size = n_fft for power-of-2 FFT)
    std::vector<float> sin_cache(n_fft);
    std::vector<float> cos_cache(n_fft);
    for (int i = 0; i < n_fft; i++) {
        double theta = (2.0 * M_PI * i) / n_fft;
        sin_cache[i] = (float)sin(theta);
        cos_cache[i] = (float)cos(theta);
    }

    // Create context
    auto * ctx = new sortformer_context();
    ctx->params      = params;
    ctx->gguf_ctx    = gguf_ctx;
    ctx->ggml_ctx    = ggml_ctx;
    ctx->backend     = nullptr;
    ctx->galloc      = nullptr;
    ctx->n_mels      = n_mels;
    ctx->n_fft       = n_fft;
    ctx->hop_length  = hop_length;
    ctx->win_length  = win_length;
    ctx->sample_rate = sample_rate;
    ctx->preemph     = 0.97f;  // NeMo default (not stored in GGUF)
    ctx->log_guard   = (float)(1.0 / (1 << 24)); // 2^-24
    ctx->pad_to      = 16;

    ctx->mel_filterbank = (const float *)fb_tensor->data;
    ctx->hann_window    = (const float *)win_tensor->data;
    ctx->sin_vals       = std::move(sin_cache);
    ctx->cos_vals       = std::move(cos_cache);
    ctx->pos_emb_cache_n_pos = 0;

    // Load encoder metadata
    ctx->d_model = gguf_get_u32(gguf_ctx, "sortformer.encoder.d_model");
    ctx->subsampling_factor = gguf_get_u32(gguf_ctx, "sortformer.encoder.subsampling_factor");
    if (ctx->d_model < 0 || ctx->subsampling_factor < 0) {
        fprintf(stderr, "%s: failed to read encoder metadata\n", __func__);
        delete ctx;
        return nullptr;
    }
    ctx->n_conf_layers   = N_CONF_LAYERS;
    ctx->n_heads         = N_HEADS;
    ctx->d_head          = D_HEAD;
    ctx->conv_kernel_size = CONV_KERNEL;

    fprintf(stderr, "%s: encoder: d_model=%d, subsampling=%d, n_layers=%d, n_heads=%d\n",
            __func__, ctx->d_model, ctx->subsampling_factor,
            ctx->n_conf_layers, ctx->n_heads);

    auto load_tensor = [&](const char * name) -> struct ggml_tensor * {
        struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
        if (!t) {
            fprintf(stderr, "%s: tensor '%s' not found\n", __func__, name);
        }
        return t;
    };

    // Load pre-encoder weights
    ctx->preenc_conv0_w = load_tensor("encoder.pre_encode.conv.0.weight");
    ctx->preenc_conv0_b = load_tensor("encoder.pre_encode.conv.0.bias");
    ctx->preenc_conv2_w = load_tensor("encoder.pre_encode.conv.2.weight");
    ctx->preenc_conv2_b = load_tensor("encoder.pre_encode.conv.2.bias");
    ctx->preenc_conv3_w = load_tensor("encoder.pre_encode.conv.3.weight");
    ctx->preenc_conv3_b = load_tensor("encoder.pre_encode.conv.3.bias");
    ctx->preenc_conv5_w = load_tensor("encoder.pre_encode.conv.5.weight");
    ctx->preenc_conv5_b = load_tensor("encoder.pre_encode.conv.5.bias");
    ctx->preenc_conv6_w = load_tensor("encoder.pre_encode.conv.6.weight");
    ctx->preenc_conv6_b = load_tensor("encoder.pre_encode.conv.6.bias");
    ctx->preenc_out_w   = load_tensor("encoder.pre_encode.out.weight");
    ctx->preenc_out_b   = load_tensor("encoder.pre_encode.out.bias");

    if (!ctx->preenc_conv0_w || !ctx->preenc_conv0_b ||
        !ctx->preenc_conv2_w || !ctx->preenc_conv2_b ||
        !ctx->preenc_conv3_w || !ctx->preenc_conv3_b ||
        !ctx->preenc_conv5_w || !ctx->preenc_conv5_b ||
        !ctx->preenc_conv6_w || !ctx->preenc_conv6_b ||
        !ctx->preenc_out_w   || !ctx->preenc_out_b) {
        fprintf(stderr, "%s: failed to load pre-encoder weights\n", __func__);
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "%s: pre-encoder weights loaded\n", __func__);

    // Load conformer layer weights
    char name_buf[256];
    bool conf_ok = true;
    for (int i = 0; i < N_CONF_LAYERS; i++) {
        auto & L = ctx->conf_layers[i];

        auto lt = [&](const char * fmt) -> struct ggml_tensor * {
            snprintf(name_buf, sizeof(name_buf), fmt, i);
            return load_tensor(name_buf);
        };

        L.norm_ff1_w   = lt("encoder.layers.%d.norm_feed_forward1.weight");
        L.norm_ff1_b   = lt("encoder.layers.%d.norm_feed_forward1.bias");
        L.ff1_up_w     = lt("encoder.layers.%d.feed_forward1.linear1.weight");
        L.ff1_up_b     = lt("encoder.layers.%d.feed_forward1.linear1.bias");
        L.ff1_down_w   = lt("encoder.layers.%d.feed_forward1.linear2.weight");
        L.ff1_down_b   = lt("encoder.layers.%d.feed_forward1.linear2.bias");

        L.norm_sa_w    = lt("encoder.layers.%d.norm_self_att.weight");
        L.norm_sa_b    = lt("encoder.layers.%d.norm_self_att.bias");
        L.sa_q_w       = lt("encoder.layers.%d.self_attn.linear_q.weight");
        L.sa_q_b       = lt("encoder.layers.%d.self_attn.linear_q.bias");
        L.sa_k_w       = lt("encoder.layers.%d.self_attn.linear_k.weight");
        L.sa_k_b       = lt("encoder.layers.%d.self_attn.linear_k.bias");
        L.sa_v_w       = lt("encoder.layers.%d.self_attn.linear_v.weight");
        L.sa_v_b       = lt("encoder.layers.%d.self_attn.linear_v.bias");
        L.sa_out_w     = lt("encoder.layers.%d.self_attn.linear_out.weight");
        L.sa_out_b     = lt("encoder.layers.%d.self_attn.linear_out.bias");
        L.sa_pos_w     = lt("encoder.layers.%d.self_attn.linear_pos.weight");
        L.pos_bias_u   = lt("encoder.layers.%d.self_attn.pos_bias_u");
        L.pos_bias_v   = lt("encoder.layers.%d.self_attn.pos_bias_v");

        L.norm_conv_w  = lt("encoder.layers.%d.norm_conv.weight");
        L.norm_conv_b  = lt("encoder.layers.%d.norm_conv.bias");
        L.conv_pw1_w   = lt("encoder.layers.%d.conv.pointwise_conv1.weight");
        L.conv_pw1_b   = lt("encoder.layers.%d.conv.pointwise_conv1.bias");
        L.conv_dw_w    = lt("encoder.layers.%d.conv.depthwise_conv.weight");
        L.conv_dw_b    = lt("encoder.layers.%d.conv.depthwise_conv.bias");
        L.conv_pw2_w   = lt("encoder.layers.%d.conv.pointwise_conv2.weight");
        L.conv_pw2_b   = lt("encoder.layers.%d.conv.pointwise_conv2.bias");

        L.norm_ff2_w   = lt("encoder.layers.%d.norm_feed_forward2.weight");
        L.norm_ff2_b   = lt("encoder.layers.%d.norm_feed_forward2.bias");
        L.ff2_up_w     = lt("encoder.layers.%d.feed_forward2.linear1.weight");
        L.ff2_up_b     = lt("encoder.layers.%d.feed_forward2.linear1.bias");
        L.ff2_down_w   = lt("encoder.layers.%d.feed_forward2.linear2.weight");
        L.ff2_down_b   = lt("encoder.layers.%d.feed_forward2.linear2.bias");

        L.norm_out_w   = lt("encoder.layers.%d.norm_out.weight");
        L.norm_out_b   = lt("encoder.layers.%d.norm_out.bias");

        // Verify all loaded
        struct ggml_tensor ** ptrs = (struct ggml_tensor **)&L;
        for (int j = 0; j < (int)(sizeof(conformer_layer_weights) / sizeof(struct ggml_tensor *)); j++) {
            if (!ptrs[j]) { conf_ok = false; break; }
        }
        if (!conf_ok) break;
    }

    if (!conf_ok) {
        fprintf(stderr, "%s: failed to load conformer weights\n", __func__);
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "%s: conformer weights loaded (%d layers)\n", __func__, N_CONF_LAYERS);

    // Load projection layer weights
    ctx->proj_w = load_tensor("sortformer_modules.encoder_proj.weight");
    ctx->proj_b = load_tensor("sortformer_modules.encoder_proj.bias");
    if (!ctx->proj_w || !ctx->proj_b) {
        fprintf(stderr, "%s: failed to load projection weights\n", __func__);
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "%s: projection weights loaded\n", __func__);

    // Load transformer encoder weights
    bool trans_ok = true;
    for (int i = 0; i < N_TRANS_LAYERS; i++) {
        auto & TL = ctx->trans_layers[i];

        auto tlt = [&](const char * fmt) -> struct ggml_tensor * {
            snprintf(name_buf, sizeof(name_buf), fmt, i);
            return load_tensor(name_buf);
        };

        TL.q_w   = tlt("transformer_encoder.layers.%d.first_sub_layer.query_net.weight");
        TL.q_b   = tlt("transformer_encoder.layers.%d.first_sub_layer.query_net.bias");
        TL.k_w   = tlt("transformer_encoder.layers.%d.first_sub_layer.key_net.weight");
        TL.k_b   = tlt("transformer_encoder.layers.%d.first_sub_layer.key_net.bias");
        TL.v_w   = tlt("transformer_encoder.layers.%d.first_sub_layer.value_net.weight");
        TL.v_b   = tlt("transformer_encoder.layers.%d.first_sub_layer.value_net.bias");
        TL.out_w = tlt("transformer_encoder.layers.%d.first_sub_layer.out_projection.weight");
        TL.out_b = tlt("transformer_encoder.layers.%d.first_sub_layer.out_projection.bias");
        TL.ln1_w = tlt("transformer_encoder.layers.%d.layer_norm_1.weight");
        TL.ln1_b = tlt("transformer_encoder.layers.%d.layer_norm_1.bias");

        TL.ff_up_w   = tlt("transformer_encoder.layers.%d.second_sub_layer.dense_in.weight");
        TL.ff_up_b   = tlt("transformer_encoder.layers.%d.second_sub_layer.dense_in.bias");
        TL.ff_down_w = tlt("transformer_encoder.layers.%d.second_sub_layer.dense_out.weight");
        TL.ff_down_b = tlt("transformer_encoder.layers.%d.second_sub_layer.dense_out.bias");
        TL.ln2_w = tlt("transformer_encoder.layers.%d.layer_norm_2.weight");
        TL.ln2_b = tlt("transformer_encoder.layers.%d.layer_norm_2.bias");

        struct ggml_tensor ** ptrs = (struct ggml_tensor **)&TL;
        for (int j = 0; j < (int)(sizeof(transformer_layer_weights) / sizeof(struct ggml_tensor *)); j++) {
            if (!ptrs[j]) { trans_ok = false; break; }
        }
        if (!trans_ok) break;
    }

    if (!trans_ok) {
        fprintf(stderr, "%s: failed to load transformer weights\n", __func__);
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "%s: transformer weights loaded (%d layers)\n", __func__, N_TRANS_LAYERS);

    // Load prediction head weights
    ctx->pred_hidden_w = load_tensor("sortformer_modules.first_hidden_to_hidden.weight");
    ctx->pred_hidden_b = load_tensor("sortformer_modules.first_hidden_to_hidden.bias");
    ctx->pred_spk_w    = load_tensor("sortformer_modules.single_hidden_to_spks.weight");
    ctx->pred_spk_b    = load_tensor("sortformer_modules.single_hidden_to_spks.bias");
    if (!ctx->pred_hidden_w || !ctx->pred_hidden_b ||
        !ctx->pred_spk_w   || !ctx->pred_spk_b) {
        fprintf(stderr, "%s: failed to load prediction head weights\n", __func__);
        delete ctx;
        return nullptr;
    }
    fprintf(stderr, "%s: prediction head weights loaded\n", __func__);

    // Initialize CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "%s: failed to init CPU backend\n", __func__);
        gguf_free(gguf_ctx);
        if (ggml_ctx) ggml_free(ggml_ctx);
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->params.n_threads);

    // Initialize graph allocator
    ctx->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ctx->galloc) {
        fprintf(stderr, "%s: failed to init graph allocator\n", __func__);
        ggml_backend_free(ctx->backend);
        gguf_free(gguf_ctx);
        if (ggml_ctx) ggml_free(ggml_ctx);
        delete ctx;
        return nullptr;
    }

#ifdef SORTFORMER_USE_COREML
    {
        std::string path_coreml = model_path;
        auto pos = path_coreml.rfind('.');
        if (pos != std::string::npos) {
            path_coreml = path_coreml.substr(0, pos);
        }
        path_coreml += "-coreml-head.mlmodelc";

        fprintf(stderr, "%s: loading CoreML model from '%s'\n", __func__, path_coreml.c_str());
        fprintf(stderr, "%s: first run on a device may take a while...\n", __func__);

        ctx->coreml_ctx = sortformer_coreml_init(path_coreml.c_str());
        if (!ctx->coreml_ctx) {
            fprintf(stderr, "%s: failed to load CoreML model (will use GGML fallback)\n", __func__);
        } else {
            fprintf(stderr, "%s: CoreML model loaded successfully\n", __func__);
        }
    }
#endif

    fprintf(stderr, "%s: model loaded successfully\n", __func__);
    return ctx;
}

// ============================================================================
// Free
// ============================================================================

void sortformer_free(struct sortformer_context * ctx) {
    if (ctx) {
#ifdef SORTFORMER_USE_COREML
        if (ctx->coreml_ctx) {
            sortformer_coreml_free(ctx->coreml_ctx);
            ctx->coreml_ctx = nullptr;
        }
#endif
        if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        if (ctx->gguf_ctx) gguf_free(ctx->gguf_ctx);
        if (ctx->ggml_ctx) ggml_free(ctx->ggml_ctx);
        delete ctx;
    }
}

// ============================================================================
// WAV loading
// ============================================================================

int sortformer_load_wav(const char * path, float ** samples_out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path);
        return -1;
    }

    // RIFF header
    char riff[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "%s: not a RIFF file\n", __func__);
        fclose(f);
        return -1;
    }

    uint32_t file_size;
    if (fread(&file_size, 4, 1, f) != 1) { fclose(f); return -1; }

    char wave[4];
    if (fread(wave, 1, 4, f) != 4 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a WAVE file\n", __func__);
        fclose(f);
        return -1;
    }

    // Read chunks
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate_wav = 0, data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                fprintf(stderr, "%s: fmt chunk too small\n", __func__);
                fclose(f);
                return -1;
            }
            if (fread(&audio_format, 2, 1, f) != 1) { fclose(f); return -1; }
            if (fread(&num_channels, 2, 1, f) != 1) { fclose(f); return -1; }
            if (fread(&sample_rate_wav, 4, 1, f) != 1) { fclose(f); return -1; }
            // skip byte_rate (4) + block_align (2)
            fseek(f, 6, SEEK_CUR);
            if (fread(&bits_per_sample, 2, 1, f) != 1) { fclose(f); return -1; }
            // skip extra fmt bytes
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
            found_fmt = true;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            // skip unknown chunk
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fprintf(stderr, "%s: missing fmt or data chunk\n", __func__);
        fclose(f);
        return -1;
    }

    // Validate format
    if (audio_format != 1) {
        fprintf(stderr, "%s: unsupported audio format %d (must be PCM=1)\n", __func__, audio_format);
        fclose(f);
        return -1;
    }
    if (num_channels != 1) {
        fprintf(stderr, "%s: unsupported channels %d (must be mono=1)\n", __func__, num_channels);
        fclose(f);
        return -1;
    }
    if (sample_rate_wav != 16000) {
        fprintf(stderr, "%s: unsupported sample rate %u (must be 16000)\n", __func__, sample_rate_wav);
        fclose(f);
        return -1;
    }
    if (bits_per_sample != 16) {
        fprintf(stderr, "%s: unsupported bits per sample %d (must be 16)\n", __func__, bits_per_sample);
        fclose(f);
        return -1;
    }

    int n_samples = (int)(data_size / 2);
    auto * raw = (int16_t *)malloc(data_size);
    if (!raw) {
        fprintf(stderr, "%s: allocation failed\n", __func__);
        fclose(f);
        return -1;
    }

    size_t read = fread(raw, 2, n_samples, f);
    fclose(f);

    if ((int)read != n_samples) {
        fprintf(stderr, "%s: expected %d samples, read %zu\n", __func__, n_samples, read);
        free(raw);
        return -1;
    }

    // Convert int16 -> float32 in [-1, 1]
    float * samples = (float *)malloc(n_samples * sizeof(float));
    if (!samples) {
        free(raw);
        return -1;
    }

    for (int i = 0; i < n_samples; i++) {
        samples[i] = (float)raw[i] / 32768.0f;
    }

    free(raw);
    *samples_out = samples;
    return n_samples;
}

// ============================================================================
// Mel spectrogram computation
// ============================================================================

int sortformer_compute_mel(
        struct sortformer_context * ctx,
        const float * samples,
        int           n_samples,
        float      ** mel_out,
        int         * n_mels_out,
        int         * seq_len_out) {

    if (!ctx || !samples || n_samples <= 0 || !mel_out || !n_mels_out) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    const int n_fft      = ctx->n_fft;       // 512
    const int hop        = ctx->hop_length;   // 160
    const int win_len    = ctx->win_length;   // 400
    const int n_mels     = ctx->n_mels;       // 128
    const int n_fft_bins = n_fft / 2 + 1;    // 257
    const float preemph  = ctx->preemph;      // 0.97
    const float log_guard = ctx->log_guard;   // 2^-24

    // 1. Apply preemphasis: y[0] = x[0], y[n] = x[n] - 0.97*x[n-1]
    std::vector<float> preemph_buf(n_samples);
    preemph_buf[0] = samples[0];
    for (int i = 1; i < n_samples; i++) {
        preemph_buf[i] = samples[i] - preemph * samples[i - 1];
    }

    // 2. Zero-pad (constant) by n_fft/2 on each side (NeMo uses pad_mode="constant")
    const int pad = n_fft / 2; // 256
    const int padded_len = n_samples + 2 * pad;
    std::vector<float> padded(padded_len, 0.0f);
    memcpy(padded.data() + pad, preemph_buf.data(), n_samples * sizeof(float));

    // 3. Create zero-padded Hann window (center 400 in 512)
    //    PyTorch center-pads: left = (n_fft - win_length) / 2 = 56
    std::vector<float> win_padded(n_fft, 0.0f);
    const int win_pad = (n_fft - win_len) / 2;
    memcpy(win_padded.data() + win_pad, ctx->hann_window, win_len * sizeof(float));

    // 4. Compute number of STFT frames
    //    n_frames = 1 + floor((padded_len - n_fft) / hop)
    const int n_stft_frames = 1 + (padded_len - n_fft) / hop;

    // 5. Compute output seq_len (NeMo's get_seq_len):
    //    seq_len = floor((n_samples + n_fft - n_fft) / hop) = floor(n_samples / hop)
    const int seq_len = n_samples / hop;

    // 6. Pad STFT frame count to multiple of pad_to
    //    NeMo pads the actual STFT output (n_stft_frames), not seq_len.
    //    Then masks frames beyond seq_len to zero.
    int n_frames_out = n_stft_frames;
    int remainder = n_frames_out % ctx->pad_to;
    if (remainder != 0) {
        n_frames_out += ctx->pad_to - remainder;
    }

    fprintf(stderr, "%s: n_samples=%d, n_stft_frames=%d, seq_len=%d, n_frames_out=%d\n",
            __func__, n_samples, n_stft_frames, seq_len, n_frames_out);

    // Allocate output: (n_mels, n_frames_out)
    float * mel = (float *)calloc(n_mels * n_frames_out, sizeof(float));
    if (!mel) {
        fprintf(stderr, "%s: allocation failed\n", __func__);
        return -1;
    }

    const int n_compute = (n_stft_frames < seq_len) ? n_stft_frames : seq_len;
    const int n_threads_raw = ctx->params.n_threads > 0 ? ctx->params.n_threads : 1;
    const int n_threads = (n_compute > 0) ? std::min(n_threads_raw, n_compute) : 0;

    auto worker = [&](int thread_id) {
        std::vector<float> fft_in(n_fft, 0.0f);
        std::vector<float> fft_out(n_fft * 2, 0.0f);

        for (int i = thread_id; i < n_compute; i += n_threads) {
            const int offset = i * hop;

            for (int j = 0; j < n_fft; j++) {
                fft_in[j] = win_padded[j] * padded[offset + j];
            }

            fft_real_to_complex(fft_in.data(), n_fft, fft_out.data());

            for (int m = 0; m < n_mels; m++) {
                float sum = 0.0f;
                const float * fb_row = ctx->mel_filterbank + m * n_fft_bins;
                for (int k = 0; k < n_fft_bins; k++) {
                    float re = fft_out[2 * k + 0];
                    float im = fft_out[2 * k + 1];
                    float pw = re * re + im * im;
                    sum += fb_row[k] * pw;
                }
                mel[m * n_frames_out + i] = logf(sum + log_guard);
            }
        }
    };

    if (n_threads > 0) {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto & t : threads) {
            t.join();
        }
    }

    // Frames from n_compute to n_frames_out are already zero (calloc)

    *mel_out    = mel;
    *n_mels_out = n_mels;
    if (seq_len_out) *seq_len_out = seq_len;
    return n_frames_out;
}

// ============================================================================
// Pre-encoder (Conv2D subsampling)
// ============================================================================

int sortformer_compute_preenc(
        struct sortformer_context * ctx,
        const float * mel_data,
        int           n_mels,
        int           n_mel_frames,
        int           seq_len,
        float      ** preenc_out,
        int         * d_model_out) {

    if (!ctx || !mel_data || !preenc_out || !d_model_out) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    // NeMo offline pipeline: trim mel to seq_len before pre-encoder
    // mel_data is (n_mels, n_mel_frames) row-major, we use only first seq_len columns
    const int T_in = seq_len;
    (void)n_mel_frames;

    // Compute output time frames: 3 stages of stride-2 conv with k=3, pad=1
    // T_out = floor((T_in - 3 + 2*1) / 2) + 1 = floor((T_in - 1) / 2) + 1
    // Applied 3 times
    int T = T_in;
    for (int i = 0; i < 3; i++) {
        T = (T - 1) / 2 + 1;
    }
    const int T_out = T;

    fprintf(stderr, "%s: mel_frames=%d, seq_len=%d, T_out=%d\n",
            __func__, n_mel_frames, seq_len, T_out);

    // Build GGML computation graph
    // Input: mel (n_mels, T_in) → reshape to (freq=n_mels, time=T_in, channels=1, batch=1)
    // GGML Conv2D expects: data [W, H, C, N], kernel [KW, KH, IC, OC]
    // NeMo Conv2D input: (batch, 1, time, freq) → GGML: ne[0]=freq, ne[1]=time, ne[2]=1, ne[3]=1

    const size_t n_tensors = 1024;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(4096, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    // Create input tensor: (freq, time, 1, 1) = (n_mels, T_in, 1, 1)
    // NeMo: input to conv is (batch, 1, time, freq)
    // GGML: ne[0]=freq, ne[1]=time, ne[2]=channels=1, ne[3]=batch=1
    struct ggml_tensor * inp = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_mels, T_in, 1, 1);
    ggml_set_name(inp, "mel_input");
    ggml_set_input(inp);

    // Copy mel data into input tensor (transpose from row-major (n_mels, T) to col-major)
    // mel_data[m * n_mel_frames + t] → inp data at [freq=m, time=t]
    // GGML stores ne[0] contiguously, so inp[t * n_mels + m] = mel_data[m * n_mel_frames + t]
    std::vector<float> inp_data((size_t)T_in * n_mels);
    for (int t = 0; t < T_in; t++) {
        for (int m = 0; m < n_mels; m++) {
            inp_data[t * n_mels + m] = mel_data[m * n_mel_frames + t];
        }
    }

    // Cast F16 bias to F32 and reshape to (1, 1, OC, 1) for broadcasting with conv2d output (OW, OH, OC, N)
    auto reshape_conv_bias = [&](struct ggml_tensor * bias) -> struct ggml_tensor * {
        struct ggml_tensor * b = ggml_cast(ctx0, bias, GGML_TYPE_F32);
        return ggml_reshape_4d(ctx0, b, 1, 1, bias->ne[0], 1);
    };

    // Cast all F16 conv weights to F32 to avoid precision loss in im2col
    // (ggml_conv_2d/ggml_conv_2d_direct convert input to kernel type in im2col)
    struct ggml_tensor * conv0_w = ggml_cast(ctx0, ctx->preenc_conv0_w, GGML_TYPE_F32);
    struct ggml_tensor * conv2_w = ggml_cast(ctx0, ctx->preenc_conv2_w, GGML_TYPE_F32);
    struct ggml_tensor * conv3_w = ggml_cast(ctx0, ctx->preenc_conv3_w, GGML_TYPE_F32);
    struct ggml_tensor * conv5_w = ggml_cast(ctx0, ctx->preenc_conv5_w, GGML_TYPE_F32);
    struct ggml_tensor * conv6_w = ggml_cast(ctx0, ctx->preenc_conv6_w, GGML_TYPE_F32);
    struct ggml_tensor * out_w   = ctx->preenc_out_w;

    // Stage 1: Conv2D(1→256, k=3, s=2, pad=1) + ReLU
    struct ggml_tensor * cur = ggml_conv_2d(ctx0, conv0_w, inp, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, reshape_conv_bias(ctx->preenc_conv0_b));
    ggml_set_name(cur, "conv0_out");
    cur = ggml_relu(ctx0, cur);

    // Stage 2: DW-Conv2D(256, k=3, s=2, pad=1) + PW-Conv2D(256→256, k=1) + ReLU
    cur = ggml_conv_2d_dw_direct(ctx0, conv2_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, reshape_conv_bias(ctx->preenc_conv2_b));
    cur = ggml_conv_2d_direct(ctx0, conv3_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, reshape_conv_bias(ctx->preenc_conv3_b));
    cur = ggml_relu(ctx0, cur);

    // Stage 3: DW-Conv2D(256, k=3, s=2, pad=1) + PW-Conv2D(256→256, k=1) + ReLU
    cur = ggml_conv_2d_dw_direct(ctx0, conv5_w, cur, 2, 2, 1, 1, 1, 1);
    cur = ggml_add(ctx0, cur, reshape_conv_bias(ctx->preenc_conv5_b));
    cur = ggml_conv_2d_direct(ctx0, conv6_w, cur, 1, 1, 0, 0, 1, 1);
    cur = ggml_add(ctx0, cur, reshape_conv_bias(ctx->preenc_conv6_b));
    cur = ggml_relu(ctx0, cur);

    // Flatten: GGML (freq, time, channels, batch) → (freq*channels, time) for mul_mat
    // permute(0,2,1,3): (freq, channels, time, batch) → cont → reshape
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);

    cur = ggml_mul_mat(ctx0, out_w, cur);
    cur = ggml_add(ctx0, cur, ggml_cast(ctx0, ctx->preenc_out_b, GGML_TYPE_F32));

    ggml_set_name(cur, "preenc_output");
    ggml_set_output(cur);

    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, cur);
    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, inp_data.data(), 0, ggml_nbytes(inp));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];

    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    // Copy output: GGML (d_model, T_out) → row-major (T_out, d_model)
    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *preenc_out = output;
    *d_model_out = out_d;
    return out_t;
}

// ============================================================================
// Conformer encoder
// ============================================================================

static void generate_sinusoidal_pos_emb(float * out, int n_pos, int d_model) {
    // NeMo interleaved layout: pe[:, 0::2] = sin, pe[:, 1::2] = cos
    // positions = [T-1, T-2, ..., 0, -1, ..., -(T-1)]
    // div_term[j] = exp(-j * log(10000) / d_model) for j=0,2,4,...,d_model-2
    const int half_d = d_model / 2;
    const int T = (n_pos + 1) / 2;

    for (int p = 0; p < n_pos; p++) {
        float pos = (float)(T - 1 - p);
        for (int j = 0; j < half_d; j++) {
            float freq = 1.0f / powf(10000.0f, (2.0f * j) / (float)d_model);
            float angle = pos * freq;
            out[p * d_model + 2 * j]     = sinf(angle);
            out[p * d_model + 2 * j + 1] = cosf(angle);
        }
    }
}

int sortformer_compute_conformer(
        struct sortformer_context * ctx,
        const float * preenc_data,
        int           T,
        int           d_model,
        int           target_layer,
        float      ** conf_out) {

    if (!ctx || !preenc_data || !conf_out || T <= 0 || d_model != ctx->d_model) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    if (target_layer < 0 || target_layer >= ctx->n_conf_layers) {
        fprintf(stderr, "%s: target_layer %d out of range [0, %d)\n",
                __func__, target_layer, ctx->n_conf_layers);
        return -1;
    }

    const int n_heads = ctx->n_heads;
    const int d_head  = ctx->d_head;
    const int kernel_size = ctx->conv_kernel_size;
    const int n_pos = 2 * T - 1;
    const float xscale = sqrtf((float)d_model);

    fprintf(stderr, "%s: T=%d, d_model=%d, target_layer=%d, xscale=%.3f\n",
            __func__, T, d_model, target_layer, xscale);

    const size_t n_tensors = 8192;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(16384, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    // Input: (d_model, T) in GGML convention
    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T);
    ggml_set_name(inp, "conf_input");
    ggml_set_input(inp);
    std::vector<float> inp_data((size_t)T * d_model);
    {
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < d_model; d++) {
                inp_data[t * d_model + d] = preenc_data[t * d_model + d];
            }
        }
    }

    // Position embeddings: (d_model, n_pos) in GGML
    struct ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, n_pos);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);
    std::vector<float> pos_emb_data((size_t)n_pos * d_model);
    {
        if (ctx->pos_emb_cache_n_pos != n_pos) {
            ctx->pos_emb_cache.resize(n_pos * d_model);
            generate_sinusoidal_pos_emb(ctx->pos_emb_cache.data(), n_pos, d_model);
            ctx->pos_emb_cache_n_pos = n_pos;
            fprintf(stderr, "%s: computed position embeddings for n_pos=%d\n", __func__, n_pos);
        } else {
            fprintf(stderr, "%s: using cached position embeddings (n_pos=%d)\n", __func__, n_pos);
        }
        std::copy(ctx->pos_emb_cache.begin(), ctx->pos_emb_cache.end(), pos_emb_data.begin());
    }

    // xscaling: multiply input by sqrt(d_model) before first layer
    struct ggml_tensor * cur = ggml_scale(ctx0, inp, xscale);

    // Helper: LayerNorm
    auto layer_norm = [&](struct ggml_tensor * x,
                          struct ggml_tensor * w,
                          struct ggml_tensor * b) -> struct ggml_tensor * {
        struct ggml_tensor * n = ggml_norm(ctx0, x, 1e-5f);
        struct ggml_tensor * wf = (w->type != GGML_TYPE_F32) ? ggml_cast(ctx0, w, GGML_TYPE_F32) : w;
        struct ggml_tensor * bf = (b->type != GGML_TYPE_F32) ? ggml_cast(ctx0, b, GGML_TYPE_F32) : b;
        return ggml_add(ctx0, ggml_mul(ctx0, n, wf), bf);
    };

    auto f32 = [&](struct ggml_tensor * t) -> struct ggml_tensor * {
        return (t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    for (int il = 0; il <= target_layer; il++) {
        const auto & L = ctx->conf_layers[il];
        struct ggml_tensor * residual = cur;

        // ===== FFN1 (Macaron half-step) =====
        {
            struct ggml_tensor * x = layer_norm(cur, L.norm_ff1_w, L.norm_ff1_b);
            x = ggml_mul_mat(ctx0, L.ff1_up_w, x);
            x = ggml_add(ctx0, x, f32(L.ff1_up_b));
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, L.ff1_down_w, x);
            x = ggml_add(ctx0, x, f32(L.ff1_down_b));
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f));
        }

        // ===== Multi-Head Self-Attention with Relative Position =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_sa_w, L.norm_sa_b);

            struct ggml_tensor * Qcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_q_w, x), f32(L.sa_q_b));
            struct ggml_tensor * Kcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_k_w, x), f32(L.sa_k_b));
            struct ggml_tensor * Vcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_v_w, x), f32(L.sa_v_b));

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_heads, T);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_heads, T);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_heads, T);

            struct ggml_tensor * Q_u = ggml_add(ctx0, Qcur, f32(L.pos_bias_u));
            struct ggml_tensor * Q_v = ggml_add(ctx0, Qcur, f32(L.pos_bias_v));

            Q_u  = ggml_permute(ctx0, Q_u,  0, 2, 1, 3);
            Q_v  = ggml_permute(ctx0, Q_v,  0, 2, 1, 3);
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            struct ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Q_u, Kcur);
            matrix_ac = ggml_cont(ctx0, ggml_permute(ctx0, matrix_ac, 1, 0, 2, 3));

            struct ggml_tensor * p = ggml_mul_mat(ctx0, L.sa_pos_w, pos_emb);
            p = ggml_reshape_3d(ctx0, p, d_head, n_heads, n_pos);
            p = ggml_permute(ctx0, p, 0, 2, 1, 3);

            struct ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, Q_v, p);
            matrix_bd = ggml_cont(ctx0, ggml_permute(ctx0, matrix_bd, 1, 0, 2, 3));

            // Relative shift (Transformer-XL style)
            {
                const int64_t pos_len = matrix_bd->ne[0];
                const int64_t q_len   = matrix_bd->ne[1];
                const int64_t h       = matrix_bd->ne[2];
                matrix_bd = ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_roll(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                    q_len, pos_len, h,
                    matrix_bd->nb[1], matrix_bd->nb[2],
                    matrix_bd->nb[0] * q_len);
                matrix_bd = ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
            }

            matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                matrix_ac->ne[0], matrix_bd->ne[1], matrix_bd->ne[2],
                matrix_bd->nb[1], matrix_bd->nb[2], 0);

            struct ggml_tensor * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
            scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)d_head));

            struct ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            struct ggml_tensor * attn_out = ggml_mul_mat(ctx0, attn, Vcur);
            attn_out = ggml_permute(ctx0, attn_out, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, d_head * n_heads, T);

            attn_out = ggml_mul_mat(ctx0, L.sa_out_w, attn_out);
            attn_out = ggml_add(ctx0, attn_out, f32(L.sa_out_b));

            residual = ggml_add(ctx0, residual, attn_out);
        }

        // ===== Conv Module =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_conv_w, L.norm_conv_b);

            struct ggml_tensor * pw1_w = L.conv_pw1_w;
            pw1_w = ggml_reshape_2d(ctx0, pw1_w, pw1_w->ne[0] * pw1_w->ne[1], pw1_w->ne[2]);
            x = ggml_mul_mat(ctx0, pw1_w, x);
            x = ggml_add(ctx0, x, f32(L.conv_pw1_b));

            {
                int64_t half = x->ne[0] / 2;
                struct ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_view_2d(ctx0, x, half, x->ne[1], x->nb[1], half * x->nb[0]));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, half, x->ne[1], x->nb[1], 0), gate);
            }

            x = ggml_cont(ctx0, ggml_transpose(ctx0, x));

            int pad = (kernel_size - 1) / 2;
            x = ggml_pad(ctx0, x, pad, 0, 0, 0);
            x = ggml_roll(ctx0, x, pad, 0, 0, 0);
            x = ggml_pad(ctx0, x, pad, 0, 0, 0);

            struct ggml_tensor * dw_w = f32(L.conv_dw_w);
            dw_w = ggml_reshape_2d(ctx0, dw_w, dw_w->ne[0], dw_w->ne[2]);
            x = ggml_ssm_conv(ctx0, x, dw_w);
            x = ggml_add(ctx0, x, f32(L.conv_dw_b));

            x = ggml_silu(ctx0, x);

            struct ggml_tensor * pw2_w = L.conv_pw2_w;
            pw2_w = ggml_reshape_2d(ctx0, pw2_w, pw2_w->ne[0] * pw2_w->ne[1], pw2_w->ne[2]);
            x = ggml_mul_mat(ctx0, pw2_w, x);
            x = ggml_add(ctx0, x, f32(L.conv_pw2_b));

            residual = ggml_add(ctx0, residual, x);
        }

        // ===== FFN2 (Macaron half-step) =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_ff2_w, L.norm_ff2_b);
            x = ggml_mul_mat(ctx0, L.ff2_up_w, x);
            x = ggml_add(ctx0, x, f32(L.ff2_up_b));
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, L.ff2_down_w, x);
            x = ggml_add(ctx0, x, f32(L.ff2_down_b));
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f));
        }

        // ===== Final LayerNorm =====
        cur = layer_norm(residual, L.norm_out_w, L.norm_out_b);
    }

    ggml_set_name(cur, "conformer_output");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, cur);

    fprintf(stderr, "%s: graph nodes = %d\n", __func__, ggml_graph_n_nodes(gf));

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, inp_data.data(), 0, ggml_nbytes(inp));
    ggml_backend_tensor_set(pos_emb, pos_emb_data.data(), 0, ggml_nbytes(pos_emb));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];
    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *conf_out = output;
    return out_t;
}

// ============================================================================
// Projection layer (512 → 192)
// ============================================================================

int sortformer_compute_projection(
        struct sortformer_context * ctx,
        const float * conf_data,
        int           T,
        int           d_model_in,
        float      ** proj_out,
        int         * d_model_out_ptr) {

    if (!ctx || !conf_data || !proj_out || !d_model_out_ptr || T <= 0) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    const int d_out = TF_D_MODEL;

    fprintf(stderr, "%s: T=%d, d_in=%d, d_out=%d\n", __func__, T, d_model_in, d_out);

    const size_t n_tensors = 512;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(4096, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    auto f32 = [&](struct ggml_tensor * t) -> struct ggml_tensor * {
        return (t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model_in, T);
    ggml_set_name(inp, "proj_input");
    ggml_set_input(inp);

    struct ggml_tensor * cur = ggml_mul_mat(ctx0, ctx->proj_w, inp);
    cur = ggml_add(ctx0, cur, f32(ctx->proj_b));
    ggml_set_name(cur, "proj_output");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, cur);
    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, conf_data, 0, ggml_nbytes(inp));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];
    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *proj_out = output;
    *d_model_out_ptr = out_d;
    return out_t;
}

// ============================================================================
// Transformer encoder (18 layers, post-LN, standard attention)
// ============================================================================

int sortformer_compute_transformer(
        struct sortformer_context * ctx,
        const float * proj_data,
        int           T,
        int           d_model,
        int           target_layer,
        float      ** trans_out) {

    if (!ctx || !proj_data || !trans_out || T <= 0 || d_model != TF_D_MODEL) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    if (target_layer < 0 || target_layer >= N_TRANS_LAYERS) {
        fprintf(stderr, "%s: target_layer %d out of range [0, %d)\n",
                __func__, target_layer, N_TRANS_LAYERS);
        return -1;
    }

    const int n_heads = TF_N_HEADS;
    const int d_head  = TF_D_HEAD;

    fprintf(stderr, "%s: T=%d, d_model=%d, target_layer=%d, n_heads=%d, d_head=%d\n",
            __func__, T, d_model, target_layer, n_heads, d_head);

    const size_t n_tensors = 4096;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(16384, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    auto f32 = [&](struct ggml_tensor * t) -> struct ggml_tensor * {
        return (t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    auto layer_norm = [&](struct ggml_tensor * x,
                          struct ggml_tensor * w,
                          struct ggml_tensor * b) -> struct ggml_tensor * {
        x = ggml_norm(ctx0, x, 1e-5f);
        x = ggml_mul(ctx0, x, f32(w));
        x = ggml_add(ctx0, x, f32(b));
        return x;
    };

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T);
    ggml_set_name(inp, "trans_input");
    ggml_set_input(inp);

    struct ggml_tensor * cur = inp;

    for (int il = 0; il <= target_layer; il++) {
        const auto & TL = ctx->trans_layers[il];

        // Self-attention sublayer
        struct ggml_tensor * attn_out;
        {
            struct ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.q_w, cur), f32(TL.q_b));
            struct ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.k_w, cur), f32(TL.k_b));
            struct ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.v_w, cur), f32(TL.v_b));

            // Reshape to (d_head, n_heads, T) and permute to (d_head, T, n_heads)
            Q = ggml_reshape_3d(ctx0, Q, d_head, n_heads, T);
            K = ggml_reshape_3d(ctx0, K, d_head, n_heads, T);
            V = ggml_reshape_3d(ctx0, V, d_head, n_heads, T);

            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3); // (d_head, T, n_heads)
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3)); // (d_head, T, n_heads)
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // (T, d_head, n_heads)

            // scores = Q @ K^T / sqrt(d_head) → (T, T, n_heads)
            struct ggml_tensor * scores = ggml_mul_mat(ctx0, Q, K);
            scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)d_head));

            struct ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            // attn @ V → (d_head, T, n_heads) via permute
            attn_out = ggml_mul_mat(ctx0, attn, V);
            attn_out = ggml_permute(ctx0, attn_out, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, d_head * n_heads, T);

            attn_out = ggml_mul_mat(ctx0, TL.out_w, attn_out);
            attn_out = ggml_add(ctx0, attn_out, f32(TL.out_b));
        }

        // POST-LN: x = layer_norm(x + attn_out)
        cur = layer_norm(ggml_add(ctx0, cur, attn_out), TL.ln1_w, TL.ln1_b);

        // FFN sublayer
        struct ggml_tensor * ff_out;
        {
            ff_out = ggml_mul_mat(ctx0, TL.ff_up_w, cur);
            ff_out = ggml_add(ctx0, ff_out, f32(TL.ff_up_b));
            ff_out = ggml_relu(ctx0, ff_out);
            ff_out = ggml_mul_mat(ctx0, TL.ff_down_w, ff_out);
            ff_out = ggml_add(ctx0, ff_out, f32(TL.ff_down_b));
        }

        // POST-LN: x = layer_norm(x + ff_out)
        cur = layer_norm(ggml_add(ctx0, cur, ff_out), TL.ln2_w, TL.ln2_b);
    }

    ggml_set_name(cur, "transformer_output");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 16384, false);
    ggml_build_forward_expand(gf, cur);

    fprintf(stderr, "%s: graph nodes = %d\n", __func__, ggml_graph_n_nodes(gf));

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, proj_data, 0, ggml_nbytes(inp));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];
    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *trans_out = output;
    return out_t;
}

// ============================================================================
// Prediction head (ReLU → Linear → ReLU → Linear → Sigmoid)
// ============================================================================

int sortformer_compute_prediction(
        struct sortformer_context * ctx,
        const float * trans_data,
        int           T,
        int           d_model,
        float      ** pred_out) {

    if (!ctx || !trans_data || !pred_out || T <= 0 || d_model != TF_D_MODEL) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    fprintf(stderr, "%s: T=%d, d_model=%d\n", __func__, T, d_model);

    const size_t n_tensors = 512;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(4096, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    auto f32 = [&](struct ggml_tensor * t) -> struct ggml_tensor * {
        return (t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T);
    ggml_set_name(inp, "pred_input");
    ggml_set_input(inp);

    // ReLU on transformer output
    struct ggml_tensor * cur = ggml_relu(ctx0, inp);

    // Linear(192 → 192)
    cur = ggml_mul_mat(ctx0, ctx->pred_hidden_w, cur);
    cur = ggml_add(ctx0, cur, f32(ctx->pred_hidden_b));

    // ReLU
    cur = ggml_relu(ctx0, cur);

    // Linear(192 → 4)
    cur = ggml_mul_mat(ctx0, ctx->pred_spk_w, cur);
    cur = ggml_add(ctx0, cur, f32(ctx->pred_spk_b));

    // Sigmoid
    cur = ggml_sigmoid(ctx0, cur);

    ggml_set_name(cur, "pred_output");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 4096, false);
    ggml_build_forward_expand(gf, cur);
    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, trans_data, 0, ggml_nbytes(inp));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];
    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *pred_out = output;
    return out_t;
}

// ============================================================================
// Streaming inference helpers
// ============================================================================

static const int STREAM_N_SPK = 4;

// Streaming configuration (model defaults)
struct stream_config {
    int chunk_len;              // 188
    int fifo_len;               // 0
    int spkcache_len;           // 188
    int spkcache_update_period; // 188
    int chunk_left_context;     // 1
    int chunk_right_context;    // 1
    int spkcache_sil_frames_per_spk; // 3
    float sil_threshold;        // 0.2
    float pred_score_threshold; // 0.25
    float scores_boost_latest;  // 0.05
    float strong_boost_rate;    // 0.75
    float weak_boost_rate;      // 1.5
    float min_pos_scores_rate;  // 0.5
    int max_index;              // 99999
};

static stream_config default_stream_config() {
    stream_config c;
    c.chunk_len              = 188;
    c.fifo_len               = 0;
    c.spkcache_len           = 188;
    c.spkcache_update_period = 188;
    c.chunk_left_context     = 1;
    c.chunk_right_context    = 1;
    c.spkcache_sil_frames_per_spk = 3;
    c.sil_threshold          = 0.2f;
    c.pred_score_threshold   = 0.25f;
    c.scores_boost_latest    = 0.05f;
    c.strong_boost_rate      = 0.75f;
    c.weak_boost_rate        = 1.5f;
    c.min_pos_scores_rate    = 0.5f;
    c.max_index              = 99999;
    return c;
}

static stream_config stream_config_from_params(const sortformer_params & p) {
    stream_config c = default_stream_config();   // AOSC tuning defaults
    c.chunk_len              = p.chunk_len;
    c.fifo_len               = p.fifo_len;
    c.spkcache_len           = p.spkcache_len;
    c.spkcache_update_period = p.spkcache_update_period;
    c.chunk_left_context     = p.chunk_left_context;
    c.chunk_right_context    = p.right_context;
    return c;
}



// Streaming state (sync mode, batch_size=1)
struct stream_state {
    std::vector<float> spkcache;       // (spkcache_len * d_model) row-major
    std::vector<float> spkcache_preds; // (spkcache_len * n_spk) row-major
    int spkcache_len;                  // current valid frames
    bool spkcache_preds_valid;         // whether spkcache_preds has been initialized

    std::vector<float> fifo;           // (fifo_len * d_model) row-major
    std::vector<float> fifo_preds;     // (fifo_len * n_spk) row-major
    int fifo_len;                      // current valid frames

    std::vector<float> mean_sil_emb;   // (d_model,)
    int n_sil_frames;
};

static stream_state init_stream_state(int d_model) {
    stream_state st;
    st.spkcache_len = 0;
    st.spkcache_preds_valid = false;
    st.fifo_len = 0;
    st.mean_sil_emb.resize(d_model, 0.0f);
    st.n_sil_frames = 0;
    return st;
}

// Update running silence profile from popped embeddings
static void update_silence_profile(
        stream_state & st,
        const stream_config & cfg,
        const float * pop_embs,   // (pop_len, d_model)
        const float * pop_preds,  // (pop_len, n_spk)
        int pop_len,
        int d_model,
        int n_spk) {
    for (int t = 0; t < pop_len; t++) {
        float pred_sum = 0;
        for (int s = 0; s < n_spk; s++) {
            pred_sum += pop_preds[t * n_spk + s];
        }
        if (pred_sum < cfg.sil_threshold) {
            st.n_sil_frames++;
            float w_old = (float)(st.n_sil_frames - 1) / (float)st.n_sil_frames;
            float w_new = 1.0f / (float)st.n_sil_frames;
            for (int d = 0; d < d_model; d++) {
                st.mean_sil_emb[d] = w_old * st.mean_sil_emb[d] + w_new * pop_embs[t * d_model + d];
            }
        }
    }
}

// AOSC: boost top-K scores per speaker
static void boost_topk_scores(
        float * scores,     // (n_frames, n_spk) row-major
        int n_frames,
        int n_spk,
        int k_per_spk,
        float scale_factor,
        float offset) {
    if (k_per_spk <= 0 || k_per_spk > n_frames) return;
    float boost = -scale_factor * logf(offset);

    for (int s = 0; s < n_spk; s++) {
        // Collect (score, frame_index) for this speaker
        std::vector<std::pair<float, int>> sv(n_frames);
        for (int t = 0; t < n_frames; t++) {
            sv[t] = {scores[t * n_spk + s], t};
        }
        // Partial sort: first k_per_spk elements are the largest
        std::nth_element(sv.begin(), sv.begin() + k_per_spk, sv.end(),
            [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
                return a.first > b.first;
            });
        for (int i = 0; i < k_per_spk; i++) {
            // -inf + finite = -inf, so disabled frames stay disabled
            scores[sv[i].second * n_spk + s] += boost;
        }
    }
}

// AOSC: compress speaker cache from st.spkcache_len frames to cfg.spkcache_len frames
static void compress_spkcache(
        stream_state & st,
        const stream_config & cfg,
        int d_model,
        int n_spk) {
    const int n_frames = st.spkcache_len;
    const int target_len = cfg.spkcache_len;
    const int spkcache_len_per_spk = target_len / n_spk - cfg.spkcache_sil_frames_per_spk;
    const int strong_k = (int)floor(spkcache_len_per_spk * cfg.strong_boost_rate);
    const int weak_k   = (int)floor(spkcache_len_per_spk * cfg.weak_boost_rate);
    const int min_pos_k = (int)floor(spkcache_len_per_spk * cfg.min_pos_scores_rate);

    fprintf(stderr, "%s: n_frames=%d → target=%d, per_spk=%d, strong_k=%d, weak_k=%d\n",
            __func__, n_frames, target_len, spkcache_len_per_spk, strong_k, weak_k);

    // 1. Compute log-based importance scores (n_frames, n_spk)
    std::vector<float> scores(n_frames * n_spk);
    for (int t = 0; t < n_frames; t++) {
        const float * p = &st.spkcache_preds[t * n_spk];
        float log_1_sum = 0;
        for (int s = 0; s < n_spk; s++)
            log_1_sum += logf(fmaxf(1.0f - p[s], cfg.pred_score_threshold));
        for (int s = 0; s < n_spk; s++) {
            float lp  = logf(fmaxf(p[s], cfg.pred_score_threshold));
            float l1p = logf(fmaxf(1.0f - p[s], cfg.pred_score_threshold));
            scores[t * n_spk + s] = lp - l1p + log_1_sum - logf(0.5f);
        }
    }

    // 2. Disable non-speech scores (preds <= 0.5 → -inf)
    for (int t = 0; t < n_frames; t++)
        for (int s = 0; s < n_spk; s++)
            if (st.spkcache_preds[t * n_spk + s] <= 0.5f)
                scores[t * n_spk + s] = -INFINITY;

    // Disable non-positive scores if speaker has enough positive ones
    for (int s = 0; s < n_spk; s++) {
        int pos_cnt = 0;
        for (int t = 0; t < n_frames; t++)
            if (scores[t * n_spk + s] > 0) pos_cnt++;
        if (pos_cnt >= min_pos_k) {
            for (int t = 0; t < n_frames; t++) {
                if (scores[t * n_spk + s] <= 0 && st.spkcache_preds[t * n_spk + s] > 0.5f)
                    scores[t * n_spk + s] = -INFINITY;
            }
        }
    }

    // 3. Boost latest frames (beyond original spkcache_len)
    if (cfg.scores_boost_latest > 0) {
        for (int t = target_len; t < n_frames; t++)
            for (int s = 0; s < n_spk; s++) {
                float & sc = scores[t * n_spk + s];
                if (sc != -INFINITY) sc += cfg.scores_boost_latest;
            }
    }

    // 4. Strong boost: top-K per speaker (scale=2)
    boost_topk_scores(scores.data(), n_frames, n_spk, strong_k, 2.0f, 0.5f);

    // 5. Weak boost: top-K per speaker (scale=1)
    int wk = std::min(weak_k, n_frames);
    boost_topk_scores(scores.data(), n_frames, n_spk, wk, 1.0f, 0.5f);

    // 6. Add silence placeholder frames at end (+inf for each speaker)
    int n_sil_pad = cfg.spkcache_sil_frames_per_spk;
    int n_total = n_frames + n_sil_pad;
    scores.resize(n_total * n_spk);
    for (int t = n_frames; t < n_total; t++)
        for (int s = 0; s < n_spk; s++)
            scores[t * n_spk + s] = INFINITY;

    // 7. Flatten as (n_spk, n_total) and find top target_len entries
    // NeMo: scores_flatten = scores.permute(0,2,1).reshape(batch,-1)
    // For batch=1: flatten[s * n_total + t] = scores[t * n_spk + s]
    int flat_len = n_spk * n_total;
    std::vector<std::pair<float, int>> flat(flat_len);
    for (int s = 0; s < n_spk; s++)
        for (int t = 0; t < n_total; t++)
            flat[s * n_total + t] = {scores[t * n_spk + s], s * n_total + t};

    // Partial sort: first target_len elements are the largest
    std::nth_element(flat.begin(), flat.begin() + target_len, flat.end(),
        [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
            return a.first > b.first;
        });

    // Replace -inf entries with max_index, keep valid entries
    std::vector<int> topk_indices(target_len);
    for (int i = 0; i < target_len; i++) {
        if (flat[i].first == -INFINITY) {
            topk_indices[i] = cfg.max_index;
        } else {
            topk_indices[i] = flat[i].second;
        }
    }

    // Sort to preserve original frame order
    std::sort(topk_indices.begin(), topk_indices.end());

    // Convert to frame indices and determine disabled mask
    int n_frames_no_sil = n_total - n_sil_pad;
    std::vector<bool> is_disabled(target_len, false);
    for (int i = 0; i < target_len; i++) {
        if (topk_indices[i] == cfg.max_index) {
            is_disabled[i] = true;
        }
        topk_indices[i] = topk_indices[i] % n_total;
        if (topk_indices[i] >= n_frames_no_sil) {
            is_disabled[i] = true;
        }
        if (is_disabled[i]) {
            topk_indices[i] = 0; // placeholder for gather
        }
    }

    // 8. Gather embeddings and predictions
    std::vector<float> new_embs(target_len * d_model);
    std::vector<float> new_preds(target_len * n_spk);

    for (int i = 0; i < target_len; i++) {
        int tidx = topk_indices[i];
        if (is_disabled[i]) {
            memcpy(&new_embs[i * d_model], st.mean_sil_emb.data(), d_model * sizeof(float));
            memset(&new_preds[i * n_spk], 0, n_spk * sizeof(float));
        } else {
            memcpy(&new_embs[i * d_model], &st.spkcache[tidx * d_model], d_model * sizeof(float));
            memcpy(&new_preds[i * n_spk], &st.spkcache_preds[tidx * n_spk], n_spk * sizeof(float));
        }
    }

    st.spkcache = std::move(new_embs);
    st.spkcache_preds = std::move(new_preds);
    st.spkcache_len = target_len;

    fprintf(stderr, "%s: compressed %d → %d frames\n", __func__, n_frames, target_len);
}



static int sortformer_compute_streaming_prediction(
        struct sortformer_context * ctx,
        const float * combined_data,
        int           T,
        int           d_model,
        float      ** pred_out) {

    if (!ctx || !combined_data || !pred_out || T <= 0 || d_model != ctx->d_model) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    const int n_heads = ctx->n_heads;
    const int d_head  = ctx->d_head;
    const int kernel_size = ctx->conv_kernel_size;
    const int n_pos = 2 * T - 1;
    const float xscale = sqrtf((float)d_model);

    fprintf(stderr, "%s: T=%d, d_model=%d\n", __func__, T, d_model);

    const size_t n_tensors = 16384;
    const int graph_nodes = 32768;
    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(graph_nodes, false),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx0 = ggml_init(gparams);
    if (!ctx0) {
        fprintf(stderr, "%s: failed to init ggml context\n", __func__);
        return -1;
    }

    auto f32 = [&](struct ggml_tensor * t) -> struct ggml_tensor * {
        return (t->type != GGML_TYPE_F32) ? ggml_cast(ctx0, t, GGML_TYPE_F32) : t;
    };

    auto layer_norm = [&](struct ggml_tensor * x,
                          struct ggml_tensor * w,
                          struct ggml_tensor * b) -> struct ggml_tensor * {
        struct ggml_tensor * n = ggml_norm(ctx0, x, 1e-5f);
        return ggml_add(ctx0, ggml_mul(ctx0, n, f32(w)), f32(b));
    };

    struct ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, T);
    ggml_set_name(inp, "stream_input");
    ggml_set_input(inp);

    struct ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, n_pos);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);

    std::vector<float> pos_emb_data((size_t)n_pos * d_model);
    if (ctx->pos_emb_cache_n_pos != n_pos) {
        ctx->pos_emb_cache.resize(n_pos * d_model);
        generate_sinusoidal_pos_emb(ctx->pos_emb_cache.data(), n_pos, d_model);
        ctx->pos_emb_cache_n_pos = n_pos;
        fprintf(stderr, "%s: computed position embeddings for n_pos=%d\n", __func__, n_pos);
    } else {
        fprintf(stderr, "%s: using cached position embeddings (n_pos=%d)\n", __func__, n_pos);
    }
    std::copy(ctx->pos_emb_cache.begin(), ctx->pos_emb_cache.end(), pos_emb_data.begin());

    struct ggml_tensor * cur = ggml_scale(ctx0, inp, xscale);

    for (int il = 0; il < ctx->n_conf_layers; il++) {
        const auto & L = ctx->conf_layers[il];
        struct ggml_tensor * residual = cur;

        // ===== FFN1 (Macaron half-step) =====
        {
            struct ggml_tensor * x = layer_norm(cur, L.norm_ff1_w, L.norm_ff1_b);
            x = ggml_mul_mat(ctx0, L.ff1_up_w, x);
            x = ggml_add(ctx0, x, f32(L.ff1_up_b));
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, L.ff1_down_w, x);
            x = ggml_add(ctx0, x, f32(L.ff1_down_b));
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f));
        }

        // ===== Multi-Head Self-Attention with Relative Position =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_sa_w, L.norm_sa_b);

            struct ggml_tensor * Qcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_q_w, x), f32(L.sa_q_b));
            struct ggml_tensor * Kcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_k_w, x), f32(L.sa_k_b));
            struct ggml_tensor * Vcur = ggml_add(ctx0,
                ggml_mul_mat(ctx0, L.sa_v_w, x), f32(L.sa_v_b));

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_heads, T);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_heads, T);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_heads, T);

            struct ggml_tensor * Q_u = ggml_add(ctx0, Qcur, f32(L.pos_bias_u));
            struct ggml_tensor * Q_v = ggml_add(ctx0, Qcur, f32(L.pos_bias_v));

            Q_u  = ggml_permute(ctx0, Q_u,  0, 2, 1, 3);
            Q_v  = ggml_permute(ctx0, Q_v,  0, 2, 1, 3);
            Kcur = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));
            Vcur = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            struct ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Q_u, Kcur);
            matrix_ac = ggml_cont(ctx0, ggml_permute(ctx0, matrix_ac, 1, 0, 2, 3));

            struct ggml_tensor * p = ggml_mul_mat(ctx0, L.sa_pos_w, pos_emb);
            p = ggml_reshape_3d(ctx0, p, d_head, n_heads, n_pos);
            p = ggml_permute(ctx0, p, 0, 2, 1, 3);

            struct ggml_tensor * matrix_bd = ggml_mul_mat(ctx0, Q_v, p);
            matrix_bd = ggml_cont(ctx0, ggml_permute(ctx0, matrix_bd, 1, 0, 2, 3));

            // Relative shift (Transformer-XL style)
            {
                const int64_t pos_len = matrix_bd->ne[0];
                const int64_t q_len   = matrix_bd->ne[1];
                const int64_t h       = matrix_bd->ne[2];
                matrix_bd = ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_roll(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                    q_len, pos_len, h,
                    matrix_bd->nb[1], matrix_bd->nb[2],
                    matrix_bd->nb[0] * q_len);
                matrix_bd = ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
            }

            matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                matrix_ac->ne[0], matrix_bd->ne[1], matrix_bd->ne[2],
                matrix_bd->nb[1], matrix_bd->nb[2], 0);

            struct ggml_tensor * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
            scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)d_head));

            struct ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            struct ggml_tensor * attn_out = ggml_mul_mat(ctx0, attn, Vcur);
            attn_out = ggml_permute(ctx0, attn_out, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, d_head * n_heads, T);

            attn_out = ggml_mul_mat(ctx0, L.sa_out_w, attn_out);
            attn_out = ggml_add(ctx0, attn_out, f32(L.sa_out_b));

            residual = ggml_add(ctx0, residual, attn_out);
        }

        // ===== Conv Module =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_conv_w, L.norm_conv_b);

            struct ggml_tensor * pw1_w = L.conv_pw1_w;
            pw1_w = ggml_reshape_2d(ctx0, pw1_w, pw1_w->ne[0] * pw1_w->ne[1], pw1_w->ne[2]);
            x = ggml_mul_mat(ctx0, pw1_w, x);
            x = ggml_add(ctx0, x, f32(L.conv_pw1_b));

            {
                int64_t half = x->ne[0] / 2;
                struct ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_view_2d(ctx0, x, half, x->ne[1], x->nb[1], half * x->nb[0]));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, half, x->ne[1], x->nb[1], 0), gate);
            }

            x = ggml_cont(ctx0, ggml_transpose(ctx0, x));

            int pad = (kernel_size - 1) / 2;
            x = ggml_pad(ctx0, x, pad, 0, 0, 0);
            x = ggml_roll(ctx0, x, pad, 0, 0, 0);
            x = ggml_pad(ctx0, x, pad, 0, 0, 0);

            struct ggml_tensor * dw_w = f32(L.conv_dw_w);
            dw_w = ggml_reshape_2d(ctx0, dw_w, dw_w->ne[0], dw_w->ne[2]);
            x = ggml_ssm_conv(ctx0, x, dw_w);
            x = ggml_add(ctx0, x, f32(L.conv_dw_b));

            x = ggml_silu(ctx0, x);

            struct ggml_tensor * pw2_w = L.conv_pw2_w;
            pw2_w = ggml_reshape_2d(ctx0, pw2_w, pw2_w->ne[0] * pw2_w->ne[1], pw2_w->ne[2]);
            x = ggml_mul_mat(ctx0, pw2_w, x);
            x = ggml_add(ctx0, x, f32(L.conv_pw2_b));

            residual = ggml_add(ctx0, residual, x);
        }

        // ===== FFN2 (Macaron half-step) =====
        {
            struct ggml_tensor * x = layer_norm(residual, L.norm_ff2_w, L.norm_ff2_b);
            x = ggml_mul_mat(ctx0, L.ff2_up_w, x);
            x = ggml_add(ctx0, x, f32(L.ff2_up_b));
            x = ggml_silu(ctx0, x);
            x = ggml_mul_mat(ctx0, L.ff2_down_w, x);
            x = ggml_add(ctx0, x, f32(L.ff2_down_b));
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, x, 0.5f));
        }

        // ===== Final LayerNorm =====
        cur = layer_norm(residual, L.norm_out_w, L.norm_out_b);
    }

    cur = ggml_mul_mat(ctx0, ctx->proj_w, cur);
    cur = ggml_add(ctx0, cur, f32(ctx->proj_b));

    for (int il = 0; il < N_TRANS_LAYERS; il++) {
        const auto & TL = ctx->trans_layers[il];

        // Self-attention sublayer
        struct ggml_tensor * attn_out;
        {
            struct ggml_tensor * Q = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.q_w, cur), f32(TL.q_b));
            struct ggml_tensor * K = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.k_w, cur), f32(TL.k_b));
            struct ggml_tensor * V = ggml_add(ctx0,
                ggml_mul_mat(ctx0, TL.v_w, cur), f32(TL.v_b));

            Q = ggml_reshape_3d(ctx0, Q, TF_D_HEAD, TF_N_HEADS, T);
            K = ggml_reshape_3d(ctx0, K, TF_D_HEAD, TF_N_HEADS, T);
            V = ggml_reshape_3d(ctx0, V, TF_D_HEAD, TF_N_HEADS, T);

            Q = ggml_permute(ctx0, Q, 0, 2, 1, 3);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

            struct ggml_tensor * scores = ggml_mul_mat(ctx0, Q, K);
            scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));
            scores = ggml_scale(ctx0, scores, 1.0f / sqrtf((float)TF_D_HEAD));

            struct ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            attn_out = ggml_mul_mat(ctx0, attn, V);
            attn_out = ggml_permute(ctx0, attn_out, 2, 0, 1, 3);
            attn_out = ggml_cont_2d(ctx0, attn_out, TF_D_HEAD * TF_N_HEADS, T);

            attn_out = ggml_mul_mat(ctx0, TL.out_w, attn_out);
            attn_out = ggml_add(ctx0, attn_out, f32(TL.out_b));
        }

        // POST-LN: x = layer_norm(x + attn_out)
        cur = layer_norm(ggml_add(ctx0, cur, attn_out), TL.ln1_w, TL.ln1_b);

        // FFN sublayer
        struct ggml_tensor * ff_out;
        {
            ff_out = ggml_mul_mat(ctx0, TL.ff_up_w, cur);
            ff_out = ggml_add(ctx0, ff_out, f32(TL.ff_up_b));
            ff_out = ggml_relu(ctx0, ff_out);
            ff_out = ggml_mul_mat(ctx0, TL.ff_down_w, ff_out);
            ff_out = ggml_add(ctx0, ff_out, f32(TL.ff_down_b));
        }

        // POST-LN: x = layer_norm(x + ff_out)
        cur = layer_norm(ggml_add(ctx0, cur, ff_out), TL.ln2_w, TL.ln2_b);
    }

    cur = ggml_relu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, ctx->pred_hidden_w, cur);
    cur = ggml_add(ctx0, cur, f32(ctx->pred_hidden_b));
    cur = ggml_relu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, ctx->pred_spk_w, cur);
    cur = ggml_add(ctx0, cur, f32(ctx->pred_spk_b));
    cur = ggml_sigmoid(ctx0, cur);

    ggml_set_name(cur, "pred_output");
    ggml_set_output(cur);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, graph_nodes, false);
    ggml_build_forward_expand(gf, cur);

    fprintf(stderr, "%s: graph nodes = %d\n", __func__, ggml_graph_n_nodes(gf));

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "%s: failed to allocate graph\n", __func__);
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_set(inp, combined_data, 0, ggml_nbytes(inp));
    ggml_backend_tensor_set(pos_emb, pos_emb_data.data(), 0, ggml_nbytes(pos_emb));

    ggml_backend_graph_compute(ctx->backend, gf);

    const int out_d = (int)cur->ne[0];
    const int out_t = (int)cur->ne[1];
    fprintf(stderr, "%s: output shape = (%d, %d)\n", __func__, out_t, out_d);

    const size_t out_size = (size_t)out_t * out_d * sizeof(float);
    float * output = (float *)malloc(out_size);
    if (!output) {
        ggml_free(ctx0);
        return -1;
    }

    ggml_backend_tensor_get(cur, output, 0, out_size);

    ggml_free(ctx0);

    *pred_out = output;
    return out_t;
}

static int validate_stream_config(const stream_config & cfg) {
    if (cfg.chunk_len < 1) {
        fprintf(stderr, "error: chunk_len must be >= 1 (got %d)\n", cfg.chunk_len);
        return -1;
    }
    if (cfg.spkcache_update_period < 1) {
        fprintf(stderr, "error: spkcache_update_period must be >= 1 (got %d)\n", cfg.spkcache_update_period);
        return -1;
    }
    if (cfg.fifo_len < 0) {
        fprintf(stderr, "error: fifo_len must be >= 0 (got %d)\n", cfg.fifo_len);
        return -1;
    }
    int min_spkcache = (1 + cfg.spkcache_sil_frames_per_spk) * STREAM_N_SPK;
    if (cfg.spkcache_len < min_spkcache) {
        fprintf(stderr, "error: spkcache_len must be >= %d (got %d)\n", min_spkcache, cfg.spkcache_len);
        return -1;
    }
    if (cfg.chunk_left_context < 0) {
        fprintf(stderr, "error: chunk_left_context must be >= 0 (got %d)\n", cfg.chunk_left_context);
        return -1;
    }
    if (cfg.chunk_right_context < 0) {
        fprintf(stderr, "error: chunk_right_context must be >= 0 (got %d)\n", cfg.chunk_right_context);
        return -1;
    }
    // Warnings (non-fatal, matching NeMo behavior)
    if (cfg.spkcache_update_period < cfg.chunk_len) {
        fprintf(stderr, "warning: spkcache_update_period (%d) < chunk_len (%d), "
                "effective period will be %d\n",
                cfg.spkcache_update_period, cfg.chunk_len, cfg.chunk_len);
    }
    if (cfg.spkcache_update_period > cfg.fifo_len + cfg.chunk_len) {
        fprintf(stderr, "warning: spkcache_update_period (%d) > fifo_len + chunk_len (%d), "
                "effective period will be %d\n",
                cfg.spkcache_update_period, cfg.fifo_len + cfg.chunk_len,
                cfg.fifo_len + cfg.chunk_len);
    }
    return 0;
}

// ============================================================================
// Diarize (streaming pipeline)
// ============================================================================

int sortformer_diarize(
        struct sortformer_context * ctx,
        const float * audio_samples,
        int           n_samples,
        float       * probs_out,
        int           n_frames_max) {

    if (!ctx || !audio_samples || n_samples <= 0 || !probs_out || n_frames_max <= 0) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }

    const int n_spk = STREAM_N_SPK;
    const int d_model = ctx->d_model;
    const int subsampling = ctx->subsampling_factor;

#ifdef SORTFORMER_USE_COREML
    const bool use_coreml = (ctx->coreml_ctx != nullptr);
    if (use_coreml) {
        fprintf(stderr, "%s: using CoreML acceleration for head\n", __func__);
    }
#else
    const bool use_coreml = false;
#endif
    const stream_config cfg = stream_config_from_params(ctx->params);
    if (validate_stream_config(cfg) != 0) {
        return -1;
    }

    fprintf(stderr, "%s: starting streaming diarization (chunk_len=%d, spkcache_len=%d)\n",
            __func__, cfg.chunk_len, cfg.spkcache_len);

    int64_t t_mel_us = 0, t_preenc_us = 0, t_head_us = 0;

    int64_t t0 = ggml_time_us();
    float * mel = nullptr;
    int n_mels = 0;
    int seq_len = 0;
    int n_mel_frames = sortformer_compute_mel(ctx, audio_samples, n_samples, &mel, &n_mels, &seq_len);
    if (n_mel_frames < 0) {
        fprintf(stderr, "%s: mel computation failed\n", __func__);
        return -1;
    }
    t_mel_us = ggml_time_us() - t0;
    fprintf(stderr, "%s: mel shape=(%d, %d), seq_len=%d\n", __func__, n_mels, n_mel_frames, seq_len);

    stream_state st = init_stream_state(d_model);

    // Use seq_len (unpadded) as feat_len for chunking
    const int feat_len = seq_len;
    std::vector<float> total_preds;
    int total_pred_frames = 0;

    // ---- Step 3: Chunk loop ----
    int stt_feat = 0;
    int chunk_idx = 0;

    while (stt_feat < feat_len) {
        int end_feat = std::min(stt_feat + cfg.chunk_len * subsampling, feat_len);
        int left_offset = std::min(cfg.chunk_left_context * subsampling, stt_feat);
        int right_offset = std::min(cfg.chunk_right_context * subsampling, feat_len - end_feat);

        int chunk_mel_start = stt_feat - left_offset;
        int chunk_mel_end   = end_feat + right_offset;
        int chunk_mel_frames = chunk_mel_end - chunk_mel_start;

        fprintf(stderr, "%s: chunk %d: stt=%d, end=%d, lo=%d, ro=%d, mel=[%d:%d] (%d frames)\n",
                __func__, chunk_idx, stt_feat, end_feat,
                left_offset, right_offset, chunk_mel_start, chunk_mel_end, chunk_mel_frames);

        // ---- 3a. Extract chunk mel into contiguous buffer ----
        std::vector<float> chunk_mel(n_mels * chunk_mel_frames);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < chunk_mel_frames; t++) {
                chunk_mel[m * chunk_mel_frames + t] = mel[m * n_mel_frames + (chunk_mel_start + t)];
            }
        }

        int lc = (int)round((double)left_offset / subsampling);
        int rc = (int)ceil((double)right_offset / subsampling);

        float * pred_out = nullptr;
        float * chunk_preenc = nullptr;
        int chunk_preenc_frames = 0;
        int n_pred = 0;
        int chunk_len_used = 0;

        {
            int64_t t1 = ggml_time_us();
            int d_model_out = 0;
            chunk_preenc_frames = sortformer_compute_preenc(
                ctx, chunk_mel.data(), n_mels, chunk_mel_frames, chunk_mel_frames,
                &chunk_preenc, &d_model_out);
            if (chunk_preenc_frames < 0) {
                fprintf(stderr, "%s: pre-encode failed for chunk %d\n", __func__, chunk_idx);
                free(mel);
                return -1;
            }
            t_preenc_us += ggml_time_us() - t1;

            chunk_len_used = chunk_preenc_frames - lc - rc;

            fprintf(stderr, "%s: chunk %d: preenc=%d frames, lc=%d, rc=%d, chunk_len_used=%d\n",
                    __func__, chunk_idx, chunk_preenc_frames, lc, rc, chunk_len_used);

            int T_total = st.spkcache_len + st.fifo_len + chunk_preenc_frames;
            std::vector<float> combined(T_total * d_model);

            if (st.spkcache_len > 0) {
                memcpy(combined.data(),
                       st.spkcache.data(),
                       st.spkcache_len * d_model * sizeof(float));
            }
            if (st.fifo_len > 0) {
                memcpy(combined.data() + st.spkcache_len * d_model,
                       st.fifo.data(),
                       st.fifo_len * d_model * sizeof(float));
            }
            memcpy(combined.data() + (st.spkcache_len + st.fifo_len) * d_model,
                   chunk_preenc,
                   chunk_preenc_frames * d_model * sizeof(float));

            int64_t t2 = ggml_time_us();
#ifdef SORTFORMER_USE_COREML
            if (use_coreml && T_total <= SORTFORMER_COREML_MAX_SEQ_LEN) {
                pred_out = (float *)malloc(T_total * n_spk * sizeof(float));
                int ret = sortformer_coreml_encode(ctx->coreml_ctx, combined.data(), T_total, pred_out);
                if (ret != 0) {
                    fprintf(stderr, "%s: CoreML head failed for chunk %d\n", __func__, chunk_idx);
                    free(chunk_preenc); free(mel); free(pred_out);
                    return -1;
                }
                n_pred = T_total;
            } else
#endif
            {
#ifdef SORTFORMER_USE_COREML
                if (use_coreml && T_total > SORTFORMER_COREML_MAX_SEQ_LEN) {
                    fprintf(stderr, "%s: T_total=%d exceeds CoreML max %d, falling back to GGML\n",
                            __func__, T_total, SORTFORMER_COREML_MAX_SEQ_LEN);
                }
#endif
                n_pred = sortformer_compute_streaming_prediction(ctx, combined.data(), T_total, d_model, &pred_out);
                if (n_pred < 0) {
                    fprintf(stderr, "%s: GGML head failed for chunk %d\n", __func__, chunk_idx);
                    free(chunk_preenc); free(mel);
                    return -1;
                }
            }
            t_head_us += ggml_time_us() - t2;
        }

        // ---- 3e. Extract chunk predictions ----
        int pred_start = st.spkcache_len + st.fifo_len + lc;
        int pred_end   = pred_start + chunk_len_used;

        fprintf(stderr, "%s: chunk %d: extracting preds [%d:%d] from %d total frames\n",
                __func__, chunk_idx, pred_start, pred_end, n_pred);

        // Append chunk predictions to total
        total_preds.insert(total_preds.end(),
                           pred_out + pred_start * n_spk,
                           pred_out + pred_end * n_spk);
        total_pred_frames += chunk_len_used;

        // ---- 3f. Streaming state update (sync mode) ----
        {
            int old_sc_len = st.spkcache_len;
            int old_fifo_len = st.fifo_len;

            // Extract FIFO predictions from full prediction output
            // (for completeness, though FIFO is always empty in our config)
            st.fifo_preds.resize(old_fifo_len * n_spk);
            if (old_fifo_len > 0) {
                memcpy(st.fifo_preds.data(),
                       pred_out + old_sc_len * n_spk,
                       old_fifo_len * n_spk * sizeof(float));
            }

            // Extract chunk predictions for update
            std::vector<float> chunk_preds_vec(chunk_len_used * n_spk);
            memcpy(chunk_preds_vec.data(),
                   pred_out + pred_start * n_spk,
                   chunk_len_used * n_spk * sizeof(float));

            // Strip context: chunk_embs = chunk_preenc[lc : lc + chunk_len_used]
            std::vector<float> chunk_embs(chunk_len_used * d_model);
            memcpy(chunk_embs.data(),
                   chunk_preenc + lc * d_model,
                   chunk_len_used * d_model * sizeof(float));

            // Append chunk to FIFO
            int new_fifo_total = old_fifo_len + chunk_len_used;
            std::vector<float> updated_fifo((old_fifo_len + chunk_len_used) * d_model);
            std::vector<float> updated_fifo_preds((old_fifo_len + chunk_len_used) * n_spk);

            if (old_fifo_len > 0) {
                memcpy(updated_fifo.data(), st.fifo.data(), old_fifo_len * d_model * sizeof(float));
                memcpy(updated_fifo_preds.data(), st.fifo_preds.data(), old_fifo_len * n_spk * sizeof(float));
            }
            memcpy(updated_fifo.data() + old_fifo_len * d_model,
                   chunk_embs.data(), chunk_len_used * d_model * sizeof(float));
            memcpy(updated_fifo_preds.data() + old_fifo_len * n_spk,
                   chunk_preds_vec.data(), chunk_len_used * n_spk * sizeof(float));

            // Check if FIFO overflows
            if (new_fifo_total > cfg.fifo_len) {
                // Compute pop_out_len
                int pop_out_len = cfg.spkcache_update_period;
                pop_out_len = std::max(pop_out_len, chunk_len_used - cfg.fifo_len + old_fifo_len);
                pop_out_len = std::min(pop_out_len, new_fifo_total);

                fprintf(stderr, "%s: chunk %d: pop_out_len=%d\n", __func__, chunk_idx, pop_out_len);

                // Pop from FIFO front
                const float * pop_embs  = updated_fifo.data();
                const float * pop_preds = updated_fifo_preds.data();

                // Update silence profile
                update_silence_profile(st, cfg, pop_embs, pop_preds, pop_out_len, d_model, n_spk);

                // Remaining FIFO
                int remaining_fifo = new_fifo_total - pop_out_len;
                st.fifo.resize(remaining_fifo * d_model);
                st.fifo_preds.resize(remaining_fifo * n_spk);
                if (remaining_fifo > 0) {
                    memcpy(st.fifo.data(),
                           updated_fifo.data() + pop_out_len * d_model,
                           remaining_fifo * d_model * sizeof(float));
                    memcpy(st.fifo_preds.data(),
                           updated_fifo_preds.data() + pop_out_len * n_spk,
                           remaining_fifo * n_spk * sizeof(float));
                }
                st.fifo_len = remaining_fifo;

                // Append popped frames to spkcache
                int new_sc_len = old_sc_len + pop_out_len;
                st.spkcache.resize(new_sc_len * d_model);
                memcpy(st.spkcache.data() + old_sc_len * d_model,
                       pop_embs, pop_out_len * d_model * sizeof(float));

                if (st.spkcache_preds_valid) {
                    st.spkcache_preds.resize(new_sc_len * n_spk);
                    memcpy(st.spkcache_preds.data() + old_sc_len * n_spk,
                           pop_preds, pop_out_len * n_spk * sizeof(float));
                }
                st.spkcache_len = new_sc_len;

                // Check if compression needed
                if (new_sc_len > cfg.spkcache_len) {
                    if (!st.spkcache_preds_valid) {
                        // First time: init spkcache_preds from current forward pass
                        st.spkcache_preds.resize(new_sc_len * n_spk);
                        // Copy predictions for old spkcache frames (from current model output)
                        memcpy(st.spkcache_preds.data(),
                               pred_out,
                               old_sc_len * n_spk * sizeof(float));
                        // Copy pop_out predictions
                        memcpy(st.spkcache_preds.data() + old_sc_len * n_spk,
                               pop_preds,
                               pop_out_len * n_spk * sizeof(float));
                        st.spkcache_preds_valid = true;
                    }
                    compress_spkcache(st, cfg, d_model, n_spk);
                }
            } else {
                // No overflow — just update FIFO
                st.fifo = std::move(updated_fifo);
                st.fifo_preds = std::move(updated_fifo_preds);
                st.fifo_len = new_fifo_total;
            }
        }

        free(chunk_preenc);
        free(pred_out);

        stt_feat = end_feat;
        chunk_idx++;
    }

    free(mel);

    // ---- Step 4: Copy results to output ----
    int n_out = std::min(total_pred_frames, n_frames_max);
    memcpy(probs_out, total_preds.data(), n_out * n_spk * sizeof(float));

    fprintf(stderr, "\n%s: timing breakdown:\n", __func__);
    fprintf(stderr, "  mel:      %7.2f ms\n", t_mel_us / 1000.0);
    fprintf(stderr, "  preenc:   %7.2f ms\n", t_preenc_us / 1000.0);
    fprintf(stderr, "  head:     %7.2f ms (conformer + transformer + prediction)\n", t_head_us / 1000.0);
    fprintf(stderr, "  total:    %7.2f ms\n", (t_mel_us + t_preenc_us + t_head_us) / 1000.0);

    fprintf(stderr, "%s: streaming complete, %d frames output\n", __func__, n_out);
    return n_out;
}

// ============================================================================
// RTTM post-processing
// ============================================================================

// 1D median filter on binary array (zero-padded at edges)
static void median_filter_1d(const uint8_t * in, uint8_t * out, int len, int win) {
    if (win <= 1) {
        memcpy(out, in, len);
        return;
    }
    int half = win / 2;
    for (int i = 0; i < len; i++) {
        int lo = i - half;
        int hi = lo + win;
        int ones = 0;
        for (int j = lo; j < hi; j++) {
            if (j >= 0 && j < len) {
                ones += in[j];
            }
            // else: zero-padded → contributes 0
        }
        // median of binary values: majority vote
        out[i] = (ones * 2 > win) ? 1 : 0;
    }
}

int sortformer_to_rttm(
        const float * probs,
        int           n_frames,
        float         threshold,
        int           median_filter_win,
        const char  * filename,
        char        * rttm_out,
        int           rttm_out_size) {

    if (!probs || n_frames <= 0 || !rttm_out || rttm_out_size <= 0) {
        return -1;
    }

    const int n_spk = 4;
    const float frame_dur = 0.08f; // 80ms per frame

    // Extract basename without extension for RTTM
    std::string fname = filename ? filename : "unknown";
    {
        size_t slash = fname.find_last_of("/\\");
        if (slash != std::string::npos) {
            fname = fname.substr(slash + 1);
        }
        size_t dot = fname.rfind('.');
        if (dot != std::string::npos) {
            fname = fname.substr(0, dot);
        }
    }

    // Step 1: Threshold to binary
    std::vector<uint8_t> binary(n_frames * n_spk);
    for (int t = 0; t < n_frames; t++) {
        for (int s = 0; s < n_spk; s++) {
            binary[t * n_spk + s] = (probs[t * n_spk + s] > threshold) ? 1 : 0;
        }
    }

    // Step 2: Median filter per speaker
    if (median_filter_win > 1) {
        std::vector<uint8_t> filtered(n_frames);
        std::vector<uint8_t> speaker_col(n_frames);
        for (int s = 0; s < n_spk; s++) {
            for (int t = 0; t < n_frames; t++) {
                speaker_col[t] = binary[t * n_spk + s];
            }
            median_filter_1d(speaker_col.data(), filtered.data(), n_frames, median_filter_win);
            for (int t = 0; t < n_frames; t++) {
                binary[t * n_spk + s] = filtered[t];
            }
        }
    }

    // Step 3: Extract segments and generate RTTM
    int written = 0;
    for (int s = 0; s < n_spk; s++) {
        int seg_start = -1;
        for (int t = 0; t <= n_frames; t++) {
            bool active = (t < n_frames) && binary[t * n_spk + s];
            if (active && seg_start < 0) {
                seg_start = t;
            } else if (!active && seg_start >= 0) {
                float start_sec = seg_start * frame_dur;
                float dur_sec   = (t - seg_start) * frame_dur;
                int n = snprintf(rttm_out + written, rttm_out_size - written,
                                 "SPEAKER %s 1 %.2f %.2f <NA> <NA> speaker_%d <NA> <NA>\n",
                                 fname.c_str(), start_sec, dur_sec, s);
                if (n < 0 || written + n >= rttm_out_size) {
                    return -1; // buffer too small
                }
                written += n;
                seg_start = -1;
            }
        }
    }

    return written;
}

// ============================================================================
// Streaming API
// ============================================================================

// Public streaming state structure
struct sortformer_stream_state {
    sortformer_context * ctx;           // Model context (not owned)
    stream_config cfg;                  // Streaming config
    stream_state st;                    // Internal state (spkcache, fifo, etc.)
    
    // Mel overlap buffer (last n_fft - hop samples for continuity)
    std::vector<float> audio_overlap;   // size: n_fft - hop = 512 - 160 = 352
    
    // Accumulated mel buffer for incomplete chunks
    std::vector<float> mel_buffer;      // (n_mels, mel_buffer_frames)
    int mel_buffer_frames;              // valid frames in mel_buffer
    
    // Accumulated sample/frame counts
    int64_t total_samples_fed;
    int64_t total_frames_output;
    
    // Mel computation state
    int stt_feat;                       // current position in mel feature space
};

// Convert sortformer_stream_params to internal stream_config
static stream_config stream_config_from_stream_params(const sortformer_stream_params & p) {
    stream_config c = default_stream_config();   // AOSC tuning defaults
    c.chunk_len              = p.chunk_len;
    c.fifo_len               = p.fifo_len;
    c.spkcache_len           = p.spkcache_len;
    c.spkcache_update_period = p.spkcache_update_period;
    c.chunk_left_context     = p.left_context;
    c.chunk_right_context    = p.right_context;
    return c;
}

struct sortformer_stream_params sortformer_stream_preset_params(enum sortformer_stream_preset preset) {
    struct sortformer_stream_params p;
    switch (preset) {
        case SORTFORMER_PRESET_LOW_LATENCY:
            p.chunk_len = 6; p.right_context = 7; p.left_context = 1;
            p.fifo_len = 188; p.spkcache_len = 188; p.spkcache_update_period = 144;
            break;
        case SORTFORMER_PRESET_2S:
            p.chunk_len = 15; p.right_context = 10; p.left_context = 1;
            p.fifo_len = 100; p.spkcache_len = 188; p.spkcache_update_period = 144;
            break;
        case SORTFORMER_PRESET_3S:
            p.chunk_len = 30; p.right_context = 7; p.left_context = 1;
            p.fifo_len = 100; p.spkcache_len = 188; p.spkcache_update_period = 100;
            break;
        case SORTFORMER_PRESET_5S:
        default:
            p.chunk_len = 55; p.right_context = 7; p.left_context = 1;
            p.fifo_len = 100; p.spkcache_len = 188; p.spkcache_update_period = 100;
            break;
    }
    return p;
}

struct sortformer_stream_state * sortformer_stream_init(
        struct sortformer_context * ctx,
        enum sortformer_stream_preset preset) {
    return sortformer_stream_init_with_params(ctx, sortformer_stream_preset_params(preset));
}

struct sortformer_stream_state * sortformer_stream_init_with_params(
        struct sortformer_context * ctx,
        struct sortformer_stream_params params) {
    
    if (!ctx) {
        fprintf(stderr, "%s: invalid context\n", __func__);
        return nullptr;
    }
    
    stream_config cfg = stream_config_from_stream_params(params);
    if (validate_stream_config(cfg) != 0) {
        return nullptr;
    }
    
    sortformer_stream_state * sst = new sortformer_stream_state();
    sst->ctx = ctx;
    sst->cfg = cfg;
    sst->st = init_stream_state(ctx->d_model);
    
    // Initialize audio overlap buffer (empty initially)
    // Size: n_fft - hop = 512 - 160 = 352 samples
    sst->audio_overlap.clear();
    
    // Initialize mel buffer (empty initially)
    sst->mel_buffer.clear();
    sst->mel_buffer_frames = 0;
    
    // Initialize counters
    sst->total_samples_fed = 0;
    sst->total_frames_output = 0;
    sst->stt_feat = 0;
    
    fprintf(stderr, "%s: initialized streaming state (chunk_len=%d, fifo_len=%d, spkcache_len=%d)\n",
            __func__, cfg.chunk_len, cfg.fifo_len, cfg.spkcache_len);
    
    return sst;
}

int sortformer_stream_feed(
        struct sortformer_stream_state * sst,
        const float * audio_samples,
        int           n_samples,
        float       * probs_out,
        int           probs_out_max) {
    
    if (!sst || !audio_samples || n_samples <= 0 || !probs_out || probs_out_max <= 0) {
        fprintf(stderr, "%s: invalid arguments\n", __func__);
        return -1;
    }
    
    // Track cumulative samples for accurate mel frame counting
    int64_t samples_before = sst->total_samples_fed;
    sst->total_samples_fed += n_samples;
    
    sortformer_context * ctx = sst->ctx;
    const stream_config & cfg = sst->cfg;
    stream_state & st = sst->st;
    
    const int n_spk = STREAM_N_SPK;
    const int d_model = ctx->d_model;
    const int subsampling = ctx->subsampling_factor;
    const int n_mels = ctx->n_mels;
    
#ifdef SORTFORMER_USE_COREML
    const bool use_coreml = (ctx->coreml_ctx != nullptr);
#else
    const bool use_coreml = false;
#endif
    
    // 1. Prepend audio overlap to incoming samples
    int total_audio_len = (int)sst->audio_overlap.size() + n_samples;
    std::vector<float> combined_audio(total_audio_len);
    if (!sst->audio_overlap.empty()) {
        memcpy(combined_audio.data(), sst->audio_overlap.data(), 
               sst->audio_overlap.size() * sizeof(float));
    }
    memcpy(combined_audio.data() + sst->audio_overlap.size(), 
           audio_samples, n_samples * sizeof(float));
    
    // 2. Compute mel spectrogram for combined audio
    float * mel = nullptr;
    int n_mels_out = 0;
    int seq_len = 0;
    int n_mel_frames = sortformer_compute_mel(ctx, combined_audio.data(), total_audio_len, 
                                               &mel, &n_mels_out, &seq_len);
    if (n_mel_frames < 0) {
        fprintf(stderr, "%s: mel computation failed\n", __func__);
        return -1;
    }
    
    // 3. Save audio overlap for next call (last n_fft - hop samples)
    int overlap_samples = ctx->n_fft - ctx->hop_length;  // 512 - 160 = 352
    if (total_audio_len > overlap_samples) {
        sst->audio_overlap.resize(overlap_samples);
        memcpy(sst->audio_overlap.data(), 
               combined_audio.data() + total_audio_len - overlap_samples,
               overlap_samples * sizeof(float));
    } else {
        // Not enough samples yet, keep all
        sst->audio_overlap = combined_audio;
    }
    
    // 4. Combine with any leftover mel frames from previous call
    // Calculate new mel frames based on cumulative sample count to avoid fractional drift
    if (samples_before < 0) samples_before = 0;
    int expected_frames_before = (int)(samples_before / ctx->hop_length);
    int expected_frames_after = (int)(sst->total_samples_fed / ctx->hop_length);
    int new_mel_frames = expected_frames_after - expected_frames_before;
    
    if (new_mel_frames > seq_len) new_mel_frames = seq_len;
    if (new_mel_frames < 0) new_mel_frames = 0;
    
    int mel_frames_to_skip = seq_len - new_mel_frames;
    
    int total_mel_frames = sst->mel_buffer_frames + new_mel_frames;
    std::vector<float> combined_mel(n_mels * total_mel_frames);
    
    // Copy leftover mel frames
    if (sst->mel_buffer_frames > 0) {
        for (int m = 0; m < n_mels; m++) {
            memcpy(combined_mel.data() + m * total_mel_frames,
                   sst->mel_buffer.data() + m * sst->mel_buffer_frames,
                   sst->mel_buffer_frames * sizeof(float));
        }
    }
    
    // Copy new mel frames (skip the overlap region)
    for (int m = 0; m < n_mels; m++) {
        memcpy(combined_mel.data() + m * total_mel_frames + sst->mel_buffer_frames,
               mel + m * n_mel_frames + mel_frames_to_skip,
               new_mel_frames * sizeof(float));
    }
    free(mel);
    
    // 5. Process chunks
    const int feat_len = total_mel_frames;
    std::vector<float> chunk_preds;
    int total_new_frames = 0;
    int stt_feat = 0;  // Start from beginning of combined mel
    
    // Minimum mel frames needed for one chunk with right context
    const int min_chunk_mel = cfg.chunk_len * subsampling + cfg.chunk_right_context * subsampling;
    
    while (stt_feat < feat_len) {
        int remaining_feat = feat_len - stt_feat;
        
        // If we don't have enough frames for a full chunk + right context, save for next call
        if (remaining_feat < min_chunk_mel) {
            break;
        }
        
        int end_feat = std::min(stt_feat + cfg.chunk_len * subsampling, feat_len);
        int left_offset = std::min(cfg.chunk_left_context * subsampling, stt_feat);
        int right_offset = std::min(cfg.chunk_right_context * subsampling, feat_len - end_feat);
        
        int chunk_mel_start = stt_feat - left_offset;
        int chunk_mel_end   = end_feat + right_offset;
        int chunk_mel_frames = chunk_mel_end - chunk_mel_start;
        
        // Extract chunk mel into contiguous buffer
        std::vector<float> chunk_mel(n_mels * chunk_mel_frames);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < chunk_mel_frames; t++) {
                chunk_mel[m * chunk_mel_frames + t] = combined_mel[m * total_mel_frames + (chunk_mel_start + t)];
            }
        }
        
        int lc = (int)round((double)left_offset / subsampling);
        int rc = (int)ceil((double)right_offset / subsampling);
        
        float * pred_out = nullptr;
        float * chunk_preenc = nullptr;
        int chunk_preenc_frames = 0;
        int n_pred = 0;
        int chunk_len_used = 0;
        
        // Pre-encode chunk
        {
            int d_model_out = 0;
            chunk_preenc_frames = sortformer_compute_preenc(
                ctx, chunk_mel.data(), n_mels, chunk_mel_frames, chunk_mel_frames,
                &chunk_preenc, &d_model_out);
            if (chunk_preenc_frames < 0) {
                fprintf(stderr, "%s: pre-encode failed\n", __func__);
                return -1;
            }
            
            chunk_len_used = chunk_preenc_frames - lc - rc;
            
            // Combine [spkcache, fifo, chunk_preenc]
            int T_total = st.spkcache_len + st.fifo_len + chunk_preenc_frames;
            std::vector<float> combined(T_total * d_model);
            
            if (st.spkcache_len > 0) {
                memcpy(combined.data(),
                       st.spkcache.data(),
                       st.spkcache_len * d_model * sizeof(float));
            }
            if (st.fifo_len > 0) {
                memcpy(combined.data() + st.spkcache_len * d_model,
                       st.fifo.data(),
                       st.fifo_len * d_model * sizeof(float));
            }
            memcpy(combined.data() + (st.spkcache_len + st.fifo_len) * d_model,
                   chunk_preenc,
                   chunk_preenc_frames * d_model * sizeof(float));
            
            // Run head (CoreML or GGML)
#ifdef SORTFORMER_USE_COREML
            if (use_coreml && T_total <= SORTFORMER_COREML_MAX_SEQ_LEN) {
                pred_out = (float *)malloc(T_total * n_spk * sizeof(float));
                int ret = sortformer_coreml_encode(ctx->coreml_ctx, combined.data(), T_total, pred_out);
                if (ret != 0) {
                    fprintf(stderr, "%s: CoreML head failed\n", __func__);
                    free(chunk_preenc); free(pred_out);
                    return -1;
                }
                n_pred = T_total;
            } else
#endif
            {
                n_pred = sortformer_compute_streaming_prediction(ctx, combined.data(), T_total, d_model, &pred_out);
                if (n_pred < 0) {
                    fprintf(stderr, "%s: GGML head failed\n", __func__);
                    free(chunk_preenc);
                    return -1;
                }
            }
        }
        
        // Extract chunk predictions
        int pred_start = st.spkcache_len + st.fifo_len + lc;
        int pred_end   = pred_start + chunk_len_used;
        
        // Append chunk predictions to output
        chunk_preds.insert(chunk_preds.end(),
                           pred_out + pred_start * n_spk,
                           pred_out + pred_end * n_spk);
        total_new_frames += chunk_len_used;
        
        // Streaming state update (same as sortformer_diarize)
        {
            int old_sc_len = st.spkcache_len;
            int old_fifo_len = st.fifo_len;
            
            // Extract FIFO predictions from full prediction output
            st.fifo_preds.resize(old_fifo_len * n_spk);
            if (old_fifo_len > 0) {
                memcpy(st.fifo_preds.data(),
                       pred_out + old_sc_len * n_spk,
                       old_fifo_len * n_spk * sizeof(float));
            }
            
            // Extract chunk predictions for update
            std::vector<float> chunk_preds_vec(chunk_len_used * n_spk);
            memcpy(chunk_preds_vec.data(),
                   pred_out + pred_start * n_spk,
                   chunk_len_used * n_spk * sizeof(float));
            
            // Strip context: chunk_embs = chunk_preenc[lc : lc + chunk_len_used]
            std::vector<float> chunk_embs(chunk_len_used * d_model);
            memcpy(chunk_embs.data(),
                   chunk_preenc + lc * d_model,
                   chunk_len_used * d_model * sizeof(float));
            
            // Append chunk to FIFO
            int new_fifo_total = old_fifo_len + chunk_len_used;
            std::vector<float> updated_fifo((old_fifo_len + chunk_len_used) * d_model);
            std::vector<float> updated_fifo_preds((old_fifo_len + chunk_len_used) * n_spk);
            
            if (old_fifo_len > 0) {
                memcpy(updated_fifo.data(), st.fifo.data(), old_fifo_len * d_model * sizeof(float));
                memcpy(updated_fifo_preds.data(), st.fifo_preds.data(), old_fifo_len * n_spk * sizeof(float));
            }
            memcpy(updated_fifo.data() + old_fifo_len * d_model,
                   chunk_embs.data(), chunk_len_used * d_model * sizeof(float));
            memcpy(updated_fifo_preds.data() + old_fifo_len * n_spk,
                   chunk_preds_vec.data(), chunk_len_used * n_spk * sizeof(float));
            
            // Check if FIFO overflows
            if (new_fifo_total > cfg.fifo_len) {
                // Compute pop_out_len
                int pop_out_len = cfg.spkcache_update_period;
                pop_out_len = std::max(pop_out_len, chunk_len_used - cfg.fifo_len + old_fifo_len);
                pop_out_len = std::min(pop_out_len, new_fifo_total);
                
                // Pop from FIFO front
                const float * pop_embs  = updated_fifo.data();
                const float * pop_preds = updated_fifo_preds.data();
                
                // Update silence profile
                update_silence_profile(st, cfg, pop_embs, pop_preds, pop_out_len, d_model, n_spk);
                
                // Remaining FIFO
                int remaining_fifo = new_fifo_total - pop_out_len;
                st.fifo.resize(remaining_fifo * d_model);
                st.fifo_preds.resize(remaining_fifo * n_spk);
                if (remaining_fifo > 0) {
                    memcpy(st.fifo.data(),
                           updated_fifo.data() + pop_out_len * d_model,
                           remaining_fifo * d_model * sizeof(float));
                    memcpy(st.fifo_preds.data(),
                           updated_fifo_preds.data() + pop_out_len * n_spk,
                           remaining_fifo * n_spk * sizeof(float));
                }
                st.fifo_len = remaining_fifo;
                
                // Append popped frames to spkcache
                int new_sc_len = old_sc_len + pop_out_len;
                st.spkcache.resize(new_sc_len * d_model);
                memcpy(st.spkcache.data() + old_sc_len * d_model,
                       pop_embs, pop_out_len * d_model * sizeof(float));
                
                if (st.spkcache_preds_valid) {
                    st.spkcache_preds.resize(new_sc_len * n_spk);
                    memcpy(st.spkcache_preds.data() + old_sc_len * n_spk,
                           pop_preds, pop_out_len * n_spk * sizeof(float));
                }
                st.spkcache_len = new_sc_len;
                
                // Check if compression needed
                if (new_sc_len > cfg.spkcache_len) {
                    if (!st.spkcache_preds_valid) {
                        // First time: init spkcache_preds from current forward pass
                        st.spkcache_preds.resize(new_sc_len * n_spk);
                        // Copy predictions for old spkcache frames (from current model output)
                        memcpy(st.spkcache_preds.data(),
                               pred_out,
                               old_sc_len * n_spk * sizeof(float));
                        // Copy pop_out predictions
                        memcpy(st.spkcache_preds.data() + old_sc_len * n_spk,
                               pop_preds,
                               pop_out_len * n_spk * sizeof(float));
                        st.spkcache_preds_valid = true;
                    }
                    compress_spkcache(st, cfg, d_model, n_spk);
                }
            } else {
                // No overflow — just update FIFO
                st.fifo = std::move(updated_fifo);
                st.fifo_preds = std::move(updated_fifo_preds);
                st.fifo_len = new_fifo_total;
            }
        }
        
        free(chunk_preenc);
        free(pred_out);
        
        stt_feat = end_feat;
    }
    
    // 6. Save remaining mel frames for next call
    int remaining_mel = feat_len - stt_feat;
    if (remaining_mel > 0) {
        sst->mel_buffer.resize(n_mels * remaining_mel);
        for (int m = 0; m < n_mels; m++) {
            memcpy(sst->mel_buffer.data() + m * remaining_mel,
                   combined_mel.data() + m * total_mel_frames + stt_feat,
                   remaining_mel * sizeof(float));
        }
        sst->mel_buffer_frames = remaining_mel;
    } else {
        sst->mel_buffer.clear();
        sst->mel_buffer_frames = 0;
    }
    
    // 7. Copy results to output
    int n_out = std::min(total_new_frames, probs_out_max);
    if (n_out > 0) {
        memcpy(probs_out, chunk_preds.data(), n_out * n_spk * sizeof(float));
    }
    
    sst->total_frames_output += n_out;
    
    return n_out;
}

int sortformer_stream_flush(
        struct sortformer_stream_state * sst,
        float       * probs_out,
        int           probs_out_max) {
    
    if (!sst || !probs_out || probs_out_max <= 0) {
        return 0;
    }
    
    if (sst->mel_buffer_frames == 0 && sst->audio_overlap.empty()) {
        return 0;
    }
    
    sortformer_context * ctx = sst->ctx;
    const stream_config & cfg = sst->cfg;
    stream_state & st = sst->st;
    
    const int n_spk = STREAM_N_SPK;
    const int d_model = ctx->d_model;
    const int subsampling = ctx->subsampling_factor;
    const int n_mels = ctx->n_mels;
    
#ifdef SORTFORMER_USE_COREML
    const bool use_coreml = (ctx->coreml_ctx != nullptr);
#else
    const bool use_coreml = false;
#endif
    
    // Only process leftover mel_buffer frames - audio_overlap samples were already counted
    std::vector<float> combined_mel;
    int total_mel_frames = sst->mel_buffer_frames;
    
    if (sst->mel_buffer_frames > 0) {
        combined_mel = sst->mel_buffer;
    } else {
        return 0;
    }
    
    if (total_mel_frames == 0) {
        return 0;
    }
    
    std::vector<float> chunk_preds;
    int total_new_frames = 0;
    int stt_feat = 0;
    
    while (stt_feat < total_mel_frames) {
        int end_feat = std::min(stt_feat + cfg.chunk_len * subsampling, total_mel_frames);
        int left_offset = std::min(cfg.chunk_left_context * subsampling, stt_feat);
        int right_offset = std::min(cfg.chunk_right_context * subsampling, total_mel_frames - end_feat);
        
        int chunk_mel_start = stt_feat - left_offset;
        int chunk_mel_end   = end_feat + right_offset;
        int chunk_mel_frames = chunk_mel_end - chunk_mel_start;
        
        if (chunk_mel_frames < subsampling) {
            break;
        }
        
        std::vector<float> chunk_mel(n_mels * chunk_mel_frames);
        for (int m = 0; m < n_mels; m++) {
            for (int t = 0; t < chunk_mel_frames; t++) {
                chunk_mel[m * chunk_mel_frames + t] = combined_mel[m * total_mel_frames + (chunk_mel_start + t)];
            }
        }
        
        int lc = (int)round((double)left_offset / subsampling);
        int rc = (int)ceil((double)right_offset / subsampling);
        
        float * pred_out = nullptr;
        float * chunk_preenc = nullptr;
        int chunk_preenc_frames = 0;
        int n_pred = 0;
        int chunk_len_used = 0;
        
        {
            int d_model_out = 0;
            chunk_preenc_frames = sortformer_compute_preenc(
                ctx, chunk_mel.data(), n_mels, chunk_mel_frames, chunk_mel_frames,
                &chunk_preenc, &d_model_out);
            if (chunk_preenc_frames < 0) {
                return -1;
            }
            
            chunk_len_used = chunk_preenc_frames - lc - rc;
            if (chunk_len_used <= 0) {
                free(chunk_preenc);
                break;
            }
            
            int T_total = st.spkcache_len + st.fifo_len + chunk_preenc_frames;
            std::vector<float> combined(T_total * d_model);
            
            if (st.spkcache_len > 0) {
                memcpy(combined.data(), st.spkcache.data(), st.spkcache_len * d_model * sizeof(float));
            }
            if (st.fifo_len > 0) {
                memcpy(combined.data() + st.spkcache_len * d_model, st.fifo.data(), st.fifo_len * d_model * sizeof(float));
            }
            memcpy(combined.data() + (st.spkcache_len + st.fifo_len) * d_model,
                   chunk_preenc, chunk_preenc_frames * d_model * sizeof(float));
            
#ifdef SORTFORMER_USE_COREML
            if (use_coreml && T_total <= SORTFORMER_COREML_MAX_SEQ_LEN) {
                pred_out = (float *)malloc(T_total * n_spk * sizeof(float));
                int ret = sortformer_coreml_encode(ctx->coreml_ctx, combined.data(), T_total, pred_out);
                if (ret != 0) {
                    free(pred_out);
                    free(chunk_preenc);
                    return -1;
                }
                n_pred = T_total;
            } else
#endif
            {
                n_pred = sortformer_compute_streaming_prediction(ctx, combined.data(), T_total, d_model, &pred_out);
                if (n_pred < 0) {
                    free(chunk_preenc);
                    return -1;
                }
            }
            
            int pred_start = st.spkcache_len + st.fifo_len + lc;
            int pred_end = pred_start + chunk_len_used;
            
            chunk_preds.insert(chunk_preds.end(),
                               pred_out + pred_start * n_spk,
                               pred_out + pred_end * n_spk);
            total_new_frames += chunk_len_used;
        }
        
        free(chunk_preenc);
        free(pred_out);
        stt_feat = end_feat;
    }
    
    sst->mel_buffer.clear();
    sst->mel_buffer_frames = 0;
    sst->audio_overlap.clear();
    
    int n_out = std::min(total_new_frames, probs_out_max);
    if (n_out > 0) {
        memcpy(probs_out, chunk_preds.data(), n_out * n_spk * sizeof(float));
    }
    
    sst->total_frames_output += n_out;
    fprintf(stderr, "%s: flushed %d frames\n", __func__, n_out);
    
    return n_out;
}

void sortformer_stream_reset(struct sortformer_stream_state * sst) {
    if (!sst) return;
    
    sst->st = init_stream_state(sst->ctx->d_model);
    sst->audio_overlap.clear();
    sst->mel_buffer.clear();
    sst->mel_buffer_frames = 0;
    sst->total_samples_fed = 0;
    sst->total_frames_output = 0;
    sst->stt_feat = 0;
    
    fprintf(stderr, "%s: streaming state reset\n", __func__);
}

void sortformer_stream_free(struct sortformer_stream_state * sst) {
    if (!sst) return;
    delete sst;
}
