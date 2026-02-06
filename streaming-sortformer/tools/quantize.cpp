#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage() {
    std::fprintf(stderr, "Usage: sortformer-quantize <input.gguf> <output.gguf> <type>\n");
    std::fprintf(stderr, "Types: q4_k, q5_k, q8_0\n");
}

static bool contains(const std::string & name, const char * needle) {
    return name.find(needle) != std::string::npos;
}

static bool is_skip_name(const std::string & name) {
    if (contains(name, ".bias")) {
        return true;
    }
    if (contains(name, "norm")) {
        return true;
    }
    if (contains(name, ".pos_bias_u") || contains(name, ".pos_bias_v")) {
        return true;
    }
    if (contains(name, "preprocessor.featurizer.")) {
        return true;
    }
    if (contains(name, "encoder.pre_encode.")) {
        return true;
    }
    if (contains(name, "sortformer_modules.encoder_proj.")) {
        return true;
    }
    if (contains(name, "sortformer_modules.first_hidden_to_hidden.")) {
        return true;
    }
    if (contains(name, "sortformer_modules.single_hidden_to_spks.")) {
        return true;
    }
    if (contains(name, ".conv.depthwise_conv.")) {
        return true;
    }
    return false;
}

static bool matches_quant_pattern(const std::string & name) {
    if (contains(name, "encoder.layers.") &&
        contains(name, ".feed_forward") &&
        contains(name, ".linear") &&
        contains(name, ".weight")) {
        return true;
    }
    if (contains(name, "encoder.layers.") &&
        contains(name, ".self_attn.linear_") &&
        contains(name, ".weight")) {
        return true;
    }
    if (contains(name, "encoder.layers.") &&
        contains(name, ".conv.pointwise_conv") &&
        contains(name, ".weight")) {
        return true;
    }
    if (contains(name, "transformer_encoder.layers.") &&
        contains(name, ".first_sub_layer.") &&
        contains(name, ".weight")) {
        return true;
    }
    if (contains(name, "transformer_encoder.layers.") &&
        contains(name, ".second_sub_layer.") &&
        contains(name, ".weight")) {
        return true;
    }
    return false;
}

static bool should_quantize(const std::string & name) {
    if (!contains(name, ".weight")) {
        return false;
    }
    if (is_skip_name(name)) {
        return false;
    }
    return matches_quant_pattern(name);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage();
        return 1;
    }

    const char * input_path = argv[1];
    const char * output_path = argv[2];
    const char * type_name = argv[3];

    enum ggml_type qtype;
    if (std::strcmp(type_name, "q4_k") == 0) {
        qtype = GGML_TYPE_Q4_K;
    } else if (std::strcmp(type_name, "q5_k") == 0) {
        qtype = GGML_TYPE_Q5_K;
    } else if (std::strcmp(type_name, "q8_0") == 0) {
        qtype = GGML_TYPE_Q8_0;
    } else {
        print_usage();
        return 1;
    }

    struct ggml_context * in_ggml_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &in_ggml_ctx,
    };

    struct gguf_context * in_ctx = gguf_init_from_file(input_path, params);
    if (!in_ctx || !in_ggml_ctx) {
        std::fprintf(stderr, "failed to load gguf: %s\n", input_path);
        if (in_ctx) {
            gguf_free(in_ctx);
        }
        return 1;
    }

    struct gguf_context * out_ctx = gguf_init_empty();
    if (!out_ctx) {
        std::fprintf(stderr, "failed to create output gguf context\n");
        gguf_free(in_ctx);
        ggml_free(in_ggml_ctx);
        return 1;
    }

    gguf_set_kv(out_ctx, in_ctx);

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    std::vector<std::vector<uint8_t>> quant_buffers;
    quant_buffers.reserve(n_tensors);

    int64_t quantized_count = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * tensor_name = gguf_get_tensor_name(in_ctx, i);
        const enum ggml_type in_type = gguf_get_tensor_type(in_ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(in_ggml_ctx, tensor_name);
        if (!tensor || !tensor->data) {
            std::fprintf(stderr, "missing tensor data: %s\n", tensor_name ? tensor_name : "(null)");
            gguf_free(out_ctx);
            gguf_free(in_ctx);
            ggml_free(in_ggml_ctx);
            return 1;
        }

        gguf_add_tensor(out_ctx, tensor);

        std::string name_str(tensor_name);
        if (!should_quantize(name_str)) {
            continue;
        }

        if (in_type != GGML_TYPE_F16 && in_type != GGML_TYPE_F32) {
            continue;
        }

        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        const int64_t blck = ggml_blck_size(qtype);
        if (n_per_row <= 0 || nrows <= 0 || n_per_row % blck != 0) {
            continue;
        }

        std::vector<float> f32_data;
        f32_data.resize(static_cast<size_t>(nrows) * static_cast<size_t>(n_per_row));

        const char * src_base = static_cast<const char *>(tensor->data);
        if (in_type == GGML_TYPE_F16) {
            for (int64_t row = 0; row < nrows; ++row) {
                const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(src_base + row * tensor->nb[1]);
                float * dst = f32_data.data() + static_cast<size_t>(row) * static_cast<size_t>(n_per_row);
                ggml_fp16_to_fp32_row(src, dst, n_per_row);
            }
        } else {
            for (int64_t row = 0; row < nrows; ++row) {
                const void * src = src_base + row * tensor->nb[1];
                float * dst = f32_data.data() + static_cast<size_t>(row) * static_cast<size_t>(n_per_row);
                std::memcpy(dst, src, static_cast<size_t>(n_per_row) * sizeof(float));
            }
        }

        const size_t quant_size = static_cast<size_t>(nrows) * static_cast<size_t>(n_per_row) /
            static_cast<size_t>(blck) * ggml_type_size(qtype);
        quant_buffers.emplace_back(quant_size);
        uint8_t * quant_buf = quant_buffers.back().data();

        const size_t written = ggml_quantize_chunk(
            qtype,
            f32_data.data(),
            quant_buf,
            0,
            nrows,
            n_per_row,
            nullptr);

        if (written != quant_size) {
            std::fprintf(stderr, "warning: quantized size mismatch for %s (got %zu, expected %zu)\n",
                         tensor_name, written, quant_size);
        }

        gguf_set_tensor_type(out_ctx, tensor_name, qtype);
        gguf_set_tensor_data(out_ctx, tensor_name, quant_buf);
        ++quantized_count;
    }

    const bool ok = gguf_write_to_file(out_ctx, output_path, false);
    if (!ok) {
        std::fprintf(stderr, "failed to write gguf: %s\n", output_path);
    } else {
        std::fprintf(stderr, "quantized %lld tensors\n", static_cast<long long>(quantized_count));
    }

    ggml_quantize_free();
    gguf_free(out_ctx);
    gguf_free(in_ctx);
    ggml_free(in_ggml_ctx);

    return ok ? 0 : 1;
}
