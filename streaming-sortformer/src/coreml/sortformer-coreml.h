// CoreML bridge for SortFormer head (conformer + transformer + prediction)
//
// The head-only model takes pre-encoder embeddings and returns speaker predictions.
// GGML handles mel spectrogram and pre-encoder computation.

#ifndef SORTFORMER_COREML_H
#define SORTFORMER_COREML_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SORTFORMER_COREML_MAX_SEQ_LEN  400
#define SORTFORMER_COREML_D_MODEL      512
#define SORTFORMER_COREML_N_SPEAKERS   4

struct sortformer_coreml_context;

struct sortformer_coreml_context * sortformer_coreml_init(const char * path_model);

void sortformer_coreml_free(struct sortformer_coreml_context * ctx);

// Run head inference (conformer + transformer + prediction)
//
// Inputs:
//   pre_encoder_embs: [T, d_model] pre-encoder embeddings (row-major)
//   seq_len:          valid sequence length T
//
// Outputs:
//   preds_out:        [T, n_speakers] speaker probabilities (caller allocates)
//
// Returns 0 on success, -1 on failure
int sortformer_coreml_encode(
    struct sortformer_coreml_context * ctx,
    const float * pre_encoder_embs,
    int32_t       seq_len,
    float       * preds_out
);

#ifdef __cplusplus
}
#endif

#endif // SORTFORMER_COREML_H
