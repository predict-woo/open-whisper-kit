#ifndef SORTFORMER_H
#define SORTFORMER_H

#ifdef __cplusplus
extern "C" {
#endif

struct sortformer_context;

struct sortformer_params {
    int   chunk_len;              // default: 188
    int   right_context;          // default: 1
    int   fifo_len;               // default: 0
    int   spkcache_len;           // default: 188
    int   spkcache_update_period; // default: 188
    float threshold;              // default: 0.5
    int   median_filter;          // default: 11
    int   n_threads;              // default: 4
    int   chunk_left_context;     // default: 1
};

struct sortformer_params sortformer_default_params(void);

struct sortformer_context * sortformer_init(const char * model_path, struct sortformer_params params);

void sortformer_free(struct sortformer_context * ctx);

// Load WAV file (16 kHz mono 16-bit PCM).
// Returns number of samples, or -1 on error.
// Caller must free *samples_out with free().
int sortformer_load_wav(const char * path, float ** samples_out);

// Compute mel spectrogram from audio samples.
// Returns number of time frames T (padded to multiple of 16), or -1 on error.
// On success, *mel_out points to (n_mels * T) float32 values in row-major order.
// *n_mels_out is set to the number of mel bins (128).
// *seq_len_out is set to the valid (unpadded) number of time frames.
// Caller must free *mel_out with free().
int sortformer_compute_mel(
    struct sortformer_context * ctx,
    const float * samples,
    int           n_samples,
    float      ** mel_out,
    int         * n_mels_out,
    int         * seq_len_out);

// Compute pre-encoder output from mel spectrogram.
// mel_data: (n_mels, n_mel_frames) row-major float32 (full padded mel)
// seq_len: valid (unpadded) number of mel frames to use
// Returns number of output time frames T_out, or -1 on error.
// On success, *preenc_out points to (T_out * d_model) float32 values in row-major order.
// Caller must free *preenc_out with free().
int sortformer_compute_preenc(
    struct sortformer_context * ctx,
    const float * mel_data,
    int           n_mels,
    int           n_mel_frames,
    int           seq_len,
    float      ** preenc_out,
    int         * d_model_out);

// Compute conformer encoder output from pre-encoder output.
// preenc_data: (T, d_model) row-major float32
// target_layer: run layers 0..target_layer, output that layer's result.
//               Use 16 (last layer) for full conformer output.
// Returns T on success (output shape is T × d_model), or -1 on error.
// Caller must free *conf_out with free().
int sortformer_compute_conformer(
    struct sortformer_context * ctx,
    const float * preenc_data,
    int           T,
    int           d_model,
    int           target_layer,
    float      ** conf_out);

// Compute projection from conformer output (512 → 192).
// conf_data: (T, d_model_in) row-major float32
// Returns T on success, -1 on error.
// Caller must free *proj_out with free().
int sortformer_compute_projection(
    struct sortformer_context * ctx,
    const float * conf_data,
    int           T,
    int           d_model_in,
    float      ** proj_out,
    int         * d_model_out_ptr);

// Compute transformer encoder output from projection output.
// proj_data: (T, d_model) row-major float32
// target_layer: run layers 0..target_layer. Use 17 for full output.
// Returns T on success, -1 on error.
// Caller must free *trans_out with free().
int sortformer_compute_transformer(
    struct sortformer_context * ctx,
    const float * proj_data,
    int           T,
    int           d_model,
    int           target_layer,
    float      ** trans_out);

// Compute prediction head from transformer output.
// trans_data: (T, d_model) row-major float32
// Returns T on success (output shape is T × 4), -1 on error.
// Caller must free *pred_out with free().
int sortformer_compute_prediction(
    struct sortformer_context * ctx,
    const float * trans_data,
    int           T,
    int           d_model,
    float      ** pred_out);

// Run diarization on audio samples (16 kHz mono float32).
// Returns number of frames written to probs_out, or -1 on error.
int sortformer_diarize(
    struct sortformer_context * ctx,
    const float * audio_samples,
    int           n_samples,
    float       * probs_out,
    int           n_frames_max);

// Convert frame-level probabilities to RTTM format.
// Returns number of bytes written to rttm_out, or -1 on error.
int sortformer_to_rttm(
    const float * probs,
    int           n_frames,
    float         threshold,
    int           median_filter,
    const char  * filename,
    char        * rttm_out,
    int           rttm_out_size);

// ============================================================================
// Streaming API
// ============================================================================

// Opaque streaming state handle
struct sortformer_stream_state;

// Latency preset enumeration
enum sortformer_stream_preset {
    SORTFORMER_PRESET_LOW_LATENCY = 0,  // ~60ms chunks, lowest latency
    SORTFORMER_PRESET_2S          = 1,  // ~2 second latency, balanced
    SORTFORMER_PRESET_3S          = 2,  // ~3 second latency
    SORTFORMER_PRESET_5S          = 3,  // ~5 second latency, best accuracy
};

// Streaming configuration
struct sortformer_stream_params {
    int   chunk_len;              // frames per chunk
    int   right_context;          // right context frames
    int   left_context;           // left context frames
    int   fifo_len;               // FIFO buffer length
    int   spkcache_len;           // speaker cache length
    int   spkcache_update_period; // update period
};

// Preset parameter values (from NeMo/HuggingFace model card):
//
// | Preset      | chunk_len | right_ctx | left_ctx | fifo_len | spkcache_len | update_period |
// |-------------|-----------|-----------|----------|----------|--------------|---------------|
// | LOW_LATENCY |     6     |     7     |    1     |   188    |     188      |      144      |
// | 2S          |    15     |    10     |    1     |   100    |     188      |      144      |
// | 3S          |    30     |     7     |    1     |   100    |     188      |      100      |
// | 5S          |    55     |     7     |    1     |   100    |     188      |      100      |

// Get preset parameters
struct sortformer_stream_params sortformer_stream_preset_params(enum sortformer_stream_preset preset);

// Initialize streaming session with preset
// Returns NULL on error
struct sortformer_stream_state * sortformer_stream_init(
    struct sortformer_context * ctx,
    enum sortformer_stream_preset preset);

// Initialize streaming session with custom params
// Returns NULL on error
struct sortformer_stream_state * sortformer_stream_init_with_params(
    struct sortformer_context * ctx,
    struct sortformer_stream_params params);

// Feed audio samples, get predictions for this chunk only
// Returns number of NEW frames written to probs_out, or -1 on error
// probs_out must have space for at least (n_samples / 1280 + 10) * 4 floats
int sortformer_stream_feed(
    struct sortformer_stream_state * st,
    const float * audio_samples,
    int           n_samples,
    float       * probs_out,
    int           probs_out_max);

// Flush remaining buffered audio at end of stream
// Returns number of frames written to probs_out, or -1 on error
int sortformer_stream_flush(
    struct sortformer_stream_state * st,
    float       * probs_out,
    int           probs_out_max);

// Reset streaming state (for new audio stream, keeps model loaded)
void sortformer_stream_reset(struct sortformer_stream_state * st);

// Free streaming state
void sortformer_stream_free(struct sortformer_stream_state * st);

#ifdef __cplusplus
}
#endif

#endif // SORTFORMER_H
