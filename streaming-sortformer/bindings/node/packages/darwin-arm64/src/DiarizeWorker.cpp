#include "DiarizeWorker.h"
#include <cmath>
#include <cstring>

// Number of speakers output by the model
static constexpr int NUM_SPEAKERS = 4;

// RTTM buffer size: ~1KB per minute of audio, plus safety margin
// For 16kHz audio with 160-sample hop, 1 minute = 6000 frames
// Estimate: 100 bytes per RTTM line, ~60 lines per minute per speaker = 24KB/min
// Use 32KB per minute as safe estimate
static constexpr int RTTM_BYTES_PER_MINUTE = 32 * 1024;

DiarizeWorker::DiarizeWorker(
    Napi::Env env,
    sortformer_context* ctx,
    std::vector<float> audio,
    DiarizeOptions options,
    Napi::Promise::Deferred deferred)
    : Napi::AsyncWorker(env)
    , ctx_(ctx)
    , audio_(std::move(audio))
    , options_(std::move(options))
    , deferred_(deferred)
{
}

void DiarizeWorker::Execute() {
    // IMPORTANT: This runs on a worker thread - NO Napi calls allowed!
    
    if (ctx_ == nullptr) {
        SetError("Model context is null or has been closed");
        return;
    }
    
    if (audio_.empty()) {
        SetError("Audio data is empty");
        return;
    }
    
    // Calculate expected number of frames
    // Formula: n_frames = (n_samples - 400) / 160 + 1, then subsampled by 8
    // Simplified: n_frames ≈ n_samples / (160 * 8) = n_samples / 1280
    // Add safety margin for padding
    int n_samples = static_cast<int>(audio_.size());
    int n_frames_max = (n_samples / 1280) + 100;  // Add margin for padding
    
    // Allocate output buffer for predictions (n_frames * 4 speakers)
    predictions_.resize(n_frames_max * NUM_SPEAKERS);
    
    // Run diarization
    n_frames_ = sortformer_diarize(
        ctx_,
        audio_.data(),
        n_samples,
        predictions_.data(),
        n_frames_max);
    
    if (n_frames_ < 0) {
        SetError("Diarization failed");
        return;
    }
    
    // Trim predictions to actual frame count
    predictions_.resize(n_frames_ * NUM_SPEAKERS);
    
    // Calculate RTTM buffer size based on audio duration
    // Audio duration in minutes = n_samples / (16000 * 60)
    float duration_minutes = static_cast<float>(n_samples) / (16000.0f * 60.0f);
    int rttm_size = static_cast<int>(std::ceil(duration_minutes + 1.0f) * RTTM_BYTES_PER_MINUTE);
    rttm_size = std::max(rttm_size, 4096);  // Minimum 4KB
    
    // Allocate RTTM buffer
    std::vector<char> rttm_buffer(rttm_size);
    
    // Convert predictions to RTTM format
    int rttm_bytes = sortformer_to_rttm(
        predictions_.data(),
        n_frames_,
        options_.threshold,
        options_.median_filter,
        options_.filename.c_str(),
        rttm_buffer.data(),
        rttm_size);
    
    if (rttm_bytes < 0) {
        SetError("Failed to convert predictions to RTTM format");
        return;
    }
    
    // Store RTTM string (null-terminated by sortformer_to_rttm)
    rttm_ = std::string(rttm_buffer.data(), rttm_bytes);
}

void DiarizeWorker::OnOK() {
    Napi::Env env = Env();
    Napi::HandleScope scope(env);
    
    // Create result object
    Napi::Object result = Napi::Object::New(env);
    
    // Add RTTM string
    result.Set("rttm", Napi::String::New(env, rttm_));
    
    // Create Float32Array for predictions
    // Copy data to a new ArrayBuffer owned by JavaScript
    Napi::ArrayBuffer buffer = Napi::ArrayBuffer::New(
        env, 
        predictions_.size() * sizeof(float));
    
    std::memcpy(buffer.Data(), predictions_.data(), predictions_.size() * sizeof(float));
    
    Napi::Float32Array predictions = Napi::Float32Array::New(
        env,
        predictions_.size(),
        buffer,
        0);
    
    result.Set("predictions", predictions);
    
    // Add frame count for convenience
    result.Set("frameCount", Napi::Number::New(env, n_frames_));
    
    // Add speaker count
    result.Set("speakerCount", Napi::Number::New(env, NUM_SPEAKERS));
    
    // Resolve the promise with the result
    deferred_.Resolve(result);
}

void DiarizeWorker::OnError(const Napi::Error& e) {
    // Reject the promise with the error
    deferred_.Reject(e.Value());
}

LatencyPreset DiarizeWorker::ParsePreset(const std::string& preset) {
    if (preset == "offline" || preset == "default") {
        return LatencyPreset::Offline;
    } else if (preset == "low") {
        return LatencyPreset::Low;
    } else if (preset == "2s") {
        return LatencyPreset::TwoSecond;
    } else if (preset == "3s") {
        return LatencyPreset::ThreeSecond;
    } else if (preset == "5s") {
        return LatencyPreset::FiveSecond;
    }
    // Default to offline for unknown presets
    return LatencyPreset::Offline;
}

void DiarizeWorker::ApplyPreset(sortformer_params& params, LatencyPreset preset) {
    switch (preset) {
        case LatencyPreset::Offline:
            // Default/offline: chunk_len=188, right_context=1, fifo_len=0, spkcache_update_period=188
            params.chunk_len = 188;
            params.right_context = 1;
            params.fifo_len = 0;
            params.spkcache_update_period = 188;
            break;
            
        case LatencyPreset::Low:
            // Low latency: chunk_len=6, right_context=7, fifo_len=188, spkcache_update_period=144
            params.chunk_len = 6;
            params.right_context = 7;
            params.fifo_len = 188;
            params.spkcache_update_period = 144;
            break;
            
        case LatencyPreset::TwoSecond:
            // 2s latency: chunk_len=15, right_context=10, fifo_len=100, spkcache_update_period=144
            params.chunk_len = 15;
            params.right_context = 10;
            params.fifo_len = 100;
            params.spkcache_update_period = 144;
            break;
            
        case LatencyPreset::ThreeSecond:
            // 3s latency: chunk_len=30, right_context=7, fifo_len=100, spkcache_update_period=100
            params.chunk_len = 30;
            params.right_context = 7;
            params.fifo_len = 100;
            params.spkcache_update_period = 100;
            break;
            
        case LatencyPreset::FiveSecond:
            // 5s latency: chunk_len=55, right_context=7, fifo_len=100, spkcache_update_period=100
            params.chunk_len = 55;
            params.right_context = 7;
            params.fifo_len = 100;
            params.spkcache_update_period = 100;
            break;
    }
}
