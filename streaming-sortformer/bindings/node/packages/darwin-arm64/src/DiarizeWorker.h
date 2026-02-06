#ifndef DIARIZE_WORKER_H
#define DIARIZE_WORKER_H

#include <napi.h>
#include <vector>
#include <string>

#include "sortformer.h"

// Latency preset enumeration
enum class LatencyPreset {
    Offline,    // Default: chunk_len=188, right_context=1, fifo_len=0, spkcache_update_period=188
    Low,        // Low latency: chunk_len=6, right_context=7, fifo_len=188, spkcache_update_period=144
    TwoSecond,  // 2s latency: chunk_len=15, right_context=10, fifo_len=100, spkcache_update_period=144
    ThreeSecond,// 3s latency: chunk_len=30, right_context=7, fifo_len=100, spkcache_update_period=100
    FiveSecond  // 5s latency: chunk_len=55, right_context=7, fifo_len=100, spkcache_update_period=100
};

// Diarization options passed from JavaScript
struct DiarizeOptions {
    LatencyPreset preset = LatencyPreset::Offline;
    float threshold = 0.5f;
    int median_filter = 11;
    std::string filename = "audio";  // For RTTM output
};

/**
 * AsyncWorker for non-blocking speaker diarization inference.
 * 
 * Runs sortformer_diarize() on a worker thread to avoid blocking the Node.js event loop.
 * Returns a Promise that resolves with { rttm: string, predictions: Float32Array }.
 */
class DiarizeWorker : public Napi::AsyncWorker {
public:
    /**
     * Create a new DiarizeWorker.
     * 
     * @param env The N-API environment
     * @param ctx The sortformer context (must remain valid during execution)
     * @param audio Audio samples (16kHz mono float32) - copied internally
     * @param options Diarization options including latency preset
     * @param deferred Promise deferred for async/await support
     */
    DiarizeWorker(
        Napi::Env env,
        sortformer_context* ctx,
        std::vector<float> audio,
        DiarizeOptions options,
        Napi::Promise::Deferred deferred);
    
    /**
     * Execute diarization on worker thread.
     * IMPORTANT: Cannot use any Napi objects here - runs off main thread.
     */
    void Execute() override;
    
    /**
     * Called on main thread when Execute() completes successfully.
     * Creates the result object and resolves the promise.
     */
    void OnOK() override;
    
    /**
     * Called on main thread when Execute() throws or SetError() is called.
     * Rejects the promise with the error message.
     */
    void OnError(const Napi::Error& e) override;
    
    // Helper to parse latency preset from JavaScript string
    static LatencyPreset ParsePreset(const std::string& preset);
    
    // Helper to apply preset to sortformer_params
    static void ApplyPreset(sortformer_params& params, LatencyPreset preset);

private:
    // Input data (copied from JavaScript)
    sortformer_context* ctx_;
    std::vector<float> audio_;
    DiarizeOptions options_;
    
    // Promise for async/await support
    Napi::Promise::Deferred deferred_;
    
    // Results (populated in Execute, used in OnOK)
    std::vector<float> predictions_;
    std::string rttm_;
    int n_frames_ = 0;
};

#endif // DIARIZE_WORKER_H
