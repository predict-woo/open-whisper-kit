#include "sortformer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char ** argv) {
    // Parse command line arguments
    std::string model_path;
    std::string audio_path;
    std::string output_path;
    std::string preset_str = "2s";  // default preset
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i < argc) audio_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) output_path = argv[i];
        } else if (arg == "--preset") {
            if (++i < argc) preset_str = argv[i];
        }
    }
    
    // Validate required arguments
    if (model_path.empty() || audio_path.empty()) {
        fprintf(stderr, "Usage: %s -m model.gguf -f audio.wav -o output.rttm --preset [low|2s|3s|5s]\n", argv[0]);
        return 1;
    }
    
    // Parse preset string to enum
    sortformer_stream_preset preset = SORTFORMER_PRESET_2S;
    if (preset_str == "low") {
        preset = SORTFORMER_PRESET_LOW_LATENCY;
    } else if (preset_str == "2s") {
        preset = SORTFORMER_PRESET_2S;
    } else if (preset_str == "3s") {
        preset = SORTFORMER_PRESET_3S;
    } else if (preset_str == "5s") {
        preset = SORTFORMER_PRESET_5S;
    } else {
        fprintf(stderr, "error: unknown preset '%s' (must be low, 2s, 3s, or 5s)\n", preset_str.c_str());
        return 1;
    }
    
    fprintf(stderr, "[test-streaming-api] Using preset: %s\n", preset_str.c_str());
    
    // Initialize model
    sortformer_params params = sortformer_default_params();
    sortformer_context * ctx = sortformer_init(model_path.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model from %s\n", model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[test-streaming-api] Model loaded\n");
    
    // Load audio file
    float * samples = nullptr;
    int n_samples = sortformer_load_wav(audio_path.c_str(), &samples);
    if (n_samples < 0) {
        fprintf(stderr, "error: failed to load audio from %s\n", audio_path.c_str());
        sortformer_free(ctx);
        return 1;
    }
    fprintf(stderr, "[test-streaming-api] Loaded %d samples (%.2f seconds)\n", n_samples, n_samples / 16000.0f);
    
    // Create streaming session with preset
    sortformer_stream_state * st = sortformer_stream_init(ctx, preset);
    if (!st) {
        fprintf(stderr, "error: failed to create streaming session\n");
        free(samples);
        sortformer_free(ctx);
        return 1;
    }
    fprintf(stderr, "[test-streaming-api] Streaming session initialized\n");
    
    // Feed audio in chunks and accumulate predictions
    // Use 3-second chunks (48000 samples at 16kHz)
    const int chunk_samples = 16000 * 3;
    std::vector<float> all_preds;
    
    for (int offset = 0; offset < n_samples; offset += chunk_samples) {
        int chunk_len = std::min(chunk_samples, n_samples - offset);
        
        // Allocate buffer for predictions from this chunk
        // Each frame produces 4 floats (one per speaker)
        // Max frames per chunk: chunk_len / 160 (hop size) + some margin
        int max_frames_per_chunk = (chunk_len / 160) + 50;
        float * chunk_preds = (float *)malloc(max_frames_per_chunk * 4 * sizeof(float));
        
        if (!chunk_preds) {
            fprintf(stderr, "error: failed to allocate prediction buffer\n");
            sortformer_stream_free(st);
            free(samples);
            sortformer_free(ctx);
            return 1;
        }
        
        // Feed chunk to streaming pipeline
        int n_frames = sortformer_stream_feed(st, samples + offset, chunk_len, 
                                               chunk_preds, max_frames_per_chunk);
        
        if (n_frames < 0) {
            fprintf(stderr, "error: stream feed failed at offset %d\n", offset);
            free(chunk_preds);
            break;
        }
        
        fprintf(stderr, "[test-streaming-api] Chunk at %.2fs: %d new frames\n", 
                offset / 16000.0f, n_frames);
        
        // Accumulate predictions from this chunk
        for (int i = 0; i < n_frames * 4; i++) {
            all_preds.push_back(chunk_preds[i]);
        }
        
        free(chunk_preds);
    }
    
    int total_frames = all_preds.size() / 4;
    fprintf(stderr, "[test-streaming-api] Total frames: %d\n", total_frames);
    
    if (total_frames <= 0) {
        fprintf(stderr, "error: no frames produced\n");
        sortformer_stream_free(st);
        free(samples);
        sortformer_free(ctx);
        return 1;
    }
    
    // Convert predictions to RTTM format
    char rttm_buffer[200000];  // generous buffer for RTTM output
    int rttm_len = sortformer_to_rttm(all_preds.data(), total_frames, 
                                       0.5f, 11, audio_path.c_str(), 
                                       rttm_buffer, sizeof(rttm_buffer));
    
    if (rttm_len < 0) {
        fprintf(stderr, "error: RTTM conversion failed\n");
        sortformer_stream_free(st);
        free(samples);
        sortformer_free(ctx);
        return 1;
    }
    
    fprintf(stderr, "[test-streaming-api] RTTM generated (%d bytes)\n", rttm_len);
    
    // Write RTTM to output file or stdout
    if (!output_path.empty()) {
        FILE * f = fopen(output_path.c_str(), "w");
        if (!f) {
            fprintf(stderr, "error: failed to open output file %s\n", output_path.c_str());
            sortformer_stream_free(st);
            free(samples);
            sortformer_free(ctx);
            return 1;
        }
        
        if (fwrite(rttm_buffer, 1, rttm_len, f) != (size_t)rttm_len) {
            fprintf(stderr, "error: failed to write RTTM to file\n");
            fclose(f);
            sortformer_stream_free(st);
            free(samples);
            sortformer_free(ctx);
            return 1;
        }
        
        fclose(f);
        fprintf(stderr, "[test-streaming-api] RTTM written to %s\n", output_path.c_str());
    } else {
        // Write to stdout
        fwrite(rttm_buffer, 1, rttm_len, stdout);
    }
    
    // Cleanup
    sortformer_stream_free(st);
    free(samples);
    sortformer_free(ctx);
    
    fprintf(stderr, "[test-streaming-api] Done\n");
    return 0;
}
