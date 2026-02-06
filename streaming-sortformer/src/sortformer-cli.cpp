#include "sortformer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

struct cli_params {
    std::string model_path   = "";
    std::string audio_file   = "";
    std::string rttm_output  = "";
    bool        dump_probs   = false;
    float       threshold    = 0.5f;
    int         median_filter = 11;
    int         chunk_len    = 188;
    int         right_context = 1;
    int         fifo_len     = 0;
    int         spkcache_len = 188;
    int         spkcache_update_period = 188;
    int         chunk_left_context = 1;
    int         n_threads    = 4;
    bool        dump_mel     = false;
    bool        dump_preenc  = false;
    int         dump_conformer  = -1;
    bool        dump_projection = false;
    int         dump_transformer = -1;
    bool        dump_prediction = false;
    bool        streaming    = false;
    bool        low_latency  = false;
    bool        high_latency = false;
    bool        latency_2s   = false;
    bool        latency_3s   = false;
    bool        latency_5s   = false;
    bool        print_help   = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Speaker diarization using Sortformer (ggml)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,  --help              show this help message and exit\n");
    fprintf(stderr, "  -m,  --model PATH        path to GGUF model file (required)\n");
    fprintf(stderr, "  -f,  --file PATH         path to input audio file (16kHz mono WAV)\n");
    fprintf(stderr, "  -o,  --rttm PATH         path to output RTTM file\n");
    fprintf(stderr, "       --probs             dump frame-level speaker probabilities\n");
    fprintf(stderr, "       --threshold FLOAT   speaker activity threshold (default: 0.5)\n");
    fprintf(stderr, "       --median-filter INT  median filter window size (default: 11)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming presets (applied before individual overrides):\n");
    fprintf(stderr, "       --low-latency         low-latency config (chunk=6, rc=7, fifo=188, update=144)\n");
    fprintf(stderr, "       --high-latency        very-high-latency config (chunk=340, rc=40, fifo=40, update=300)\n");
    fprintf(stderr, "       --2s-latency          2-second latency (chunk=15, rc=10, fifo=100, update=144)\n");
    fprintf(stderr, "       --3s-latency          3-second latency (chunk=30, rc=7, fifo=100, update=100)\n");
    fprintf(stderr, "       --5s-latency          5-second latency (chunk=55, rc=7, fifo=100, update=100)\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Streaming parameters:\n");
    fprintf(stderr, "       --chunk-len INT      chunk length in frames (default: 188)\n");
    fprintf(stderr, "       --right-context INT  right context frames (default: 1)\n");
    fprintf(stderr, "       --chunk-left-context INT  left context frames (default: 1)\n");
    fprintf(stderr, "       --fifo-len INT       FIFO buffer length (default: 0)\n");
    fprintf(stderr, "       --spkcache-len INT   speaker cache length (default: 188)\n");
    fprintf(stderr, "       --spkcache-update-period INT  speaker cache update period (default: 188)\n");
    fprintf(stderr, "       --threads INT        number of threads (default: 4)\n");
    fprintf(stderr, "       --dump-mel           dump mel spectrogram\n");
    fprintf(stderr, "       --dump-preenc        dump pre-encoder output\n");
    fprintf(stderr, "       --dump-conformer N   dump conformer layer N output\n");
    fprintf(stderr, "       --dump-projection    dump projection output\n");
    fprintf(stderr, "       --dump-transformer N dump transformer layer N output\n");
    fprintf(stderr, "       --dump-prediction    dump prediction output\n");
    fprintf(stderr, "       --streaming          use streaming inference pipeline\n");
    fprintf(stderr, "\n");
}

// Get peak RSS memory usage in MB from /proc/self/status (Linux only)
// Returns 0 if unavailable (non-Linux or file can't be read)
static float get_peak_rss_mb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return 0.0f;
    
    char line[256];
    float peak_kb = 0.0f;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "VmHWM: %f kB", &peak_kb) == 1) {
            break;
        }
    }
    fclose(f);
    
    return peak_kb / 1024.0f;  // Convert kB to MB
}

static bool parse_args(int argc, char ** argv, cli_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            params.print_help = true;
            return true;
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.model_path = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.audio_file = argv[i];
        } else if (arg == "-o" || arg == "--rttm") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.rttm_output = argv[i];
        } else if (arg == "--probs") {
            params.dump_probs = true;
        } else if (arg == "--threshold") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.threshold = std::atof(argv[i]);
        } else if (arg == "--median-filter") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.median_filter = std::atoi(argv[i]);
        } else if (arg == "--chunk-len") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.chunk_len = std::atoi(argv[i]);
        } else if (arg == "--right-context") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.right_context = std::atoi(argv[i]);
        } else if (arg == "--fifo-len") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.fifo_len = std::atoi(argv[i]);
        } else if (arg == "--spkcache-len") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.spkcache_len = std::atoi(argv[i]);
        } else if (arg == "--spkcache-update-period") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.spkcache_update_period = std::atoi(argv[i]);
        } else if (arg == "--chunk-left-context") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.chunk_left_context = std::atoi(argv[i]);
        } else if (arg == "--low-latency") {
            params.low_latency = true;
            params.chunk_len = 6;
            params.right_context = 7;
            params.fifo_len = 188;
            params.spkcache_update_period = 144;
        } else if (arg == "--high-latency") {
            params.high_latency = true;
            params.chunk_len = 340;
            params.right_context = 40;
            params.fifo_len = 40;
            params.spkcache_update_period = 300;
        } else if (arg == "--2s-latency") {
            params.latency_2s = true;
            params.chunk_len = 15;
            params.right_context = 10;
            params.fifo_len = 100;
            params.spkcache_update_period = 144;
        } else if (arg == "--3s-latency") {
            params.latency_3s = true;
            params.chunk_len = 30;
            params.right_context = 7;
            params.fifo_len = 100;
            params.spkcache_update_period = 100;
        } else if (arg == "--5s-latency") {
            params.latency_5s = true;
            params.chunk_len = 55;
            params.right_context = 7;
            params.fifo_len = 100;
            params.spkcache_update_period = 100;
        } else if (arg == "--threads") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.n_threads = std::atoi(argv[i]);
        } else if (arg == "--dump-mel") {
            params.dump_mel = true;
        } else if (arg == "--dump-preenc") {
            params.dump_preenc = true;
        } else if (arg == "--dump-conformer") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.dump_conformer = std::atoi(argv[i]);
        } else if (arg == "--dump-projection") {
            params.dump_projection = true;
        } else if (arg == "--dump-transformer") {
            if (++i >= argc) { fprintf(stderr, "error: missing argument for %s\n", arg.c_str()); return false; }
            params.dump_transformer = std::atoi(argv[i]);
        } else if (arg == "--dump-prediction") {
            params.dump_prediction = true;
        } else if (arg == "--streaming") {
            params.streaming = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    cli_params params;

    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    if (params.print_help || argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    if (params.model_path.empty()) {
        fprintf(stderr, "error: no model path specified\n");
        print_usage(argv[0]);
        return 1;
    }

    if (params.audio_file.empty()) {
        fprintf(stderr, "error: no audio file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    // Build sortformer params from CLI
    struct sortformer_params sparams = sortformer_default_params();
    sparams.chunk_len              = params.chunk_len;
    sparams.right_context          = params.right_context;
    sparams.chunk_left_context     = params.chunk_left_context;
    sparams.fifo_len               = params.fifo_len;
    sparams.spkcache_len           = params.spkcache_len;
    sparams.spkcache_update_period = params.spkcache_update_period;
    sparams.threshold              = params.threshold;
    sparams.median_filter          = params.median_filter;
    sparams.n_threads              = params.n_threads;

    fprintf(stderr, "sortformer-diarize: model     = %s\n", params.model_path.c_str());
    fprintf(stderr, "sortformer-diarize: audio     = %s\n", params.audio_file.c_str());
    fprintf(stderr, "sortformer-diarize: threads   = %d\n", sparams.n_threads);
    fprintf(stderr, "sortformer-diarize: threshold = %.2f\n", sparams.threshold);

    struct sortformer_context * ctx = sortformer_init(params.model_path.c_str(), sparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to initialize sortformer\n");
        return 1;
    }

    // --dump-mel: compute mel spectrogram and write raw float32 binary
    if (params.dump_mel) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: mel shape = (%d, %d)\n", n_mels, n_frames);

        // Write raw float32 binary to cpp_mel.raw
        const char * out_path = "cpp_mel.raw";
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(mel);
            sortformer_free(ctx);
            return 1;
        }

        // Write in row-major order: (n_mels, n_frames)
        // The data is already in row-major: mel[m * n_frames + t]
        size_t n_written = fwrite(mel, sizeof(float), n_mels * n_frames, fout);
        fclose(fout);

        if ((int)n_written != n_mels * n_frames) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_mels * n_frames);
            free(mel);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote mel to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_mels, n_frames, n_mels * n_frames,
                (size_t)(n_mels * n_frames) * sizeof(float));

        free(mel);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-preenc: compute pre-encoder output and write raw float32 binary
    if (params.dump_preenc) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: mel shape = (%d, %d), seq_len = %d\n",
                n_mels, n_frames, seq_len);

        float * preenc = nullptr;
        int d_model = 0;
        int n_preenc_frames = sortformer_compute_preenc(ctx, mel, n_mels, n_frames, seq_len,
                                                         &preenc, &d_model);
        free(mel);

        if (n_preenc_frames < 0) {
            fprintf(stderr, "error: pre-encoder computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: preenc shape = (%d, %d)\n",
                n_preenc_frames, d_model);

        const char * out_path = "cpp_preenc.raw";
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(preenc);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(preenc, sizeof(float), n_preenc_frames * d_model, fout);
        fclose(fout);

        if ((int)n_written != n_preenc_frames * d_model) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_preenc_frames * d_model);
            free(preenc);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote preenc to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_preenc_frames, d_model, n_preenc_frames * d_model,
                (size_t)(n_preenc_frames * d_model) * sizeof(float));

        free(preenc);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-conformer N: compute conformer layer N and write raw float32 binary
    if (params.dump_conformer >= 0) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * preenc = nullptr;
        int d_model = 0;
        int n_preenc_frames = sortformer_compute_preenc(ctx, mel, n_mels, n_frames, seq_len,
                                                         &preenc, &d_model);
        free(mel);

        if (n_preenc_frames < 0) {
            fprintf(stderr, "error: pre-encoder computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: preenc shape = (%d, %d)\n",
                n_preenc_frames, d_model);

        float * conf = nullptr;
        int n_conf_frames = sortformer_compute_conformer(ctx, preenc, n_preenc_frames,
                                                          d_model, params.dump_conformer, &conf);
        free(preenc);

        if (n_conf_frames < 0) {
            fprintf(stderr, "error: conformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: conformer layer %d shape = (%d, %d)\n",
                params.dump_conformer, n_conf_frames, d_model);

        char out_path[64];
        snprintf(out_path, sizeof(out_path), "cpp_conf%d.raw", params.dump_conformer);
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(conf);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(conf, sizeof(float), n_conf_frames * d_model, fout);
        fclose(fout);

        if ((int)n_written != n_conf_frames * d_model) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_conf_frames * d_model);
            free(conf);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote conformer to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_conf_frames, d_model, n_conf_frames * d_model,
                (size_t)(n_conf_frames * d_model) * sizeof(float));

        free(conf);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-projection: compute projection and write raw float32 binary
    if (params.dump_projection) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * preenc = nullptr;
        int d_model = 0;
        int n_preenc_frames = sortformer_compute_preenc(ctx, mel, n_mels, n_frames, seq_len,
                                                         &preenc, &d_model);
        free(mel);

        if (n_preenc_frames < 0) {
            fprintf(stderr, "error: pre-encoder computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * conf = nullptr;
        int n_conf_frames = sortformer_compute_conformer(ctx, preenc, n_preenc_frames,
                                                          d_model, 16, &conf);
        free(preenc);

        if (n_conf_frames < 0) {
            fprintf(stderr, "error: conformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * proj = nullptr;
        int d_proj = 0;
        int n_proj_frames = sortformer_compute_projection(ctx, conf, n_conf_frames,
                                                           d_model, &proj, &d_proj);
        free(conf);

        if (n_proj_frames < 0) {
            fprintf(stderr, "error: projection computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: projection shape = (%d, %d)\n",
                n_proj_frames, d_proj);

        const char * out_path = "cpp_proj.raw";
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(proj);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(proj, sizeof(float), n_proj_frames * d_proj, fout);
        fclose(fout);

        if ((int)n_written != n_proj_frames * d_proj) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_proj_frames * d_proj);
            free(proj);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote projection to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_proj_frames, d_proj, n_proj_frames * d_proj,
                (size_t)(n_proj_frames * d_proj) * sizeof(float));

        free(proj);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-transformer N: compute transformer layer N and write raw float32 binary
    if (params.dump_transformer >= 0) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * preenc = nullptr;
        int d_model = 0;
        int n_preenc_frames = sortformer_compute_preenc(ctx, mel, n_mels, n_frames, seq_len,
                                                         &preenc, &d_model);
        free(mel);

        if (n_preenc_frames < 0) {
            fprintf(stderr, "error: pre-encoder computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * conf = nullptr;
        int n_conf_frames = sortformer_compute_conformer(ctx, preenc, n_preenc_frames,
                                                          d_model, 16, &conf);
        free(preenc);

        if (n_conf_frames < 0) {
            fprintf(stderr, "error: conformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * proj = nullptr;
        int d_proj = 0;
        int n_proj_frames = sortformer_compute_projection(ctx, conf, n_conf_frames,
                                                           d_model, &proj, &d_proj);
        free(conf);

        if (n_proj_frames < 0) {
            fprintf(stderr, "error: projection computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * trans = nullptr;
        int n_trans_frames = sortformer_compute_transformer(ctx, proj, n_proj_frames,
                                                             d_proj, params.dump_transformer, &trans);
        free(proj);

        if (n_trans_frames < 0) {
            fprintf(stderr, "error: transformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: transformer layer %d shape = (%d, %d)\n",
                params.dump_transformer, n_trans_frames, d_proj);

        char out_path[64];
        snprintf(out_path, sizeof(out_path), "cpp_trans%d.raw", params.dump_transformer);
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(trans);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(trans, sizeof(float), n_trans_frames * d_proj, fout);
        fclose(fout);

        if ((int)n_written != n_trans_frames * d_proj) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_trans_frames * d_proj);
            free(trans);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote transformer to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_trans_frames, d_proj, n_trans_frames * d_proj,
                (size_t)(n_trans_frames * d_proj) * sizeof(float));

        free(trans);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-prediction --streaming: run streaming pipeline via sortformer_diarize()
    if (params.dump_prediction && params.streaming) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        const int max_frames = 4096;
        const int n_spk = 4;
        float * probs = (float *)malloc(max_frames * n_spk * sizeof(float));
        if (!probs) {
            fprintf(stderr, "error: allocation failed\n");
            free(audio);
            sortformer_free(ctx);
            return 1;
        }

        int n_frames = sortformer_diarize(ctx, audio, n_samples, probs, max_frames);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: streaming diarization failed\n");
            free(probs);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: streaming prediction shape = (%d, %d)\n",
                n_frames, n_spk);

        const char * out_path = "cpp_pred.raw";
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(probs);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(probs, sizeof(float), n_frames * n_spk, fout);
        fclose(fout);

        if ((int)n_written != n_frames * n_spk) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_frames * n_spk);
            free(probs);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote streaming prediction to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_frames, n_spk, n_frames * n_spk,
                (size_t)(n_frames * n_spk) * sizeof(float));

        free(probs);
        sortformer_free(ctx);
        return 0;
    }

    // --dump-prediction: compute full offline pipeline and write raw float32 binary
    if (params.dump_prediction) {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        float * mel = nullptr;
        int n_mels = 0;
        int seq_len = 0;
        int n_frames = sortformer_compute_mel(ctx, audio, n_samples, &mel, &n_mels, &seq_len);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: mel computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * preenc = nullptr;
        int d_model = 0;
        int n_preenc_frames = sortformer_compute_preenc(ctx, mel, n_mels, n_frames, seq_len,
                                                         &preenc, &d_model);
        free(mel);

        if (n_preenc_frames < 0) {
            fprintf(stderr, "error: pre-encoder computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * conf = nullptr;
        int n_conf_frames = sortformer_compute_conformer(ctx, preenc, n_preenc_frames,
                                                          d_model, 16, &conf);
        free(preenc);

        if (n_conf_frames < 0) {
            fprintf(stderr, "error: conformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * proj = nullptr;
        int d_proj = 0;
        int n_proj_frames = sortformer_compute_projection(ctx, conf, n_conf_frames,
                                                           d_model, &proj, &d_proj);
        free(conf);

        if (n_proj_frames < 0) {
            fprintf(stderr, "error: projection computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * trans = nullptr;
        int n_trans_frames = sortformer_compute_transformer(ctx, proj, n_proj_frames,
                                                             d_proj, 17, &trans);
        free(proj);

        if (n_trans_frames < 0) {
            fprintf(stderr, "error: transformer computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        float * pred = nullptr;
        int n_pred_frames = sortformer_compute_prediction(ctx, trans, n_trans_frames,
                                                           d_proj, &pred);
        free(trans);

        if (n_pred_frames < 0) {
            fprintf(stderr, "error: prediction computation failed\n");
            sortformer_free(ctx);
            return 1;
        }

        const int n_spk = 4;
        fprintf(stderr, "sortformer-diarize: prediction shape = (%d, %d)\n",
                n_pred_frames, n_spk);

        const char * out_path = "cpp_pred.raw";
        FILE * fout = fopen(out_path, "wb");
        if (!fout) {
            fprintf(stderr, "error: failed to open '%s' for writing\n", out_path);
            free(pred);
            sortformer_free(ctx);
            return 1;
        }

        size_t n_written = fwrite(pred, sizeof(float), n_pred_frames * n_spk, fout);
        fclose(fout);

        if ((int)n_written != n_pred_frames * n_spk) {
            fprintf(stderr, "error: wrote %zu of %d floats\n", n_written, n_pred_frames * n_spk);
            free(pred);
            sortformer_free(ctx);
            return 1;
        }

        fprintf(stderr, "sortformer-diarize: wrote prediction to '%s' (%d x %d = %d floats, %zu bytes)\n",
                out_path, n_pred_frames, n_spk, n_pred_frames * n_spk,
                (size_t)(n_pred_frames * n_spk) * sizeof(float));

        free(pred);
        sortformer_free(ctx);
        return 0;
    }

    // Default mode: run streaming diarization
    {
        float * audio = nullptr;
        int n_samples = sortformer_load_wav(params.audio_file.c_str(), &audio);
        if (n_samples < 0) {
            fprintf(stderr, "error: failed to load audio\n");
            sortformer_free(ctx);
            return 1;
        }
        fprintf(stderr, "sortformer-diarize: loaded %d samples (%.2f seconds)\n",
                n_samples, (float)n_samples / 16000.0f);

        auto t0 = std::chrono::high_resolution_clock::now();

        const int max_frames = n_samples / 1280 + 100;
        float * probs = (float *)malloc(max_frames * 4 * sizeof(float));
        if (!probs) {
            fprintf(stderr, "error: allocation failed\n");
            free(audio);
            sortformer_free(ctx);
            return 1;
        }

        int n_frames = sortformer_diarize(ctx, audio, n_samples, probs, max_frames);
        free(audio);

        if (n_frames < 0) {
            fprintf(stderr, "error: diarization failed\n");
            free(probs);
            sortformer_free(ctx);
            return 1;
        }

         auto t1 = std::chrono::high_resolution_clock::now();
         float elapsed = std::chrono::duration<float>(t1 - t0).count();
         float audio_dur = (float)n_samples / 16000.0f;
         fprintf(stderr, "\nsortformer-diarize: total time = %.2f s\n", elapsed);
         fprintf(stderr, "sortformer-diarize: audio duration = %.2f s\n", audio_dur);
         fprintf(stderr, "sortformer-diarize: real-time factor = %.2f\n",
                 (elapsed > 0) ? (audio_dur / elapsed) : 0.0f);
         fprintf(stderr, "sortformer-diarize: peak memory = %.0f MB\n", get_peak_rss_mb());

        if (params.dump_probs) {
            for (int t = 0; t < n_frames; t++) {
                printf("%.6f %.6f %.6f %.6f\n",
                       probs[t*4+0], probs[t*4+1], probs[t*4+2], probs[t*4+3]);
            }
        } else {
            char rttm_buf[65536];
            int rttm_len = sortformer_to_rttm(probs, n_frames, params.threshold,
                                               params.median_filter,
                                               params.audio_file.c_str(),
                                               rttm_buf, sizeof(rttm_buf));
            if (rttm_len < 0) {
                fprintf(stderr, "error: RTTM generation failed (buffer too small?)\n");
                free(probs);
                sortformer_free(ctx);
                return 1;
            }

            if (!params.rttm_output.empty()) {
                FILE * f = fopen(params.rttm_output.c_str(), "w");
                if (!f) {
                    fprintf(stderr, "error: failed to open '%s' for writing\n",
                            params.rttm_output.c_str());
                    free(probs);
                    sortformer_free(ctx);
                    return 1;
                }
                fwrite(rttm_buf, 1, rttm_len, f);
                fclose(f);
                fprintf(stderr, "sortformer-diarize: wrote RTTM to '%s'\n",
                        params.rttm_output.c_str());
            } else {
                fwrite(rttm_buf, 1, rttm_len, stdout);
            }
        }

        free(probs);
        sortformer_free(ctx);
        return 0;
    }
}
