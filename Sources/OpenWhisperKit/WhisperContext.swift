import Foundation
import whisper

// MARK: - WhisperContext Actor

/// Owns and manages the `whisper_context` C pointer, bridging Swift to whisper.cpp's C API.
/// Actor isolation ensures single-threaded access required by whisper.cpp.
actor WhisperContext {
    private let context: OpaquePointer

    init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        whisper_free(context)
    }

    // MARK: - Static Factory

    static func createContext(config: OpenWhisperKitConfig) throws -> WhisperContext {
        var params = whisper_context_default_params()
        params.use_gpu = config.computeOptions.useGPU
        params.flash_attn = config.computeOptions.flashAttention
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #endif

        guard let ctx = whisper_init_from_file_with_params(config.modelPath, params) else {
            throw WhisperError.modelLoadFailed("Failed to load model at \(config.modelPath)")
        }
        return WhisperContext(context: ctx)
    }

    // MARK: - Core Transcription

    func fullTranscribe(samples: [Float], options: DecodingOptions, config: OpenWhisperKitConfig? = nil, bridge: CallbackBridge? = nil) throws {
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)

        let threadCount: Int
        if let configThreads = config?.computeOptions.threadCount {
            threadCount = max(1, configThreads)
        } else {
            threadCount = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        }
        params.n_threads = Int32(threadCount)

        params.translate = (options.task == .translate)

        params.no_timestamps = options.withoutTimestamps
        params.token_timestamps = options.wordTimestamps

        params.temperature = options.temperature
        params.temperature_inc = options.temperatureIncrementOnFallback
        params.entropy_thold = options.compressionRatioThreshold ?? 2.4
        params.logprob_thold = options.logProbThreshold ?? -1.0
        params.no_speech_thold = options.noSpeechThreshold ?? 0.6

        params.suppress_blank = options.suppressBlank
        params.max_initial_ts = options.maxInitialTimestamp ?? 1.0

        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
        params.print_special = false

        // Wire callback bridge for progress/segment/abort callbacks
        if let bridge = bridge {
            bridge.configureParams(&params)
        }

        let ctx = self.context
        let callWhisperFull: (inout whisper_full_params) -> Int32 = { params in
            let langStr = options.language ?? "auto"
            return langStr.withCString { langCStr in
                params.language = langCStr
                if let prompt = options.prompt {
                    return prompt.withCString { promptCStr in
                        params.initial_prompt = promptCStr
                        return samples.withUnsafeBufferPointer { ptr in
                            whisper_full(ctx, params, ptr.baseAddress, Int32(ptr.count))
                        }
                    }
                } else {
                    return samples.withUnsafeBufferPointer { ptr in
                        whisper_full(ctx, params, ptr.baseAddress, Int32(ptr.count))
                    }
                }
            }
        }

        // withExtendedLifetime ensures the bridge outlives the whisper_full call
        let result: Int32
        if let bridge = bridge {
            result = withExtendedLifetime(bridge) {
                callWhisperFull(&params)
            }
        } else {
            result = callWhisperFull(&params)
        }

        if result != 0 {
            throw WhisperError.transcriptionFailed("whisper_full returned \(result)")
        }
    }

    // MARK: - Result Extraction

    func getSegments(wordTimestamps: Bool = false) -> [TranscriptionSegment] {
        let nSegments = whisper_full_n_segments(context)
        var segments: [TranscriptionSegment] = []
        segments.reserveCapacity(Int(nSegments))

        for i in 0..<nSegments {
            // whisper.cpp timestamps are in centiseconds (int64_t)
            let t0 = whisper_full_get_segment_t0(context, i)
            let t1 = whisper_full_get_segment_t1(context, i)
            let text = String(cString: whisper_full_get_segment_text(context, i))
            let noSpeechProb = whisper_full_get_segment_no_speech_prob(context, i)

            let nTokens = whisper_full_n_tokens(context, i)
            var tokens: [Int] = []
            tokens.reserveCapacity(Int(nTokens))
            var words: [WordTiming]? = wordTimestamps ? [] : nil

            for j in 0..<nTokens {
                let tokenData = whisper_full_get_token_data(context, i, j)
                tokens.append(Int(tokenData.id))

                if wordTimestamps {
                    let tokenText = String(cString: whisper_full_get_token_text(context, i, j))
                    words?.append(WordTiming(
                        word: tokenText,
                        tokens: [Int(tokenData.id)],
                        start: Float(tokenData.t0) / 100.0,
                        end: Float(tokenData.t1) / 100.0,
                        probability: tokenData.p
                    ))
                }
            }

            segments.append(TranscriptionSegment(
                id: Int(i),
                seek: 0,
                start: Float(t0) / 100.0,
                end: Float(t1) / 100.0,
                text: text,
                tokens: tokens,
                temperature: 0,
                avgLogprob: 0,
                compressionRatio: 0,
                noSpeechProb: noSpeechProb,
                words: words
            ))
        }
        return segments
    }

    // MARK: - Timings

    func getTimings(inputAudioSeconds: Double) -> TranscriptionTimings {
        guard let timingsPtr = whisper_get_timings(context) else {
            return TranscriptionTimings(
                inputAudioSeconds: inputAudioSeconds,
                modelLoadingTime: 0,
                encodingTime: 0,
                decodingTime: 0,
                fullPipeline: 0
            )
        }
        let t = timingsPtr.pointee
        return TranscriptionTimings(
            inputAudioSeconds: inputAudioSeconds,
            modelLoadingTime: 0,
            encodingTime: Double(t.encode_ms),
            decodingTime: Double(t.decode_ms),
            fullPipeline: Double(t.sample_ms + t.encode_ms + t.decode_ms + t.batchd_ms + t.prompt_ms)
        )
    }

    // MARK: - Language Detection

    func detectLanguage(samples: [Float]) throws -> (String, [String: Float]) {
        let threadCount = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        let melResult = samples.withUnsafeBufferPointer { ptr in
            whisper_pcm_to_mel(context, ptr.baseAddress, Int32(ptr.count), threadCount)
        }
        if melResult != 0 {
            throw WhisperError.transcriptionFailed("whisper_pcm_to_mel failed with code \(melResult)")
        }

        let nLangs = whisper_lang_max_id() + 1
        var probs = [Float](repeating: 0, count: Int(nLangs))
        let langId = probs.withUnsafeMutableBufferPointer { ptr in
            whisper_lang_auto_detect(context, 0, threadCount, ptr.baseAddress)
        }

        if langId < 0 {
            throw WhisperError.transcriptionFailed("whisper_lang_auto_detect failed with code \(langId)")
        }

        let detectedLang = String(cString: whisper_lang_str(langId))
        var langProbs: [String: Float] = [:]
        for i in 0..<Int(nLangs) {
            let langStr = String(cString: whisper_lang_str(Int32(i)))
            langProbs[langStr] = probs[i]
        }

        return (detectedLang, langProbs)
    }

    // MARK: - Model Info

    var isMultilingual: Bool {
        whisper_is_multilingual(context) != 0
    }

    var modelType: String {
        String(cString: whisper_model_type_readable(context))
    }
}
