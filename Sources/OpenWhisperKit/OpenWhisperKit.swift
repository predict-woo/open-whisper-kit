import Foundation

/// Main entry point for the OpenWhisperKit SDK.
/// Provides a WhisperKit-inspired API backed by whisper.cpp's fast inference engine.
public final class OpenWhisperKit: @unchecked Sendable {

    public private(set) var modelState: ModelState = .unloaded

    public private(set) var currentTimings: TranscriptionTimings?

    public let config: OpenWhisperKitConfig

    private let whisperContext: WhisperContext
    private let vadProcessor: VADProcessor?

    // MARK: - Initialization

    public init(_ config: OpenWhisperKitConfig) async throws {
        guard FileManager.default.fileExists(atPath: config.modelPath) else {
            throw WhisperError.modelNotFound("Model not found at \(config.modelPath)")
        }

        self.config = config
        self.modelState = .loading

        self.whisperContext = try WhisperContext.createContext(config: config)

        if let vadPath = config.vadModelPath {
            self.vadProcessor = try VADProcessor(modelPath: vadPath)
        } else {
            self.vadProcessor = nil
        }

        self.modelState = .loaded
    }

    public convenience init(modelPath: String, vadModelPath: String? = nil) async throws {
        let config = OpenWhisperKitConfig(modelPath: modelPath, vadModelPath: vadModelPath)
        try await self.init(config)
    }
}

// MARK: - Transcribing Conformance

extension OpenWhisperKit: Transcribing {

    public func transcribe(
        audioPath: String,
        options: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult {
        let samples = try AudioProcessor.loadAudio(fromPath: audioPath)
        return try await transcribe(audioSamples: samples, options: options, callback: callback)
    }

    public func transcribe(
        audioSamples: [Float],
        options: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async throws -> TranscriptionResult {
        guard !audioSamples.isEmpty else {
            return TranscriptionResult(
                text: "",
                segments: [],
                language: "",
                timings: TranscriptionTimings(
                    inputAudioSeconds: 0,
                    modelLoadingTime: 0,
                    encodingTime: 0,
                    decodingTime: 0,
                    fullPipeline: 0
                )
            )
        }

        let opts = options ?? DecodingOptions()

        if opts.chunkingStrategy == .vad, let vad = vadProcessor {
            return try await TranscriptionTask.transcribeWithVAD(
                samples: audioSamples,
                options: opts,
                config: config,
                context: whisperContext,
                vad: vad,
                callback: callback,
                timingsStore: { [weak self] in self?.currentTimings = $0 }
            )
        }

        return try await TranscriptionTask.transcribeDirect(
            samples: audioSamples,
            options: opts,
            config: config,
            context: whisperContext,
            callback: callback,
            timingsStore: { [weak self] in self?.currentTimings = $0 }
        )
    }

    public func detectLanguage(
        audioPath: String
    ) async throws -> (language: String, probabilities: [String: Float]) {
        let samples = try AudioProcessor.loadAudio(fromPath: audioPath)
        return try await detectLanguage(audioSamples: samples)
    }

    public func detectLanguage(
        audioSamples: [Float]
    ) async throws -> (language: String, probabilities: [String: Float]) {
        let result = try await whisperContext.detectLanguage(samples: audioSamples)
        return (language: result.0, probabilities: result.1)
    }
}

// MARK: - Batch Transcription

extension OpenWhisperKit {

    public func transcribe(
        audioPaths: [String],
        options: DecodingOptions? = nil,
        callback: TranscriptionCallback = nil
    ) async -> [Result<TranscriptionResult, Error>] {
        var results: [Result<TranscriptionResult, Error>] = []
        results.reserveCapacity(audioPaths.count)

        for path in audioPaths {
            do {
                let result = try await transcribe(audioPath: path, options: options, callback: callback)
                results.append(.success(result))
            } catch {
                results.append(.failure(error))
            }
        }

        return results
    }
}
