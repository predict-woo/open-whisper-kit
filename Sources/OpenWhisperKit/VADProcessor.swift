import Foundation
import whisper

/// Wraps whisper.cpp's built-in Silero VAD for speech detection.
/// Used to pre-filter audio and skip silence before transcription.
public final class VADProcessor: @unchecked Sendable {
    private let vadContext: OpaquePointer

    /// Initialize with path to a Silero VAD model (.bin file)
    public init(modelPath: String) throws {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw WhisperError.vadModelLoadFailed("VAD model not found at \(modelPath)")
        }

        var params = whisper_vad_default_context_params()
        params.use_gpu = true
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #endif

        guard let ctx = whisper_vad_init_from_file_with_params(modelPath, params) else {
            throw WhisperError.vadModelLoadFailed("Failed to initialize VAD model at \(modelPath)")
        }
        self.vadContext = ctx
    }

    deinit {
        whisper_vad_free(vadContext)
    }

    /// Detect speech segments in audio samples.
    /// Returns array of SpeechSegment with start/end times in seconds.
    public func detectSpeech(
        in samples: [Float],
        sampleRate: Int = 16000,
        params: whisper_vad_params? = nil
    ) throws -> [SpeechSegment] {
        let vadParams = params ?? whisper_vad_default_params()

        let segments = samples.withUnsafeBufferPointer { ptr in
            whisper_vad_segments_from_samples(
                vadContext,
                vadParams,
                ptr.baseAddress,
                Int32(ptr.count)
            )
        }

        guard let segments else {
            return []
        }

        defer { whisper_vad_free_segments(segments) }

        let nSegments = whisper_vad_segments_n_segments(segments)
        var result: [SpeechSegment] = []
        result.reserveCapacity(Int(nSegments))

        for i in 0..<nSegments {
            let t0 = whisper_vad_segments_get_segment_t0(segments, i)
            let t1 = whisper_vad_segments_get_segment_t1(segments, i)

            let startSample = Int(t0 * Float(sampleRate))
            let endSample = min(Int(t1 * Float(sampleRate)), samples.count)

            result.append(SpeechSegment(
                startTime: t0,
                endTime: t1,
                startSample: startSample,
                endSample: endSample
            ))
        }

        return result
    }

    /// Filter silence from audio, returning only speech samples.
    /// Useful for reducing processing time on long audio with lots of silence.
    public func filterSilence(
        from samples: [Float],
        sampleRate: Int = 16000
    ) throws -> [Float] {
        let speechSegments = try detectSpeech(in: samples, sampleRate: sampleRate)

        if speechSegments.isEmpty {
            return []
        }

        var filtered: [Float] = []
        for segment in speechSegments {
            let start = max(0, segment.startSample)
            let end = min(segment.endSample, samples.count)
            if start < end {
                filtered.append(contentsOf: samples[start..<end])
            }
        }

        return filtered
    }
}
