import Foundation
import sortformer

public actor SortFormerContext {
    private let context: OpaquePointer

    /// Frame duration in seconds (80ms per frame, due to 8x subsampling of mel spectrogram)
    public static let frameDuration: Float = 0.08

    /// Maximum number of speakers the model supports
    public static let maxSpeakers: Int = 4

    private init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        sortformer_free(context)
    }

    // MARK: - Factory Method

    public static func createContext(
        modelPath: String,
        threadCount: Int? = nil
    ) throws -> SortFormerContext {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw DiarizationError.modelNotFound("Model not found at \(modelPath)")
        }

        var params = sortformer_default_params()

        if let threads = threadCount {
            params.n_threads = Int32(max(1, threads))
        } else {
            params.n_threads = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        }

        guard let ctx = sortformer_init(modelPath, params) else {
            throw DiarizationError.modelLoadFailed("Failed to load sortformer model at \(modelPath)")
        }

        return SortFormerContext(context: ctx)
    }

    // MARK: - Diarization

    public func diarize(
        samples: [Float],
        threshold: Float = 0.5,
        medianFilter: Int = 11
    ) throws -> DiarizationResult {
        let maxFrames = (samples.count / 1280) + 10
        var probs = [Float](repeating: 0, count: maxFrames * 4)

        let startTime = CFAbsoluteTimeGetCurrent()
        let nFrames = samples.withUnsafeBufferPointer { samplesPtr in
            probs.withUnsafeMutableBufferPointer { probsPtr in
                sortformer_diarize(
                    context,
                    samplesPtr.baseAddress,
                    Int32(samples.count),
                    probsPtr.baseAddress,
                    Int32(maxFrames)
                )
            }
        }
        let diarizationTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

        guard nFrames >= 0 else {
            throw DiarizationError.diarizationFailed("sortformer_diarize returned \(nFrames)")
        }

        var frameProbabilities: [[Float]] = []
        frameProbabilities.reserveCapacity(Int(nFrames))
        for i in 0..<Int(nFrames) {
            let offset = i * 4
            frameProbabilities.append(Array(probs[offset..<(offset + 4)]))
        }

        let rttmBufferSize = max(1, Int(nFrames) * 120)
        var rttmBuffer = [CChar](repeating: 0, count: rttmBufferSize)
        let rttmBufferCount = Int32(rttmBuffer.count)
        let rttmLen = probs.withUnsafeBufferPointer { probsPtr in
            rttmBuffer.withUnsafeMutableBufferPointer { rttmPtr in
                "audio".withCString { filenamePtr in
                    sortformer_to_rttm(
                        probsPtr.baseAddress,
                        nFrames,
                        threshold,
                        Int32(medianFilter),
                        filenamePtr,
                        rttmPtr.baseAddress,
                        rttmBufferCount
                    )
                }
            }
        }

        let rttm: String
        if rttmLen > 0 {
            rttm = String(cString: rttmBuffer)
        } else {
            rttm = ""
        }

        let segments = RTTMParser.parse(rttm)
        let inputAudioSeconds = Double(samples.count) / 16000.0

        return DiarizationResult(
            segments: segments,
            rttm: rttm,
            frameProbabilities: frameProbabilities,
            timings: DiarizationTimings(
                inputAudioSeconds: inputAudioSeconds,
                modelLoadingTime: 0,
                diarizationTime: diarizationTime,
                fullPipeline: diarizationTime
            )
        )
    }
}
