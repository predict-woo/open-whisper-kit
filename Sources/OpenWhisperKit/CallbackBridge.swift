import Foundation
import whisper

/// Bridges whisper.cpp C callbacks to Swift closures.
/// One instance per transcription call — NOT a singleton.
/// MUST be kept alive with `withExtendedLifetime` during `whisper_full()`.
final class CallbackBridge: @unchecked Sendable {
    var progressCallback: ((TranscriptionProgress) -> Bool?)?
    var segmentCallback: (([TranscriptionSegment]) -> Void)?
    var shouldCancel: Bool = false
    var currentText: String = ""

    // MARK: - C Function Pointers

    /// Matches `whisper_progress_callback` — normalizes 0-100 int to 0.0-1.0 Float.
    static let cProgressCallback: @convention(c) (OpaquePointer?, OpaquePointer?, Int32, UnsafeMutableRawPointer?) -> Void = { _, _, progress, userData in
        guard let userData = userData else { return }
        let bridge = Unmanaged<CallbackBridge>.fromOpaque(userData).takeUnretainedValue()

        let progressValue = TranscriptionProgress(
            progress: Float(progress) / 100.0,
            text: bridge.currentText,
            windowId: 0
        )

        if let callback = bridge.progressCallback {
            let shouldContinue = callback(progressValue)
            if shouldContinue == false {
                bridge.shouldCancel = true
            }
        }
    }

    /// Matches `ggml_abort_callback` — returns true to abort transcription.
    static let cAbortCallback: @convention(c) (UnsafeMutableRawPointer?) -> Bool = { userData in
        guard let userData = userData else { return false }
        let bridge = Unmanaged<CallbackBridge>.fromOpaque(userData).takeUnretainedValue()
        return bridge.shouldCancel
    }

    /// Matches `whisper_new_segment_callback` — extracts segments from whisper context.
    static let cNewSegmentCallback: @convention(c) (OpaquePointer?, OpaquePointer?, Int32, UnsafeMutableRawPointer?) -> Void = { ctx, _, nNew, userData in
        guard let userData = userData, let ctx = ctx else { return }
        let bridge = Unmanaged<CallbackBridge>.fromOpaque(userData).takeUnretainedValue()

        let totalSegments = whisper_full_n_segments(ctx)
        let startIdx = totalSegments - nNew

        var newSegments: [TranscriptionSegment] = []
        for i in startIdx..<totalSegments {
            let t0 = whisper_full_get_segment_t0(ctx, i)
            let t1 = whisper_full_get_segment_t1(ctx, i)
            let text = String(cString: whisper_full_get_segment_text(ctx, i))

            newSegments.append(TranscriptionSegment(
                id: Int(i),
                seek: 0,
                start: Float(t0) / 100.0,
                end: Float(t1) / 100.0,
                text: text,
                tokens: [],
                temperature: 0,
                avgLogprob: 0,
                compressionRatio: 0,
                noSpeechProb: 0
            ))
        }

        bridge.currentText = newSegments.map(\.text).joined()
        bridge.segmentCallback?(newSegments)
    }

    // MARK: - Params Configuration

    /// Wires all three callbacks into `whisper_full_params`.
    /// MUST be called within `withExtendedLifetime(self)` scope.
    func configureParams(_ params: inout whisper_full_params) {
        let bridgePtr = Unmanaged.passUnretained(self).toOpaque()

        params.progress_callback = CallbackBridge.cProgressCallback
        params.progress_callback_user_data = UnsafeMutableRawPointer(bridgePtr)

        params.new_segment_callback = CallbackBridge.cNewSegmentCallback
        params.new_segment_callback_user_data = UnsafeMutableRawPointer(bridgePtr)

        params.abort_callback = CallbackBridge.cAbortCallback
        params.abort_callback_user_data = UnsafeMutableRawPointer(bridgePtr)
    }
}
