import Foundation

// MARK: - Callback Type

/// Callback function type for transcription progress updates.
/// Returns a boolean indicating whether to continue transcription (true) or cancel (false).
/// Returning nil is treated as continue.
public typealias TranscriptionCallback = ((TranscriptionProgress) -> Bool?)?

// MARK: - Transcribing Protocol

/// Protocol for transcription operations.
public protocol Transcribing: Sendable {
    /// Transcribe audio from a file path.
    /// - Parameters:
    ///   - audioPath: Path to the audio file
    ///   - options: Decoding options (uses defaults if nil)
    ///   - callback: Optional callback for progress updates
    /// - Returns: The transcription result
    /// - Throws: WhisperError if transcription fails
    func transcribe(
        audioPath: String,
        options: DecodingOptions?,
        callback: TranscriptionCallback
    ) async throws -> TranscriptionResult
    
    /// Transcribe audio from raw samples.
    /// - Parameters:
    ///   - audioSamples: Array of audio samples (typically 16-bit PCM at 16kHz)
    ///   - options: Decoding options (uses defaults if nil)
    ///   - callback: Optional callback for progress updates
    /// - Returns: The transcription result
    /// - Throws: WhisperError if transcription fails
    func transcribe(
        audioSamples: [Float],
        options: DecodingOptions?,
        callback: TranscriptionCallback
    ) async throws -> TranscriptionResult
    
    /// Detect the language of audio from a file path.
    /// - Parameters:
    ///   - audioPath: Path to the audio file
    /// - Returns: Tuple containing the detected language code and probability scores for all languages
    /// - Throws: WhisperError if language detection fails
    func detectLanguage(
        audioPath: String
    ) async throws -> (language: String, probabilities: [String: Float])
}
