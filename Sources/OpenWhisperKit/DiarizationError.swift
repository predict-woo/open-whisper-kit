import Foundation

/// Errors that can occur during diarization operations.
public enum DiarizationError: Error, LocalizedError, Sendable {
    /// Model file not found at the specified path
    case modelNotFound(String)
    
    /// Failed to load the diarization model
    case modelLoadFailed(String)
    
    /// Diarization operation failed
    case diarizationFailed(String)
    
    /// Audio format is not supported
    case invalidAudioFormat(String)
    
    /// Word-level timestamp alignment failed
    case alignmentFailed(String)
    
    /// Word timestamps are required but not available
    case wordTimestampsRequired
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Diarization model file not found at path: \(path)"
        case .modelLoadFailed(let reason):
            return "Failed to load diarization model: \(reason)"
        case .diarizationFailed(let reason):
            return "Diarization failed: \(reason)"
        case .invalidAudioFormat(let format):
            return "Invalid audio format: \(format)"
        case .alignmentFailed(let reason):
            return "Word alignment failed: \(reason)"
        case .wordTimestampsRequired:
            return "Word-level timestamps are required for diarization alignment"
        }
    }
    
    public var failureReason: String? {
        switch self {
        case .modelNotFound:
            return "The specified diarization model file does not exist"
        case .modelLoadFailed:
            return "The diarization model could not be loaded into memory"
        case .diarizationFailed:
            return "An error occurred during diarization processing"
        case .invalidAudioFormat:
            return "The audio format is not supported for diarization"
        case .alignmentFailed:
            return "Word-level alignment could not be completed"
        case .wordTimestampsRequired:
            return "Transcription must include word-level timestamps"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound(let path):
            return "Ensure the diarization model file exists at: \(path)"
        case .modelLoadFailed:
            return "Check that you have sufficient memory and the model file is not corrupted"
        case .diarizationFailed:
            return "Try again with different audio or check the logs for more details"
        case .invalidAudioFormat:
            return "Convert the audio to a supported format (WAV, MP3, etc.)"
        case .alignmentFailed:
            return "Ensure the transcription includes accurate word-level timestamps"
        case .wordTimestampsRequired:
            return "Enable word-level timestamps in the transcription configuration"
        }
    }
}
