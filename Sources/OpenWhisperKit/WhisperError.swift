import Foundation

/// Errors that can occur during transcription operations.
public enum WhisperError: Error, LocalizedError, Sendable {
    /// Model file not found at the specified path
    case modelNotFound(String)
    
    /// Failed to load the model
    case modelLoadFailed(String)
    
    /// Failed to load the VAD model
    case vadModelLoadFailed(String)
    
    /// Transcription operation failed
    case transcriptionFailed(String)
    
    /// Failed to load audio file
    case audioLoadFailed(String)
    
    /// Audio format is not supported
    case invalidAudioFormat(String)
    
    /// Transcription was cancelled by the user
    case cancelled
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let path):
            return "Model file not found at path: \(path)"
        case .modelLoadFailed(let reason):
            return "Failed to load model: \(reason)"
        case .vadModelLoadFailed(let reason):
            return "Failed to load VAD model: \(reason)"
        case .transcriptionFailed(let reason):
            return "Transcription failed: \(reason)"
        case .audioLoadFailed(let reason):
            return "Failed to load audio: \(reason)"
        case .invalidAudioFormat(let format):
            return "Invalid audio format: \(format)"
        case .cancelled:
            return "Transcription was cancelled"
        }
    }
    
    public var failureReason: String? {
        switch self {
        case .modelNotFound:
            return "The specified model file does not exist"
        case .modelLoadFailed:
            return "The model could not be loaded into memory"
        case .vadModelLoadFailed:
            return "The VAD model could not be loaded into memory"
        case .transcriptionFailed:
            return "An error occurred during transcription processing"
        case .audioLoadFailed:
            return "The audio file could not be read"
        case .invalidAudioFormat:
            return "The audio format is not supported"
        case .cancelled:
            return "The operation was cancelled by the user"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .modelNotFound(let path):
            return "Ensure the model file exists at: \(path)"
        case .modelLoadFailed:
            return "Check that you have sufficient memory and the model file is not corrupted"
        case .vadModelLoadFailed:
            return "Check that you have sufficient memory and the VAD model file is not corrupted"
        case .transcriptionFailed:
            return "Try again with different audio or check the logs for more details"
        case .audioLoadFailed:
            return "Ensure the audio file exists and is readable"
        case .invalidAudioFormat:
            return "Convert the audio to a supported format (WAV, MP3, etc.)"
        case .cancelled:
            return "The operation was cancelled. You can try again."
        }
    }
}
