import AVFoundation
import Foundation

/// Utility for loading and converting audio files to the format whisper.cpp expects.
/// whisper.cpp requires: Float32 PCM, 16kHz sample rate, mono channel.
public enum AudioProcessor {
    /// The sample rate whisper.cpp expects.
    public static let sampleRate: Double = 16_000.0

    /// Load audio from a file path and convert to Float32 PCM @ 16kHz mono.
    /// Supports any format AVFoundation can read: wav, mp3, m4a, flac, caf, etc.
    public static func loadAudio(fromPath path: String) throws -> [Float] {
        guard FileManager.default.fileExists(atPath: path) else {
            throw WhisperError.audioLoadFailed("Cannot open audio file at \(path): file does not exist")
        }

        let url = URL(fileURLWithPath: path)
        return try loadAudio(fromURL: url)
    }

    /// Load audio from a URL and convert to Float32 PCM @ 16kHz mono.
    public static func loadAudio(fromURL url: URL) throws -> [Float] {
        let audioFile: AVAudioFile
        do {
            audioFile = try AVAudioFile(forReading: url)
        } catch {
            throw WhisperError.audioLoadFailed("Cannot open audio file at \(url.path): \(error.localizedDescription)")
        }

        let inputFrameCount = AVAudioFrameCount(audioFile.length)
        guard inputFrameCount > 0 else {
            throw WhisperError.audioLoadFailed("Audio file contains no frames: \(url.path)")
        }

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: audioFile.processingFormat,
            frameCapacity: inputFrameCount
        ) else {
            throw WhisperError.audioLoadFailed("Cannot create input buffer")
        }

        do {
            try audioFile.read(into: inputBuffer)
        } catch {
            throw WhisperError.audioLoadFailed("Cannot read audio file: \(error.localizedDescription)")
        }

        return try convertToMono16kHz(inputBuffer)
    }

    /// Convert an existing AVAudioPCMBuffer to mono 16kHz Float32.
    public static func convertToMono16kHz(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard buffer.frameLength > 0 else {
            throw WhisperError.audioLoadFailed("Audio buffer contains no frames")
        }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw WhisperError.invalidAudioFormat("Cannot create target audio format")
        }

        if buffer.format.sampleRate == sampleRate,
           buffer.format.channelCount == 1,
           buffer.format.commonFormat == .pcmFormatFloat32,
           !buffer.format.isInterleaved {
            return try extractSamples(from: buffer)
        }

        guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat) else {
            throw WhisperError.invalidAudioFormat(
                "Cannot create audio converter from \(buffer.format) to \(targetFormat)"
            )
        }

        let ratio = sampleRate / buffer.format.sampleRate
        let estimatedOutputFrames = Int(ceil(Double(buffer.frameLength) * ratio))
        let outputCapacity = AVAudioFrameCount(max(1, estimatedOutputFrames + 1))

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outputCapacity
        ) else {
            throw WhisperError.audioLoadFailed("Cannot create output buffer")
        }

        var didProvideInput = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if didProvideInput {
                outStatus.pointee = .endOfStream
                return nil
            }

            didProvideInput = true
            outStatus.pointee = .haveData
            return buffer
        }

        if let error = conversionError {
            throw WhisperError.invalidAudioFormat("Audio conversion failed: \(error.localizedDescription)")
        }

        guard status != .error else {
            throw WhisperError.invalidAudioFormat("Audio conversion failed")
        }

        guard outputBuffer.frameLength > 0 else {
            throw WhisperError.audioLoadFailed("Audio conversion produced no output samples")
        }

        return try extractSamples(from: outputBuffer)
    }

    private static func extractSamples(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard let floatData = buffer.floatChannelData else {
            throw WhisperError.audioLoadFailed("Cannot access float channel data")
        }

        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else {
            return []
        }

        return Array(UnsafeBufferPointer(start: floatData[0], count: frameLength))
    }
}
