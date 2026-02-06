import Foundation

// MARK: - Main Result Container

/// The complete transcription result containing text, segments, language, and timing information.
public struct TranscriptionResult: Sendable {
    /// The full transcribed text
    public var text: String
    
    /// Array of transcription segments with timestamps and metadata
    public var segments: [TranscriptionSegment]
    
    /// Detected language code (e.g., "en", "fr", "de")
    public var language: String
    
    /// Performance timing metrics for the transcription
    public var timings: TranscriptionTimings
    
    /// Initializes a TranscriptionResult.
    public init(
        text: String,
        segments: [TranscriptionSegment],
        language: String,
        timings: TranscriptionTimings
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.timings = timings
    }
}

// MARK: - Segment

/// A single segment of transcribed audio with timestamps, tokens, and optional word-level timings.
public struct TranscriptionSegment: Codable, Hashable, Sendable {
    /// Unique identifier for this segment
    public var id: Int
    
    /// Seek position in the audio (in centiseconds)
    public var seek: Int
    
    /// Start time of the segment in seconds
    public var start: Float
    
    /// End time of the segment in seconds
    public var end: Float
    
    /// The transcribed text for this segment
    public var text: String
    
    /// Array of token IDs for this segment
    public var tokens: [Int]
    
    /// Temperature used for decoding this segment
    public var temperature: Float
    
    /// Average log probability of tokens in this segment
    public var avgLogprob: Float
    
    /// Compression ratio of the segment
    public var compressionRatio: Float
    
    /// Probability that this segment contains no speech
    public var noSpeechProb: Float
    
    /// Optional word-level timing information
    public var words: [WordTiming]?
    
    /// Computed duration of the segment in seconds
    public var duration: Float { end - start }
    
    /// Initializes a TranscriptionSegment.
    public init(
        id: Int,
        seek: Int,
        start: Float,
        end: Float,
        text: String,
        tokens: [Int],
        temperature: Float,
        avgLogprob: Float,
        compressionRatio: Float,
        noSpeechProb: Float,
        words: [WordTiming]? = nil
    ) {
        self.id = id
        self.seek = seek
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens
        self.temperature = temperature
        self.avgLogprob = avgLogprob
        self.compressionRatio = compressionRatio
        self.noSpeechProb = noSpeechProb
        self.words = words
    }
    
    enum CodingKeys: String, CodingKey {
        case id, seek, start, end, text, tokens, temperature
        case avgLogprob = "avg_logprob"
        case compressionRatio = "compression_ratio"
        case noSpeechProb = "no_speech_prob"
        case words
    }
}

// MARK: - Word Timing

/// Word-level timing information with probability scores.
public struct WordTiming: Codable, Hashable, Sendable {
    /// The word text
    public var word: String
    
    /// Token IDs that make up this word
    public var tokens: [Int]
    
    /// Start time of the word in seconds
    public var start: Float
    
    /// End time of the word in seconds
    public var end: Float
    
    /// Probability score for this word
    public var probability: Float
    
    /// Computed duration of the word in seconds
    public var duration: Float { end - start }
    
    /// Initializes a WordTiming.
    public init(
        word: String,
        tokens: [Int],
        start: Float,
        end: Float,
        probability: Float
    ) {
        self.word = word
        self.tokens = tokens
        self.start = start
        self.end = end
        self.probability = probability
    }
}

// MARK: - Timings

/// Performance metrics for transcription including timing breakdowns.
public struct TranscriptionTimings: Sendable {
    /// Duration of input audio in seconds
    public var inputAudioSeconds: Double
    
    /// Time taken to load the model in milliseconds
    public var modelLoadingTime: Double
    
    /// Time taken for encoding in milliseconds
    public var encodingTime: Double
    
    /// Time taken for decoding in milliseconds
    public var decodingTime: Double
    
    /// Total time for the full pipeline in milliseconds
    public var fullPipeline: Double
    
    /// Initializes TranscriptionTimings.
    public init(
        inputAudioSeconds: Double,
        modelLoadingTime: Double,
        encodingTime: Double,
        decodingTime: Double,
        fullPipeline: Double
    ) {
        self.inputAudioSeconds = inputAudioSeconds
        self.modelLoadingTime = modelLoadingTime
        self.encodingTime = encodingTime
        self.decodingTime = decodingTime
        self.fullPipeline = fullPipeline
    }
    
    /// Computed tokens per second (tokens / decoding time)
    public var tokensPerSecond: Double {
        guard decodingTime > 0 else { return 0 }
        return 1000.0 / decodingTime
    }
    
    /// Computed real-time factor (full pipeline time / input audio time)
    public var realTimeFactor: Double {
        guard inputAudioSeconds > 0 else { return 0 }
        return (fullPipeline / 1000.0) / inputAudioSeconds
    }
    
    /// Computed speed factor (input audio time / full pipeline time)
    public var speedFactor: Double {
        guard fullPipeline > 0 else { return 0 }
        return (inputAudioSeconds * 1000.0) / fullPipeline
    }
}

// MARK: - Progress

/// Progress information during active transcription.
public struct TranscriptionProgress: Sendable {
    /// Progress value from 0.0 to 1.0
    public var progress: Float
    
    /// Partial transcribed text so far
    public var text: String
    
    /// ID of the current processing window
    public var windowId: Int
    
    /// Initializes TranscriptionProgress.
    public init(
        progress: Float,
        text: String,
        windowId: Int
    ) {
        self.progress = progress
        self.text = text
        self.windowId = windowId
    }
}

// MARK: - Model State

/// Lifecycle state of the transcription model.
public enum ModelState: String, Sendable {
    /// Model is not loaded
    case unloaded
    
    /// Model is currently loading
    case loading
    
    /// Model is fully loaded and ready
    case loaded
}

// MARK: - Model Variant

/// Available Whisper model sizes and variants.
public enum ModelVariant: String, CaseIterable, Sendable {
    /// Tiny model (39M parameters)
    case tiny
    
    /// Tiny English-only model
    case tinyEn = "tiny.en"
    
    /// Base model (74M parameters)
    case base
    
    /// Base English-only model
    case baseEn = "base.en"
    
    /// Small model (244M parameters)
    case small
    
    /// Small English-only model
    case smallEn = "small.en"
    
    /// Medium model (769M parameters)
    case medium
    
    /// Medium English-only model
    case mediumEn = "medium.en"
    
    /// Large v1 model (1.5B parameters)
    case largeV1 = "large-v1"
    
    /// Large v2 model (1.5B parameters)
    case largeV2 = "large-v2"
    
    /// Large v3 model (1.5B parameters)
    case largeV3 = "large-v3"
    
    /// Large v3 Turbo model (optimized)
    case largeV3Turbo = "large-v3-turbo"
    
    /// Whether this model supports multiple languages
    public var isMultilingual: Bool {
        !rawValue.hasSuffix(".en")
    }
}

// MARK: - Speech Segment

/// A segment of detected speech from Voice Activity Detection.
public struct SpeechSegment: Sendable {
    /// Start time of the speech segment in seconds
    public var startTime: Float
    
    /// End time of the speech segment in seconds
    public var endTime: Float
    
    /// Start sample index in the audio
    public var startSample: Int
    
    /// End sample index in the audio
    public var endSample: Int
    
    /// Initializes a SpeechSegment.
    public init(
        startTime: Float,
        endTime: Float,
        startSample: Int,
        endSample: Int
    ) {
        self.startTime = startTime
        self.endTime = endTime
        self.startSample = startSample
        self.endSample = endSample
    }
}
