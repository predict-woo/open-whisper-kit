import Foundation

// MARK: - Main Configuration

/// Configuration for initializing the OpenWhisperKit SDK.
public struct OpenWhisperKitConfig: Sendable {
    /// Path to the Whisper model file
    public var modelPath: String
    
    /// Optional path to the Voice Activity Detection model file
    public var vadModelPath: String?
    
    /// Compute options for model inference
    public var computeOptions: ComputeOptions
    
    /// Initializes OpenWhisperKitConfig with default compute options.
    public init(
        modelPath: String,
        vadModelPath: String? = nil,
        computeOptions: ComputeOptions = ComputeOptions()
    ) {
        self.modelPath = modelPath
        self.vadModelPath = vadModelPath
        self.computeOptions = computeOptions
    }
}

// MARK: - Compute Options

/// Options controlling how the model is executed (GPU, CPU, threading, etc.).
public struct ComputeOptions: Sendable {
    /// Whether to use GPU acceleration if available
    public var useGPU: Bool
    
    /// Whether to use Core ML acceleration on Apple platforms
    public var useCoreML: Bool
    
    /// Whether to use Flash Attention optimization if available
    public var flashAttention: Bool
    
    /// Number of threads for CPU inference (nil = auto-detect)
    public var threadCount: Int?
    
    /// Initializes ComputeOptions with sensible defaults.
    public init(
        useGPU: Bool = true,
        useCoreML: Bool = true,
        flashAttention: Bool = true,
        threadCount: Int? = nil
    ) {
        self.useGPU = useGPU
        self.useCoreML = useCoreML
        self.flashAttention = flashAttention
        self.threadCount = threadCount
    }
}

// MARK: - Decoding Options

/// Options controlling the transcription decoding process.
public struct DecodingOptions: Sendable {
    /// Task to perform (transcribe or translate)
    public var task: DecodingTask
    
    /// Language code to use (nil = auto-detect)
    public var language: String?
    
    /// Temperature for sampling (0.0 = deterministic, higher = more random)
    public var temperature: Float
    
    /// Temperature increment on fallback
    public var temperatureIncrementOnFallback: Float
    
    /// Number of temperature fallback attempts
    public var temperatureFallbackCount: Int
    
    /// Top-K sampling parameter
    public var topK: Int
    
    /// Whether to suppress timestamps in output
    public var withoutTimestamps: Bool
    
    /// Whether to include word-level timestamps
    public var wordTimestamps: Bool
    
    /// Maximum initial timestamp in seconds (nil = no limit)
    public var maxInitialTimestamp: Float?
    
    /// Whether to suppress segments with no speech
    public var suppressBlank: Bool
    
    /// Compression ratio threshold for filtering (nil = no filtering)
    public var compressionRatioThreshold: Float?
    
    /// Log probability threshold for filtering (nil = no filtering)
    public var logProbThreshold: Float?
    
    /// No-speech probability threshold (nil = no filtering)
    public var noSpeechThreshold: Float?
    
    /// Strategy for chunking audio during processing
    public var chunkingStrategy: ChunkingStrategy
    
    /// Optional prompt to guide the model
    public var prompt: String?
    
    /// Initializes DecodingOptions with sensible defaults.
    public init(
        task: DecodingTask = .transcribe,
        language: String? = nil,
        temperature: Float = 0.0,
        temperatureIncrementOnFallback: Float = 0.2,
        temperatureFallbackCount: Int = 5,
        topK: Int = 5,
        withoutTimestamps: Bool = false,
        wordTimestamps: Bool = false,
        maxInitialTimestamp: Float? = nil,
        suppressBlank: Bool = true,
        compressionRatioThreshold: Float? = 2.4,
        logProbThreshold: Float? = -1.0,
        noSpeechThreshold: Float? = 0.6,
        chunkingStrategy: ChunkingStrategy = .none,
        prompt: String? = nil
    ) {
        self.task = task
        self.language = language
        self.temperature = temperature
        self.temperatureIncrementOnFallback = temperatureIncrementOnFallback
        self.temperatureFallbackCount = temperatureFallbackCount
        self.topK = topK
        self.withoutTimestamps = withoutTimestamps
        self.wordTimestamps = wordTimestamps
        self.maxInitialTimestamp = maxInitialTimestamp
        self.suppressBlank = suppressBlank
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logProbThreshold = logProbThreshold
        self.noSpeechThreshold = noSpeechThreshold
        self.chunkingStrategy = chunkingStrategy
        self.prompt = prompt
    }
}

// MARK: - Decoding Task

/// The task to perform during transcription.
public enum DecodingTask: String, Codable, Sendable {
    /// Transcribe audio in its original language
    case transcribe
    
    /// Transcribe audio and translate to English
    case translate
}

// MARK: - Chunking Strategy

/// Strategy for chunking audio during processing.
public enum ChunkingStrategy: String, Sendable {
    /// No chunking, process entire audio at once
    case none
    
    /// Use Voice Activity Detection to chunk audio
    case vad
}
