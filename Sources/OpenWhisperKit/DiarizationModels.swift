import Foundation

// MARK: - Main Result Container

/// The complete diarization result containing speaker segments, RTTM format, and timing information.
public struct DiarizationResult: Sendable {
    /// Array of diarization segments with speaker labels and timestamps
    public var segments: [DiarizationSegment]
    
    /// RTTM-formatted text representation of the diarization
    public var rttm: String
    
    /// Raw frame-level probabilities from the diarization model [T][4] where T is number of frames
    public var frameProbabilities: [[Float]]
    
    /// Performance timing metrics for the diarization
    public var timings: DiarizationTimings
    
    /// Initializes a DiarizationResult.
    public init(
        segments: [DiarizationSegment],
        rttm: String,
        frameProbabilities: [[Float]],
        timings: DiarizationTimings
    ) {
        self.segments = segments
        self.rttm = rttm
        self.frameProbabilities = frameProbabilities
        self.timings = timings
    }
}

// MARK: - Segment

/// A single diarization segment with speaker label and timestamps.
public struct DiarizationSegment: Codable, Hashable, Sendable {
    /// Speaker identifier (e.g., "SPEAKER_00", "SPEAKER_01")
    public var speaker: String
    
    /// Start time of the segment in seconds
    public var start: Float
    
    /// End time of the segment in seconds
    public var end: Float
    
    /// Computed duration of the segment in seconds
    public var duration: Float { end - start }
    
    /// Initializes a DiarizationSegment.
    public init(
        speaker: String,
        start: Float,
        end: Float
    ) {
        self.speaker = speaker
        self.start = start
        self.end = end
    }
}

// MARK: - Timings

/// Performance metrics for diarization including timing breakdowns.
public struct DiarizationTimings: Sendable {
    /// Duration of input audio in seconds
    public var inputAudioSeconds: Double
    
    /// Time taken to load the model in milliseconds
    public var modelLoadingTime: Double
    
    /// Time taken for diarization in milliseconds
    public var diarizationTime: Double
    
    /// Total time for the full pipeline in milliseconds
    public var fullPipeline: Double
    
    /// Initializes DiarizationTimings.
    public init(
        inputAudioSeconds: Double,
        modelLoadingTime: Double,
        diarizationTime: Double,
        fullPipeline: Double
    ) {
        self.inputAudioSeconds = inputAudioSeconds
        self.modelLoadingTime = modelLoadingTime
        self.diarizationTime = diarizationTime
        self.fullPipeline = fullPipeline
    }
}

// MARK: - Word Alignment

/// Word-level diarization information with speaker assignment and timing.
public struct DiarizedWord: Codable, Hashable, Sendable {
    /// The word text
    public var word: String
    
    /// Start time of the word in seconds
    public var start: Float
    
    /// End time of the word in seconds
    public var end: Float
    
    /// Speaker identifier for this word (optional)
    public var speaker: String?
    
    /// Probability score for this word
    public var probability: Float
    
    /// Computed duration of the word in seconds
    public var duration: Float { end - start }
    
    /// Initializes a DiarizedWord.
    public init(
        word: String,
        start: Float,
        end: Float,
        speaker: String? = nil,
        probability: Float
    ) {
        self.word = word
        self.start = start
        self.end = end
        self.speaker = speaker
        self.probability = probability
    }
}

// MARK: - Utterance

/// A speaker turn or utterance with word-level alignment.
public struct DiarizedUtterance: Codable, Hashable, Sendable {
    /// Speaker identifier for this utterance (optional)
    public var speaker: String?
    
    /// The transcribed text for this utterance
    public var text: String
    
    /// Start time of the utterance in seconds
    public var start: Float
    
    /// End time of the utterance in seconds
    public var end: Float
    
    /// Word-level timing information for this utterance
    public var words: [DiarizedWord]
    
    /// Computed duration of the utterance in seconds
    public var duration: Float { end - start }
    
    /// Initializes a DiarizedUtterance.
    public init(
        speaker: String? = nil,
        text: String,
        start: Float,
        end: Float,
        words: [DiarizedWord]
    ) {
        self.speaker = speaker
        self.text = text
        self.start = start
        self.end = end
        self.words = words
    }
}

// MARK: - Transcription

/// The complete diarized transcription with word and utterance alignment.
public struct DiarizedTranscription: Sendable {
    /// Word-level diarization information
    public var words: [DiarizedWord]
    
    /// Utterance-level diarization information
    public var segments: [DiarizedUtterance]
    
    /// Full transcribed text
    public var text: String
    
    /// Initializes a DiarizedTranscription.
    public init(
        words: [DiarizedWord],
        segments: [DiarizedUtterance],
        text: String
    ) {
        self.words = words
        self.segments = segments
        self.text = text
    }
}
