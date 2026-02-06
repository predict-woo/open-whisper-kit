import Foundation

/// Utility for parsing and generating RTTM (Rich Transcription Time Marked) format.
/// RTTM is a standard format for speaker diarization output.
public enum RTTMParser {
    /// Parse RTTM-formatted text into diarization segments.
    ///
    /// RTTM line format:
    /// `SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>`
    ///
    /// - Parameter rttmText: RTTM-formatted text with one segment per line
    /// - Returns: Array of DiarizationSegment sorted by start time
    public static func parse(_ rttmText: String) -> [DiarizationSegment] {
        guard !rttmText.isEmpty else {
            return []
        }
        
        var segments: [DiarizationSegment] = []
        
        let lines = rttmText.split(separator: "\n", omittingEmptySubsequences: true)
        
        for line in lines {
            let fields = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            
            // RTTM format: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
            // Indices:     0        1      2        3       4         5    6    7           8    9
            guard fields.count >= 8 else {
                continue
            }
            
            guard let start = Float(fields[3]),
                  let duration = Float(fields[4]) else {
                continue
            }
            
            let speaker = fields[7]
            let end = start + duration
            
            let segment = DiarizationSegment(
                speaker: speaker,
                start: start,
                end: end
            )
            segments.append(segment)
        }
        
        // Sort by start time
        return segments.sorted { $0.start < $1.start }
    }
    
    /// Generate RTTM-formatted text from diarization segments.
    ///
    /// - Parameters:
    ///   - segments: Array of DiarizationSegment to convert
    ///   - filename: Filename to include in the RTTM output
    /// - Returns: RTTM-formatted text with one segment per line
    public static func generate(segments: [DiarizationSegment], filename: String) -> String {
        let lines = segments.map { segment in
            let start = String(format: "%.2f", segment.start)
            let duration = String(format: "%.2f", segment.duration)
            return "SPEAKER \(filename) 1 \(start) \(duration) <NA> <NA> \(segment.speaker) <NA> <NA>"
        }
        
        return lines.joined(separator: "\n")
    }
}
