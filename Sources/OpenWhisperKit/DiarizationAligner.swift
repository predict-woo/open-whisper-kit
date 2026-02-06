import Foundation

/// Utility for aligning word-level transcription timestamps with diarization segments.
public enum DiarizationAligner {
    public struct AlignmentOptions: Sendable {
        public var fillNearest: Bool
        public var sentenceSmoothing: Bool
        public var maxWordsInSentence: Int

        public init(
            fillNearest: Bool = false,
            sentenceSmoothing: Bool = true,
            maxWordsInSentence: Int = 50
        ) {
            self.fillNearest = fillNearest
            self.sentenceSmoothing = sentenceSmoothing
            self.maxWordsInSentence = maxWordsInSentence
        }
    }

    public static func align(
        words: [WordTiming],
        diarizationSegments: [DiarizationSegment],
        options: AlignmentOptions = AlignmentOptions()
    ) throws -> DiarizedTranscription {
        guard options.maxWordsInSentence > 0 else {
            throw DiarizationError.alignmentFailed("maxWordsInSentence must be greater than 0")
        }

        guard !words.isEmpty else {
            return DiarizedTranscription(words: [], segments: [], text: "")
        }

        let orderedSegments = diarizationSegments
            .enumerated()
            .sorted { lhs, rhs in
                if lhs.element.start != rhs.element.start {
                    return lhs.element.start < rhs.element.start
                }
                return lhs.offset < rhs.offset
            }
            .map(\.element)

        let speakerOrder = buildSpeakerOrder(segments: orderedSegments)

        var diarizedWords = words.map { word in
            DiarizedWord(
                word: word.word,
                start: word.start,
                end: word.end,
                speaker: assignSpeaker(
                    to: word,
                    using: orderedSegments,
                    speakerOrder: speakerOrder,
                    fillNearest: options.fillNearest
                ),
                probability: word.probability
            )
        }

        if options.sentenceSmoothing {
            smoothSentenceBoundaries(words: &diarizedWords, maxWordsInSentence: options.maxWordsInSentence)
        }

        let utterances = groupUtterances(words: diarizedWords)
        let text = utterances
            .map { utterance in
                let speakerLabel = utterance.speaker ?? "unknown"
                return "[\(speakerLabel)]: \(utterance.text)"
            }
            .joined(separator: "\n")

        return DiarizedTranscription(words: diarizedWords, segments: utterances, text: text)
    }

    private static func assignSpeaker(
        to word: WordTiming,
        using segments: [DiarizationSegment],
        speakerOrder: [String: Int],
        fillNearest: Bool
    ) -> String? {
        guard !segments.isEmpty else {
            return nil
        }

        let wordStart = min(word.start, word.end)
        let wordEnd = max(word.start, word.end)

        if wordStart == wordEnd {
            for segment in segments where contains(time: wordStart, in: segment) {
                return segment.speaker
            }

            if fillNearest {
                return nearestSpeaker(toStart: wordStart, wordEnd: wordEnd, in: segments)
            }

            return nil
        }

        var overlapBySpeaker: [String: Float] = [:]
        for segment in segments {
            let intersection = min(segment.end, wordEnd) - max(segment.start, wordStart)
            if intersection > 0 {
                overlapBySpeaker[segment.speaker, default: 0] += intersection
            }
        }

        if !overlapBySpeaker.isEmpty {
            return overlapBySpeaker.max { lhs, rhs in
                if lhs.value == rhs.value {
                    let leftOrder = speakerOrder[lhs.key, default: .max]
                    let rightOrder = speakerOrder[rhs.key, default: .max]
                    return leftOrder > rightOrder
                }
                return lhs.value < rhs.value
            }?.key
        }

        if fillNearest {
            return nearestSpeaker(toStart: wordStart, wordEnd: wordEnd, in: segments)
        }

        return nil
    }

    private static func smoothSentenceBoundaries(words: inout [DiarizedWord], maxWordsInSentence: Int) {
        guard words.count > 1 else {
            return
        }

        var index = 1
        while index < words.count {
            guard words[index].speaker != words[index - 1].speaker else {
                index += 1
                continue
            }

            if endsSentence(words[index - 1].word) {
                index += 1
                continue
            }

            let start = sentenceStartIndex(
                in: words,
                before: index,
                maxWords: maxWordsInSentence
            )
            let end = sentenceEndIndex(
                in: words,
                from: index,
                maxWords: maxWordsInSentence
            )

            let majoritySpeaker = majoritySpeaker(in: words, start: start, end: end)
            for reassignedIndex in start...end {
                words[reassignedIndex].speaker = majoritySpeaker
            }

            index = end + 1
        }
    }

    private static func sentenceStartIndex(
        in words: [DiarizedWord],
        before changeIndex: Int,
        maxWords: Int
    ) -> Int {
        var start = max(0, changeIndex - 1)
        var cursor = changeIndex - 1
        var steps = 0

        while cursor >= 0, steps < maxWords {
            if endsSentence(words[cursor].word) {
                return min(changeIndex - 1, cursor + 1)
            }

            start = cursor
            cursor -= 1
            steps += 1
        }

        return start
    }

    private static func sentenceEndIndex(
        in words: [DiarizedWord],
        from changeIndex: Int,
        maxWords: Int
    ) -> Int {
        var end = min(words.count - 1, changeIndex)
        var cursor = changeIndex
        var steps = 0

        while cursor < words.count, steps < maxWords {
            end = cursor
            if endsSentence(words[cursor].word) {
                return end
            }

            cursor += 1
            steps += 1
        }

        return end
    }

    private static func majoritySpeaker(in words: [DiarizedWord], start: Int, end: Int) -> String? {
        var counts: [String?: Int] = [:]
        var firstSeenOrder: [String?: Int] = [:]
        var seen = 0

        for index in start...end {
            let speaker = words[index].speaker
            counts[speaker, default: 0] += 1
            if firstSeenOrder[speaker] == nil {
                firstSeenOrder[speaker] = seen
                seen += 1
            }
        }

        return counts.max { lhs, rhs in
            if lhs.value == rhs.value {
                let leftOrder = firstSeenOrder[lhs.key, default: .max]
                let rightOrder = firstSeenOrder[rhs.key, default: .max]
                return leftOrder > rightOrder
            }
            return lhs.value < rhs.value
        }?.key
    }

    private static func groupUtterances(words: [DiarizedWord]) -> [DiarizedUtterance] {
        guard !words.isEmpty else {
            return []
        }

        var utterances: [DiarizedUtterance] = []
        var currentWords: [DiarizedWord] = [words[0]]

        for word in words.dropFirst() {
            if word.speaker == currentWords[0].speaker {
                currentWords.append(word)
                continue
            }

            utterances.append(makeUtterance(from: currentWords))
            currentWords = [word]
        }

        utterances.append(makeUtterance(from: currentWords))
        return utterances
    }

    private static func makeUtterance(from words: [DiarizedWord]) -> DiarizedUtterance {
        let text = words.map(\.word).joined(separator: " ")
        return DiarizedUtterance(
            speaker: words[0].speaker,
            text: text,
            start: words[0].start,
            end: words[words.count - 1].end,
            words: words
        )
    }

    private static func endsSentence(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.last else {
            return false
        }
        return last == "." || last == "?" || last == "!"
    }

    private static func contains(time: Float, in segment: DiarizationSegment) -> Bool {
        time >= segment.start && time <= segment.end
    }

    private static func nearestSpeaker(toStart wordStart: Float, wordEnd: Float, in segments: [DiarizationSegment]) -> String? {
        guard let nearest = segments.min(by: { lhs, rhs in
            let leftDistance = distanceBetween(wordStart: wordStart, wordEnd: wordEnd, segment: lhs)
            let rightDistance = distanceBetween(wordStart: wordStart, wordEnd: wordEnd, segment: rhs)
            if leftDistance == rightDistance {
                return lhs.start < rhs.start
            }
            return leftDistance < rightDistance
        }) else {
            return nil
        }

        return nearest.speaker
    }

    private static func distanceBetween(wordStart: Float, wordEnd: Float, segment: DiarizationSegment) -> Float {
        if wordEnd < segment.start {
            return segment.start - wordEnd
        }
        if segment.end < wordStart {
            return wordStart - segment.end
        }
        return 0
    }

    private static func buildSpeakerOrder(segments: [DiarizationSegment]) -> [String: Int] {
        var order: [String: Int] = [:]
        for (index, segment) in segments.enumerated() where order[segment.speaker] == nil {
            order[segment.speaker] = index
        }
        return order
    }
}
