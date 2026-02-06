import XCTest
@testable import OpenWhisperKit

final class DiarizationAlignerTests: XCTestCase {

    private func makeWord(_ text: String, start: Float, end: Float, prob: Float = 0.9) -> WordTiming {
        WordTiming(word: text, tokens: [1], start: start, end: end, probability: prob)
    }

    private func makeSeg(_ speaker: String, start: Float, end: Float) -> DiarizationSegment {
        DiarizationSegment(speaker: speaker, start: start, end: end)
    }

    func testBasicAlignment() throws {
        let words = [
            makeWord("hello", start: 0.1, end: 0.5),
            makeWord("world", start: 0.6, end: 1.0)
        ]
        let segs = [makeSeg("speaker_0", start: 0.0, end: 2.0)]

        let result = try DiarizationAligner.align(words: words, diarizationSegments: segs)
        XCTAssertEqual(result.words[0].speaker, "speaker_0")
        XCTAssertEqual(result.words[1].speaker, "speaker_0")
    }

    func testBoundarySpanning() throws {
        let words = [makeWord("crossing", start: 0.8, end: 1.4)]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 1.0),
            makeSeg("speaker_1", start: 1.0, end: 2.0)
        ]

        let result = try DiarizationAligner.align(words: words, diarizationSegments: segs)
        XCTAssertEqual(result.words[0].speaker, "speaker_1")
    }

    func testZeroDurationWord() throws {
        let words = [makeWord(",", start: 1.5, end: 1.5)]
        let segs = [makeSeg("speaker_0", start: 0.0, end: 2.0)]

        let result = try DiarizationAligner.align(words: words, diarizationSegments: segs)
        XCTAssertEqual(result.words[0].speaker, "speaker_0")
    }

    func testNoMatchingSegment() throws {
        let words = [makeWord("orphan", start: 5.0, end: 6.0)]
        let segs = [makeSeg("speaker_0", start: 0.0, end: 2.0)]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(fillNearest: false)
        )
        XCTAssertNil(result.words[0].speaker)
    }

    func testFillNearest() throws {
        let words = [makeWord("gap", start: 3.0, end: 4.0)]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 2.0),
            makeSeg("speaker_1", start: 5.0, end: 7.0)
        ]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(fillNearest: true)
        )
        XCTAssertEqual(result.words[0].speaker, "speaker_0")
    }

    func testEmptyWords() throws {
        let result = try DiarizationAligner.align(
            words: [],
            diarizationSegments: [makeSeg("s0", start: 0, end: 1)]
        )

        XCTAssertTrue(result.words.isEmpty)
        XCTAssertTrue(result.segments.isEmpty)
        XCTAssertEqual(result.text, "")
    }

    func testEmptyDiarizationSegments() throws {
        let words = [makeWord("alone", start: 0, end: 1)]
        let result = try DiarizationAligner.align(words: words, diarizationSegments: [])
        XCTAssertNil(result.words[0].speaker)
    }

    func testSentenceSmoothing() throws {
        let words = [
            makeWord("Hello", start: 0.0, end: 0.5),
            makeWord("world,", start: 0.5, end: 1.0),
            makeWord("how", start: 1.0, end: 1.5),
            makeWord("are", start: 1.5, end: 2.0),
            makeWord("you?", start: 2.0, end: 2.5)
        ]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 1.0),
            makeSeg("speaker_1", start: 1.0, end: 2.0),
            makeSeg("speaker_0", start: 2.0, end: 3.0)
        ]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(sentenceSmoothing: true)
        )

        let uniqueSpeakers = Set(result.words.compactMap { $0.speaker })
        XCTAssertEqual(uniqueSpeakers.count, 1)
    }

    func testSentenceSmoothingDisabled() throws {
        let words = [
            makeWord("Hello", start: 0.0, end: 0.5),
            makeWord("world", start: 0.5, end: 1.0)
        ]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 0.6),
            makeSeg("speaker_1", start: 0.6, end: 1.5)
        ]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(sentenceSmoothing: false)
        )

        XCTAssertEqual(result.words[0].speaker, "speaker_0")
        XCTAssertEqual(result.words[1].speaker, "speaker_1")
    }

    func testUtteranceGrouping() throws {
        let words = [
            makeWord("Hello.", start: 0.0, end: 1.0),
            makeWord("Hi.", start: 1.0, end: 2.0)
        ]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 1.0),
            makeSeg("speaker_1", start: 1.0, end: 2.0)
        ]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(sentenceSmoothing: false)
        )

        XCTAssertEqual(result.segments.count, 2)
        XCTAssertEqual(result.segments[0].speaker, "speaker_0")
        XCTAssertEqual(result.segments[1].speaker, "speaker_1")
    }

    func testTextGeneration() throws {
        let words = [
            makeWord("Hello.", start: 0.0, end: 1.0),
            makeWord("Hi.", start: 1.0, end: 2.0)
        ]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 1.0),
            makeSeg("speaker_1", start: 1.0, end: 2.0)
        ]

        let result = try DiarizationAligner.align(
            words: words,
            diarizationSegments: segs,
            options: .init(sentenceSmoothing: false)
        )

        XCTAssertTrue(result.text.contains("speaker_0"))
        XCTAssertTrue(result.text.contains("speaker_1"))
    }

    func testTieBreakingUsesEarlierSpeakerOrder() throws {
        let words = [makeWord("equal", start: 0.5, end: 1.5)]
        let segs = [
            makeSeg("speaker_0", start: 0.0, end: 1.0),
            makeSeg("speaker_1", start: 1.0, end: 2.0)
        ]

        let result = try DiarizationAligner.align(words: words, diarizationSegments: segs)
        XCTAssertEqual(result.words[0].speaker, "speaker_0")
    }

    func testMaxWordsInSentenceValidation() {
        XCTAssertThrowsError(
            try DiarizationAligner.align(
                words: [makeWord("hello", start: 0, end: 1)],
                diarizationSegments: [makeSeg("speaker_0", start: 0, end: 2)],
                options: .init(maxWordsInSentence: 0)
            )
        ) { error in
            guard case DiarizationError.alignmentFailed(let message) = error else {
                XCTFail("Expected alignmentFailed error")
                return
            }
            XCTAssertTrue(message.contains("maxWordsInSentence"))
        }
    }
}
