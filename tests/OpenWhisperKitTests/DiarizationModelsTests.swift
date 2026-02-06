import XCTest
@testable import OpenWhisperKit

final class DiarizationModelsTests: XCTestCase {

    func testDiarizationSegmentDuration() {
        let segment = DiarizationSegment(speaker: "speaker_0", start: 1.5, end: 4.25)
        XCTAssertEqual(segment.duration, 2.75, accuracy: 0.001)
    }

    func testDiarizationResultInit() {
        let timings = DiarizationTimings(
            inputAudioSeconds: 3.0,
            modelLoadingTime: 10,
            diarizationTime: 25,
            fullPipeline: 35
        )
        let segment = DiarizationSegment(speaker: "speaker_0", start: 0, end: 1)
        let result = DiarizationResult(
            segments: [segment],
            rttm: "SPEAKER file 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>",
            frameProbabilities: [[0.9, 0.1, 0, 0]],
            timings: timings
        )

        XCTAssertEqual(result.segments.count, 1)
        XCTAssertEqual(result.segments[0].speaker, "speaker_0")
        XCTAssertEqual(result.rttm, "SPEAKER file 1 0.00 1.00 <NA> <NA> speaker_0 <NA> <NA>")
        XCTAssertEqual(result.frameProbabilities.count, 1)
        XCTAssertEqual(result.timings.fullPipeline, 35, accuracy: 0.001)
    }

    func testDiarizedWordInit() {
        let word = DiarizedWord(word: "hello", start: 0.2, end: 0.7, speaker: "speaker_1", probability: 0.88)
        XCTAssertEqual(word.word, "hello")
        XCTAssertEqual(word.speaker, "speaker_1")
        XCTAssertEqual(word.duration, 0.5, accuracy: 0.001)
    }

    func testDiarizedUtteranceInit() {
        let word = DiarizedWord(word: "hi", start: 0, end: 0.4, speaker: "speaker_0", probability: 0.9)
        let utterance = DiarizedUtterance(
            speaker: "speaker_0",
            text: "hi",
            start: 0,
            end: 0.4,
            words: [word]
        )

        XCTAssertEqual(utterance.speaker, "speaker_0")
        XCTAssertEqual(utterance.text, "hi")
        XCTAssertEqual(utterance.words.count, 1)
        XCTAssertEqual(utterance.duration, 0.4, accuracy: 0.001)
    }

    func testDiarizationTimingsInit() {
        let timings = DiarizationTimings(
            inputAudioSeconds: 12.5,
            modelLoadingTime: 100,
            diarizationTime: 450,
            fullPipeline: 550
        )

        XCTAssertEqual(timings.inputAudioSeconds, 12.5, accuracy: 0.001)
        XCTAssertEqual(timings.modelLoadingTime, 100, accuracy: 0.001)
        XCTAssertEqual(timings.diarizationTime, 450, accuracy: 0.001)
        XCTAssertEqual(timings.fullPipeline, 550, accuracy: 0.001)
    }

    func testDiarizedTranscriptionInit() {
        let words = [DiarizedWord(word: "hello", start: 0, end: 0.5, speaker: "speaker_0", probability: 0.95)]
        let segments = [
            DiarizedUtterance(
                speaker: "speaker_0",
                text: "hello",
                start: 0,
                end: 0.5,
                words: words
            )
        ]
        let transcription = DiarizedTranscription(words: words, segments: segments, text: "[speaker_0]: hello")

        XCTAssertEqual(transcription.words.count, 1)
        XCTAssertEqual(transcription.segments.count, 1)
        XCTAssertEqual(transcription.text, "[speaker_0]: hello")
    }
}
