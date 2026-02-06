import XCTest
@testable import OpenWhisperKit

final class ModelsTests: XCTestCase {

    func testTranscriptionSegmentDuration() {
        let segment = TranscriptionSegment(
            id: 0,
            seek: 0,
            start: 1.0,
            end: 3.5,
            text: "hello",
            tokens: [1, 2],
            temperature: 0,
            avgLogprob: 0,
            compressionRatio: 0,
            noSpeechProb: 0
        )

        XCTAssertEqual(segment.duration, 2.5, accuracy: 0.001)
    }

    func testWordTimingDuration() {
        let word = WordTiming(word: "hello", tokens: [1], start: 0.5, end: 1.2, probability: 0.9)
        XCTAssertEqual(word.duration, 0.7, accuracy: 0.001)
    }

    func testModelVariantMultilingual() {
        XCTAssertTrue(ModelVariant.tiny.isMultilingual)
        XCTAssertFalse(ModelVariant.tinyEn.isMultilingual)
        XCTAssertTrue(ModelVariant.base.isMultilingual)
        XCTAssertFalse(ModelVariant.baseEn.isMultilingual)
        XCTAssertTrue(ModelVariant.largeV3.isMultilingual)
        XCTAssertTrue(ModelVariant.largeV3Turbo.isMultilingual)
    }

    func testModelStateValues() {
        XCTAssertEqual(ModelState.unloaded.rawValue, "unloaded")
        XCTAssertEqual(ModelState.loading.rawValue, "loading")
        XCTAssertEqual(ModelState.loaded.rawValue, "loaded")
    }

    func testTranscriptionResultCreation() {
        let result = TranscriptionResult(
            text: "hello world",
            segments: [],
            language: "en",
            timings: TranscriptionTimings(
                inputAudioSeconds: 5.0,
                modelLoadingTime: 0,
                encodingTime: 0.1,
                decodingTime: 0.2,
                fullPipeline: 0.3
            )
        )

        XCTAssertEqual(result.text, "hello world")
        XCTAssertEqual(result.language, "en")
        XCTAssertTrue(result.segments.isEmpty)
    }

    func testTranscriptionTimingsComputed() {
        let timings = TranscriptionTimings(
            inputAudioSeconds: 5.0,
            modelLoadingTime: 100,
            encodingTime: 250,
            decodingTime: 250,
            fullPipeline: 1000
        )

        XCTAssertEqual(timings.tokensPerSecond, 4.0, accuracy: 0.0001)
        XCTAssertEqual(timings.realTimeFactor, 0.2, accuracy: 0.0001)
        XCTAssertEqual(timings.speedFactor, 5.0, accuracy: 0.0001)

        let zeroTimings = TranscriptionTimings(
            inputAudioSeconds: 0,
            modelLoadingTime: 0,
            encodingTime: 0,
            decodingTime: 0,
            fullPipeline: 0
        )

        XCTAssertEqual(zeroTimings.tokensPerSecond, 0)
        XCTAssertEqual(zeroTimings.realTimeFactor, 0)
        XCTAssertEqual(zeroTimings.speedFactor, 0)
    }

    func testSpeechSegmentCreation() {
        let seg = SpeechSegment(startTime: 1.0, endTime: 3.0, startSample: 16000, endSample: 48000)
        XCTAssertEqual(seg.startTime, 1.0)
        XCTAssertEqual(seg.endTime, 3.0)
        XCTAssertEqual(seg.startSample, 16000)
        XCTAssertEqual(seg.endSample, 48000)
    }

    func testTranscriptionProgressCreation() {
        let progress = TranscriptionProgress(progress: 0.5, text: "partial", windowId: 0)
        XCTAssertEqual(progress.progress, 0.5)
        XCTAssertEqual(progress.text, "partial")
        XCTAssertEqual(progress.windowId, 0)
    }
}
