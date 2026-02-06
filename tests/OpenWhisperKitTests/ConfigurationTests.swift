import XCTest
@testable import OpenWhisperKit

final class ConfigurationTests: XCTestCase {

    func testComputeOptionsDefaults() {
        let opts = ComputeOptions()
        XCTAssertTrue(opts.useGPU)
        XCTAssertTrue(opts.useCoreML)
        XCTAssertTrue(opts.flashAttention)
        XCTAssertNil(opts.threadCount)
    }

    func testDecodingOptionsDefaults() {
        let opts = DecodingOptions()
        XCTAssertEqual(opts.task, .transcribe)
        XCTAssertNil(opts.language)
        XCTAssertEqual(opts.temperature, 0.0)
        XCTAssertFalse(opts.withoutTimestamps)
        XCTAssertFalse(opts.wordTimestamps)
        XCTAssertTrue(opts.suppressBlank)
        XCTAssertEqual(opts.chunkingStrategy, .none)
    }

    func testDecodingOptionsCustom() {
        let opts = DecodingOptions(
            task: .translate,
            language: "es",
            temperature: 0.5,
            wordTimestamps: true,
            chunkingStrategy: .vad
        )

        XCTAssertEqual(opts.task, .translate)
        XCTAssertEqual(opts.language, "es")
        XCTAssertEqual(opts.temperature, 0.5)
        XCTAssertTrue(opts.wordTimestamps)
        XCTAssertEqual(opts.chunkingStrategy, .vad)
    }

    func testOpenWhisperKitConfigCreation() {
        let config = OpenWhisperKitConfig(modelPath: "/path/to/model.bin")
        XCTAssertEqual(config.modelPath, "/path/to/model.bin")
        XCTAssertNil(config.vadModelPath)
    }

    func testDecodingTaskValues() {
        XCTAssertEqual(DecodingTask.transcribe.rawValue, "transcribe")
        XCTAssertEqual(DecodingTask.translate.rawValue, "translate")
    }
}
