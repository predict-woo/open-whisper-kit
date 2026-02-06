import Foundation
import XCTest
@testable import OpenWhisperKit

final class IntegrationTests: XCTestCase {

    static let testModelPath: String = {
        let paths = [
            "models/ggml-base.en.bin",
            "models/ggml-tiny.en.bin",
            "models/ggml-tiny.bin"
        ]

        let base = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        for path in paths {
            let fullPath = base.appendingPathComponent(path).path
            if FileManager.default.fileExists(atPath: fullPath) {
                return fullPath
            }
        }

        return ""
    }()

    static let testAudioPath: String = {
        URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("samples/jfk.wav")
            .path
    }()

    func testInitialization() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testModelPath), "Model not found")

        let kit = try await OpenWhisperKit(modelPath: Self.testModelPath)
        XCTAssertEqual(kit.modelState, .loaded)
    }

    func testTranscribeAudioFile() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testModelPath), "Model not found")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testAudioPath), "Audio not found")

        let kit = try await OpenWhisperKit(modelPath: Self.testModelPath)
        let result = try await kit.transcribe(audioPath: Self.testAudioPath)

        XCTAssertFalse(result.text.isEmpty)
        XCTAssertFalse(result.segments.isEmpty)
    }

    func testProgressCallback() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testModelPath), "Model not found")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testAudioPath), "Audio not found")

        let kit = try await OpenWhisperKit(modelPath: Self.testModelPath)
        let progressExpectation = expectation(description: "Progress callback fired")

        let result = try await kit.transcribe(audioPath: Self.testAudioPath) { _ in
            progressExpectation.fulfill()
            return true
        }

        await fulfillment(of: [progressExpectation], timeout: 2.0)
        XCTAssertFalse(result.text.isEmpty)
    }

    func testModelNotFound() async {
        do {
            _ = try await OpenWhisperKit(modelPath: "/nonexistent/model.bin")
            XCTFail("Should have thrown")
        } catch {
            guard case WhisperError.modelNotFound(_) = error else {
                XCTFail("Expected modelNotFound, got \(error)")
                return
            }
        }
    }

    func testDetectLanguage() async throws {
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testModelPath), "Model not found")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.testAudioPath), "Audio not found")

        let kit = try await OpenWhisperKit(modelPath: Self.testModelPath)
        let (language, probabilities) = try await kit.detectLanguage(audioPath: Self.testAudioPath)

        XCTAssertEqual(language, "en")
        XCTAssertFalse(probabilities.isEmpty)
    }
}
