import Foundation
import XCTest
@testable import OpenWhisperKit

final class DiarizationIntegrationTests: XCTestCase {

    static let testModelPath: String = {
        let base = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let paths = [
            "models/sortformer.gguf",
            "models/sortformer-f16.gguf"
        ]

        for path in paths {
            let fullPath = base.appendingPathComponent(path).path
            if FileManager.default.fileExists(atPath: fullPath) {
                return fullPath
            }
        }

        return ""
    }()

    func testSortFormerContextInit() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.testModelPath),
            "Sortformer model not found - skipping integration test"
        )

        let context = try SortFormerContext.createContext(modelPath: Self.testModelPath)
        _ = context
    }

    func testDiarizeShortAudio() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.testModelPath),
            "Sortformer model not found - skipping integration test"
        )

        let context = try SortFormerContext.createContext(modelPath: Self.testModelPath)
        let samples = [Float](repeating: 0, count: 16_000 * 5)
        let result = try await context.diarize(samples: samples)

        XCTAssertNotNil(result.timings)
    }
}
