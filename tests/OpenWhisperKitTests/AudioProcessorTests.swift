import Foundation
import XCTest
@testable import OpenWhisperKit

final class AudioProcessorTests: XCTestCase {

    func testLoadAudioNonexistentFile() {
        XCTAssertThrowsError(try AudioProcessor.loadAudio(fromPath: "/nonexistent/path.wav")) { error in
            guard case WhisperError.audioLoadFailed(_) = error else {
                XCTFail("Expected WhisperError.audioLoadFailed, got \(error)")
                return
            }
        }
    }

    func testSampleRateConstant() {
        XCTAssertEqual(AudioProcessor.sampleRate, 16000.0)
    }

    func testLoadAudioFromExistingSample() throws {
        let samplePath = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("samples/jfk.wav")
            .path

        try XCTSkipUnless(FileManager.default.fileExists(atPath: samplePath), "Sample audio not found")

        let samples = try AudioProcessor.loadAudio(fromPath: samplePath)
        XCTAssertFalse(samples.isEmpty)
        XCTAssertGreaterThan(samples.count, 100000)
    }
}
