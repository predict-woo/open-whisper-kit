import XCTest
@testable import OpenWhisperKit

final class RTTMParserTests: XCTestCase {

    func testParseEmpty() {
        let segments = RTTMParser.parse("")
        XCTAssertTrue(segments.isEmpty)
    }

    func testParseSingleLine() {
        let rttm = "SPEAKER file 1 0.500 1.200 <NA> <NA> speaker_0 <NA> <NA>"
        let segments = RTTMParser.parse(rttm)

        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].speaker, "speaker_0")
        XCTAssertEqual(segments[0].start, 0.5, accuracy: 0.001)
        XCTAssertEqual(segments[0].end, 1.7, accuracy: 0.001)
    }

    func testParseMultiLine() {
        let rttm = [
            "SPEAKER file 1 1.000 0.500 <NA> <NA> speaker_1 <NA> <NA>",
            "SPEAKER file 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>"
        ].joined(separator: "\n")

        let segments = RTTMParser.parse(rttm)
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].speaker, "speaker_0")
        XCTAssertEqual(segments[1].speaker, "speaker_1")
    }

    func testParseMalformedLine() {
        let malformed = "SPEAKER file 1 bad 1.000 <NA> <NA> speaker_0 <NA> <NA>"
        let segments = RTTMParser.parse(malformed)
        XCTAssertTrue(segments.isEmpty)
    }

    func testParseSkipsMalformed() {
        let rttm = [
            "SPEAKER file 1 0.000 1.000 <NA> <NA> speaker_0 <NA> <NA>",
            "SPEAKER file 1 BAD 0.500 <NA> <NA> speaker_bad <NA> <NA>",
            "SPEAKER file 1 1.200 0.800 <NA> <NA> speaker_1 <NA> <NA>"
        ].joined(separator: "\n")

        let segments = RTTMParser.parse(rttm)
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0].speaker, "speaker_0")
        XCTAssertEqual(segments[1].speaker, "speaker_1")
    }

    func testGenerateRTTM() {
        let segments = [
            DiarizationSegment(speaker: "speaker_0", start: 0.0, end: 1.25),
            DiarizationSegment(speaker: "speaker_1", start: 1.25, end: 2.0)
        ]

        let rttm = RTTMParser.generate(segments: segments, filename: "audio")
        XCTAssertTrue(rttm.contains("SPEAKER audio 1 0.00 1.25 <NA> <NA> speaker_0 <NA> <NA>"))
        XCTAssertTrue(rttm.contains("SPEAKER audio 1 1.25 0.75 <NA> <NA> speaker_1 <NA> <NA>"))
    }

    func testRoundTrip() {
        let input = [
            DiarizationSegment(speaker: "speaker_0", start: 0.0, end: 1.2),
            DiarizationSegment(speaker: "speaker_1", start: 1.2, end: 2.35)
        ]

        let rttm = RTTMParser.generate(segments: input, filename: "sample")
        let output = RTTMParser.parse(rttm)

        XCTAssertEqual(output.count, input.count)
        XCTAssertEqual(output[0].speaker, input[0].speaker)
        XCTAssertEqual(output[1].speaker, input[1].speaker)
        XCTAssertEqual(output[0].start, input[0].start, accuracy: 0.01)
        XCTAssertEqual(output[0].end, input[0].end, accuracy: 0.01)
        XCTAssertEqual(output[1].start, input[1].start, accuracy: 0.01)
        XCTAssertEqual(output[1].end, input[1].end, accuracy: 0.01)
    }

    func testSortedByStartTime() {
        let rttm = [
            "SPEAKER file 1 2.000 0.500 <NA> <NA> speaker_2 <NA> <NA>",
            "SPEAKER file 1 0.000 0.500 <NA> <NA> speaker_0 <NA> <NA>",
            "SPEAKER file 1 1.000 0.500 <NA> <NA> speaker_1 <NA> <NA>"
        ].joined(separator: "\n")

        let segments = RTTMParser.parse(rttm)
        XCTAssertEqual(segments.count, 3)
        XCTAssertEqual(segments.map(\.speaker), ["speaker_0", "speaker_1", "speaker_2"])
    }
}
