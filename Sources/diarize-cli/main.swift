import Foundation
import OpenWhisperKit

@main
struct DiarizeCLI {
    static func main() async throws {
        let args = CommandLine.arguments

        guard args.count >= 3 else {
            printUsage()
            Foundation.exit(1)
        }

        let modelPath = args[1]
        let wavFiles = Array(args.dropFirst(2))

        // Load model
        fputs("Loading sortformer model from \(modelPath)...\n", stderr)
        let loadStart = CFAbsoluteTimeGetCurrent()
        let context: SortFormerContext
        do {
            context = try SortFormerContext.createContext(modelPath: modelPath)
        } catch {
            fputs("Error: Failed to load model: \(error)\n", stderr)
            Foundation.exit(1)
        }
        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        fputs("Model loaded in \(String(format: "%.2f", loadTime))s\n\n", stderr)

        // Process each WAV file
        var successCount = 0
        for wavPath in wavFiles {
            fputs("Processing: \(wavPath)\n", stderr)

            do {
                // Load audio
                let samples = try AudioProcessor.loadAudio(fromPath: wavPath)
                let audioDuration = Double(samples.count) / 16000.0
                fputs("  Audio: \(String(format: "%.1f", audioDuration))s, \(samples.count) samples\n", stderr)

                // Run diarization
                let result = try await context.diarize(samples: samples)

                // Determine output path
                let wavURL = URL(fileURLWithPath: wavPath)
                let rttmPath = wavURL.deletingPathExtension().appendingPathExtension("rttm").path

                // Write RTTM file
                try result.rttm.write(toFile: rttmPath, atomically: true, encoding: .utf8)

                // Print summary for this file
                fputs("  Speakers: \(Set(result.segments.map(\.speaker)).count)\n", stderr)
                fputs("  Segments: \(result.segments.count)\n", stderr)
                fputs("  Diarization time: \(String(format: "%.0f", result.timings.diarizationTime))ms\n", stderr)
                fputs("  RTF: \(String(format: "%.3f", result.timings.diarizationTime / 1000.0 / audioDuration))\n", stderr)
                fputs("  Output: \(rttmPath)\n\n", stderr)

                successCount += 1
            } catch {
                fputs("  Error: \(error)\n\n", stderr)
            }
        }

        fputs("Done: \(successCount)/\(wavFiles.count) files processed successfully.\n", stderr)
    }

    static func printUsage() {
        fputs("""
        Usage: diarize-cli <model-path> <wav-file> [wav-file2 ...]

        Arguments:
          model-path    Path to sortformer .gguf model file
                        (CoreML model auto-detected if present alongside)
          wav-file      One or more WAV files to diarize

        Output:
          Generates .rttm files alongside each input WAV file.
          e.g., input.wav → input.rttm

        Example:
          diarize-cli /path/to/model.gguf audio1.wav audio2.wav

        """, stderr)
    }
}
