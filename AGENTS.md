# OpenWhisperKit

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Swift SDK for on-device speech transcription and speaker diarization on Apple platforms. OpenWhisperKit combines whisper.cpp for fast speech-to-text with streaming-sortformer for speaker diarization, delivering a unified package for iOS and macOS applications.

## What is OpenWhisperKit?

OpenWhisperKit is a Swift SDK that wraps [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and streaming-sortformer into a unified package for speech transcription and speaker diarization on Apple platforms (iOS, macOS). Built on top of whisper.cpp (upstream: ggml-org/whisper.cpp) with additional diarization capabilities, it's designed for on-device, offline inference with CoreML and Metal acceleration.

## Key Features

- **Speech-to-text**: Powered by whisper.cpp with CoreML encoder acceleration
- **Speaker diarization**: Powered by streaming-sortformer with CoreML head acceleration
- **Transcription alignment**: Combines word-level timestamps with diarization to produce speaker-attributed transcripts
- **RTTM support**: Parse and generate RTTM (Rich Transcription Time Marked) format
- **Apple-optimized**: CoreML, Metal, ANE acceleration on iOS and macOS
- **Swift-native**: Actor-based concurrency, Sendable types, Swift Package Manager

## Architecture Overview

- `OpenWhisperKit` — Main SDK class for transcription (wraps whisper.cpp)
- `SortFormerContext` — Actor for speaker diarization (wraps streaming-sortformer)
- `DiarizationAligner` — Aligns word timestamps with speaker segments
- `RTTMParser` — RTTM format parser/generator
- `AudioProcessor` — Audio loading and preprocessing

## Installation

### Swift Package Manager

Add the repository URL to your Swift package dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/predict-woo/open-whisper-kit", from: "1.0.0")
]
```

### Prebuilt XCFrameworks

Prebuilt xcframeworks are available in `build-apple/`:
- `whisper.xcframework` — whisper.cpp inference engine
- `sortformer.xcframework` — streaming-sortformer diarization engine

## Quick Start

### Transcription

```swift
import OpenWhisperKit

// Initialize with model path
let config = OpenWhisperKitConfig(modelPath: "/path/to/ggml-model.bin")
let whisperKit = try await OpenWhisperKit(config)

// Transcribe audio with word timestamps
let result = try await whisperKit.transcribe(
    audioPath: "audio.wav",
    options: DecodingOptions(wordTimestamps: true)
)
print(result.text)
```

### Diarization

```swift
import OpenWhisperKit

// Create diarization context
let sortformer = try SortFormerContext.createContext(modelPath: "/path/to/model.gguf")

// Load audio samples
let samples = try AudioProcessor.loadAudio(fromPath: "audio.wav")

// Run diarization
let diarization = try await sortformer.diarize(samples: samples)

// Print RTTM output
print(diarization.rttm)
```

### Alignment (Transcription + Diarization)

```swift
import OpenWhisperKit

// Transcribe with word timestamps
let config = OpenWhisperKitConfig(modelPath: "/path/to/ggml-model.bin")
let whisperKit = try await OpenWhisperKit(config)
let result = try await whisperKit.transcribe(
    audioPath: "audio.wav",
    options: DecodingOptions(wordTimestamps: true)
)

// Diarize
let sortformer = try SortFormerContext.createContext(modelPath: "/path/to/model.gguf")
let samples = try AudioProcessor.loadAudio(fromPath: "audio.wav")
let diarization = try await sortformer.diarize(samples: samples)

// Align words with speaker segments
let aligned = try DiarizationAligner.align(
    words: result.segments.flatMap { $0.words ?? [] },
    diarizationSegments: diarization.segments
)

// Print speaker-attributed transcript
for utterance in aligned.segments {
    print("[\(utterance.speaker ?? "?")] \(utterance.text)")
}
```

## CLI Tool

The `diarize-cli` tool provides command-line diarization:

```bash
# Build the CLI
swift build --product diarize-cli

# Run diarization
.build/debug/diarize-cli /path/to/model.gguf audio.wav

# Output: audio.rttm
```

## Demo Apps

### iOS Demo App

Located in `examples/OpenWhisperKitDemoiOS/`:
- Model download from HuggingFace
- Real-time transcription
- Speaker diarization
- Speaker-labeled UI

### macOS Demo App

Located in `examples/OpenWhisperKitDemo/`:
- Model download from HuggingFace
- Batch transcription
- Speaker diarization
- Speaker-labeled UI

## Models

### Whisper Models

Download from HuggingFace ([ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)):

**GGML model** (required):
```bash
# Example: large-v3-turbo quantized
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
```

**CoreML encoder** (optional, for acceleration):
```bash
# Example: large-v3-turbo encoder
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-encoder.mlmodelc.zip
unzip ggml-large-v3-turbo-encoder.mlmodelc.zip
```

### Sortformer Models

Download from HuggingFace ([andyye/streaming-sortformer](https://huggingface.co/andyye/streaming-sortformer)):

**GGUF model** (required):
```bash
wget https://huggingface.co/andyye/streaming-sortformer/resolve/main/model.gguf
```

**CoreML head** (optional, for acceleration):
```bash
wget https://huggingface.co/andyye/streaming-sortformer/resolve/main/model-coreml-head.mlmodelc.zip
unzip model-coreml-head.mlmodelc.zip
```

## Building from Source

### Build Whisper XCFramework

```bash
bash build-xcframework.sh
```

### Build Sortformer XCFramework

```bash
bash build-sortformer-xcframework.sh
```

### Build and Test

```bash
# Build the package
swift build

# Run tests
swift test
```

## Releasing a New SDK Version

A GitHub Actions workflow (`.github/workflows/release-xcframeworks.yml`) automates the entire release process. It builds both xcframeworks from source, zips them, uploads them as release assets, and updates `Package.swift` with the new download URLs and checksums.

### How to Release

1. Create a new release on GitHub with a semver tag:
   ```bash
   gh release create v1.1.0 --title "v1.1.0" --notes "Release notes here" --target master
   ```
2. The workflow runs automatically and:
   - Builds `whisper.xcframework` and `sortformer.xcframework` on a macOS runner
   - Zips and uploads them as release assets
   - Computes Swift package checksums
   - Commits the updated `Package.swift` to `master`
3. That's it. Consumers pulling the package will get the new version.

### Notes

- The tag must follow semver (e.g. `v1.0.0`, `v1.1.0`, `v2.0.0`)
- The workflow only triggers on **release creation** — drafts won't trigger it
- Build checksums are printed in the workflow's job summary for verification
- If the build fails, delete the release, fix the issue, and create a new one

## Upstream

OpenWhisperKit is based on whisper.cpp from [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp).

To fetch upstream updates:

```bash
git fetch upstream
git merge upstream/master
```

## License

MIT License (same as whisper.cpp)
