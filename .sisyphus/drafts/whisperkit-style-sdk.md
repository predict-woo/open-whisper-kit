# Draft: WhisperKit-Style SDK for whisper.cpp XCFramework

## Requirements (confirmed)
- Build a Swift SDK on top of whisper.cpp's existing XCFramework
- Follow WhisperKit's API architecture (protocol-oriented, modern async/await)
- Use whisper.cpp's C engine underneath (NOT CoreML models for decoder — key speed advantage)
- CoreML acceleration for encoder via whisper.cpp's existing CoreML support
- Metal acceleration via whisper.cpp's existing GGML Metal backend
- Exclude real-time streaming (user says WhisperKit's implementation is inefficient)

## Technical Decisions
- **Engine**: whisper.cpp C library via XCFramework (fast GGML decoder + optional CoreML encoder)
- **API Surface**: Inspired by WhisperKit — similar ease of use and API patterns, but NOT a 1:1 clone. Freedom to adapt to what makes sense for whisper.cpp backend.
- **Distribution**: Swift Package wrapping the existing whisper.xcframework

## Research Findings
- whisper.cpp already has: XCFramework build script, CoreML encoder support, Metal GPU, existing Swift wrapper (LibWhisper.swift)
- WhisperKit has: Protocol-oriented architecture, rich configuration, transcription results with segments/words/timings
- Key WhisperKit features to replicate: transcribe(audioPath:), transcribe(audioArray:), language detection, VAD, word timestamps, batch transcription, DecodingOptions, TranscriptionResult, progress callbacks, model management

## Features IN Scope
- Full transcription pipeline (file and array input)
- Batch transcription
- Language detection
- Voice Activity Detection (VAD) — whisper.cpp has Silero VAD
- Word-level timestamps
- Segment-level timestamps
- DecodingOptions (temperature, language, task, thresholds, etc.)
- TranscriptionResult / TranscriptionSegment / WordTiming types
- Progress callbacks and early stopping
- Model loading from local path
- Model state tracking
- Transcription timings
- CoreML encoder acceleration
- Metal GPU acceleration
- Protocol-based architecture for testability

## Features OUT of Scope (Explicit Exclusions)
- Real-time streaming / live microphone transcription (user excluded this)
- Model downloading from HuggingFace (whisper.cpp uses .bin format, not CoreML .mlmodelc)
- CoreML model format (uses GGML .bin format)
- WhisperKit's model recommendation system (device-specific model selection)
- Prefill cache (CoreML-specific optimization)

## Decisions Made
- **SDK Name**: `OpenWhisperKit` — `import OpenWhisperKit`
- **Location**: Top-level `Sources/` with `Package.swift` at repo root
- **Platforms**: iOS + macOS
- **Model Management**: Local path only — developer bundles or downloads models themselves
- **VAD**: Yes — wrap whisper.cpp's Silero VAD for skip-silence optimization
- **API Philosophy**: Inspired by WhisperKit, not a 1:1 clone. Adapt to whisper.cpp's strengths.

## Open Questions
- Minimum deployment target — iOS 16.4 / macOS 13.3 (matching XCFramework)?
- Test strategy?

## Scope Boundaries
- INCLUDE: Swift SDK layer, all WhisperKit-equivalent features (minus streaming)
- EXCLUDE: Changes to whisper.cpp C code, XCFramework build modifications, real-time streaming
