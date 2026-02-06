# OpenWhisperKit — Swift SDK for whisper.cpp

## TL;DR

> **Quick Summary**: Build `OpenWhisperKit`, a modern Swift SDK on top of whisper.cpp's XCFramework that provides a WhisperKit-inspired API with whisper.cpp's fast inference (GGML decoder + Metal + CoreML encoder). The SDK wraps the C API with Swift async/await, rich result types, VAD support, and protocol-oriented design.
>
> **Deliverables**:
> - `Package.swift` at repo root with binary target + Swift wrapper target + test target
> - `Sources/OpenWhisperKit/` — complete Swift SDK (~12-15 source files)
> - `Tests/OpenWhisperKitTests/` — XCTest suite
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (Package.swift) → Task 2 (Core types) → Tasks 3-6 (parallel features) → Task 7 (tests)

---

## Context

### Original Request
Build a Swift SDK for whisper.cpp's XCFramework that follows WhisperKit's API patterns (not a 1:1 clone) for ease of use. Must include CoreML + Metal acceleration. Exclude real-time streaming. Include VAD.

### Interview Summary
**Key Discussions**:
- **SDK Name**: `OpenWhisperKit` — `import OpenWhisperKit`
- **Location**: Top-level `Sources/` with `Package.swift` at repo root
- **Platforms**: iOS 16.4+ and macOS 13.3+ (matching XCFramework minimums)
- **Model Management**: Local path only — developer provides path to .ggml/.bin model
- **VAD**: Yes — wrap whisper.cpp's Silero VAD
- **API Philosophy**: Inspired by WhisperKit, not a clone. Adapt to whisper.cpp's strengths.
- **Tests**: XCTest with test targets

**Research Findings**:
- whisper.cpp C API has 100+ functions: `whisper_full()`, `whisper_full_parallel()`, segment/token iteration, timing info, language detection, VAD
- Existing Swift wrapper (`LibWhisper.swift`) is a minimal actor with `fullTranscribe()` and `getTranscription()` — good foundation pattern
- WhisperKit uses protocol-oriented design with rich result types (`TranscriptionResult`, `TranscriptionSegment`, `WordTiming`)
- XCFramework build already includes CoreML encoder + Metal GPU + Accelerate BLAS
- whisper.cpp has built-in Silero VAD via C API (`whisper_vad_*` functions)
- Key C callbacks to bridge: `whisper_new_segment_callback`, `whisper_progress_callback`, `whisper_encoder_begin_callback`

### Metis Review
**Identified Gaps** (addressed):
- **Thread safety model**: Resolved → Use actor pattern (proven in existing LibWhisper.swift)
- **Callback bridging complexity**: Resolved → Start with progress + new_segment callbacks; use `@convention(c)` with context pointer bridging
- **XCFramework SPM integration**: Resolved → Binary target with local path; wrapper target depends on it
- **Memory safety across await**: Resolved → Copy data into Swift arrays before await; never capture C pointers across suspension points
- **Audio format responsibility**: Resolved → SDK accepts Float32 PCM @ 16kHz; caller handles conversion. Provide helper utility for file loading via AVFoundation.
- **Protocol scope**: Resolved → Lean initial set (not replicating WhisperKit's 7 protocols — whisper.cpp handles the pipeline internally)

---

## Work Objectives

### Core Objective
Create `OpenWhisperKit`, a Swift Package that wraps whisper.cpp's XCFramework with a modern, ergonomic Swift API inspired by WhisperKit — delivering whisper.cpp's fast inference with Swift-native types, async/await, and protocol-oriented extensibility.

### Concrete Deliverables
- `Package.swift` — Swift Package definition with binary target, wrapper target, test target
- `Sources/OpenWhisperKit/OpenWhisperKit.swift` — Main public class
- `Sources/OpenWhisperKit/Configuration.swift` — `OpenWhisperKitConfig`, `DecodingOptions`
- `Sources/OpenWhisperKit/Models.swift` — `TranscriptionResult`, `TranscriptionSegment`, `WordTiming`, `TranscriptionTimings`
- `Sources/OpenWhisperKit/Protocols.swift` — `Transcribing`, `VADProcessing` protocols
- `Sources/OpenWhisperKit/WhisperContext.swift` — Actor wrapping the C API context
- `Sources/OpenWhisperKit/AudioProcessor.swift` — Audio file loading + PCM conversion utility
- `Sources/OpenWhisperKit/VADProcessor.swift` — Silero VAD wrapper
- `Sources/OpenWhisperKit/CallbackBridge.swift` — C-to-Swift callback bridging
- `Sources/OpenWhisperKit/WhisperError.swift` — Error types
- `Sources/OpenWhisperKit/TranscriptionTask.swift` — Orchestrates the full transcription pipeline
- `Tests/OpenWhisperKitTests/` — XCTest suite

### Definition of Done
- [ ] `swift build` succeeds for iOS and macOS targets
- [ ] Test target compiles and runs basic tests
- [ ] Can transcribe an audio file from path → `TranscriptionResult` with segments and timestamps
- [ ] Can transcribe from `[Float]` array → `TranscriptionResult`
- [ ] Language detection returns detected language
- [ ] Word-level timestamps available in results
- [ ] VAD pre-filters audio to skip silence
- [ ] Progress callbacks fire during transcription
- [ ] Early stopping works via callback return value
- [ ] CoreML encoder acceleration works when .mlmodelc present
- [ ] Metal GPU acceleration enabled by default on device

### Must Have
- Modern Swift async/await API
- Actor-based thread safety for the C context
- Rich result types (segments, words, timings)
- DecodingOptions for temperature, language, task, thresholds
- Progress reporting with early stopping
- VAD integration
- Audio file loading utility (wav, mp3, m4a via AVFoundation)
- CoreML + Metal acceleration passthrough
- Proper memory management (C context lifecycle)

### Must NOT Have (Guardrails)
- No real-time streaming or live microphone capture
- No model downloading or HuggingFace integration
- No CoreML model format support (GGML only)
- No changes to the whisper.cpp C source code
- No changes to `build-xcframework.sh`
- No UI components or SwiftUI views
- No replication of WhisperKit's internal protocol decomposition (AudioEncoding, FeatureExtracting, etc.) — whisper.cpp handles the pipeline internally
- No premature abstractions — keep the API surface lean
- No unnecessary dependencies beyond the XCFramework and AVFoundation

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: NO — setting up from scratch
- **Automated tests**: YES (tests-after)
- **Framework**: XCTest (Swift native)

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| Swift Package compilation | Bash | `swift build --target OpenWhisperKit` |
| Test execution | Bash | `swift test --filter OpenWhisperKitTests` |
| API surface correctness | Bash (swift REPL / compilation check) | Compile a test file that exercises the public API |

**Note on XCFramework dependency**: The XCFramework must be pre-built (`./build-xcframework.sh`) before `swift build` will work. Test scenarios should verify compilation succeeds assuming the XCFramework exists at `build-apple/whisper.xcframework`. If it doesn't exist, the build step should run the build script first.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Package.swift + project structure
└── Task 2: Core data types (Models.swift, Configuration.swift, WhisperError.swift, Protocols.swift)

Wave 2 (After Wave 1):
├── Task 3: WhisperContext actor (C API bridge)
├── Task 4: CallbackBridge (C-to-Swift callbacks)
└── Task 5: AudioProcessor (file loading utility)

Wave 3 (After Wave 2):
├── Task 6: VADProcessor (Silero VAD wrapper)
├── Task 7: TranscriptionTask + OpenWhisperKit main class (orchestration)
└── Task 8: XCTest suite

Wave 4 (After Wave 3):
└── Task 9: Build verification + integration smoke test
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5, 6, 7, 8 | 2 |
| 2 | None | 3, 4, 5, 6, 7, 8 | 1 |
| 3 | 1, 2 | 6, 7 | 4, 5 |
| 4 | 1, 2 | 7 | 3, 5 |
| 5 | 1, 2 | 7 | 3, 4 |
| 6 | 1, 2, 3 | 7 | — |
| 7 | 3, 4, 5, 6 | 8 | — |
| 8 | 7 | 9 | — |
| 9 | 8 | None | — |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | quick (structural setup) |
| 2 | 3, 4, 5 | medium (C interop work) |
| 3 | 6, 7, 8 | unspecified-high (orchestration + tests) |
| 4 | 9 | quick (verification) |

---

## TODOs

- [ ] 1. Create Package.swift and project directory structure

  **What to do**:
  - Create `Package.swift` at repo root with:
    - `.binaryTarget(name: "whisper", path: "build-apple/whisper.xcframework")` 
    - `.target(name: "OpenWhisperKit", dependencies: ["whisper"], path: "Sources/OpenWhisperKit")`
    - `.testTarget(name: "OpenWhisperKitTests", dependencies: ["OpenWhisperKit"], path: "Tests/OpenWhisperKitTests")`
    - Platform constraints: `.iOS(.v16)`, `.macOS(.v13)`
    - Swift tools version 5.9
  - Create directory structure:
    - `Sources/OpenWhisperKit/`
    - `Tests/OpenWhisperKitTests/`
  - Create placeholder files so the package resolves

  **Must NOT do**:
  - Do NOT add any external SPM dependencies
  - Do NOT modify `build-xcframework.sh`
  - Do NOT add remote binary target URL — use local path

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Structural setup, small number of files, well-defined content
  - **Skills**: []
    - No special skills needed for file creation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `build-xcframework.sh:1-18` — XCFramework platform versions (iOS 16.4, macOS 13.3) — use these as minimum deployment targets
  - `build-xcframework.sh:127-144` — Module map structure — shows the module is named `whisper` and what headers are exposed
  - `reference/WhisperKit/Package.swift` — WhisperKit's package structure as inspiration for how to organize targets

  **API/Type References**:
  - `build-apple/whisper.xcframework` — The binary XCFramework output path that the binary target should reference

  **External References**:
  - SPM binary target docs: https://developer.apple.com/documentation/xcode/distributing-binary-frameworks-as-swift-packages

  **Acceptance Criteria**:
  - [ ] `Package.swift` exists at repo root and is valid Swift
  - [ ] Directory `Sources/OpenWhisperKit/` exists
  - [ ] Directory `Tests/OpenWhisperKitTests/` exists
  - [ ] Package resolves: `swift package describe` shows OpenWhisperKit target, whisper binary target, and test target

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Package resolves and shows correct targets
    Tool: Bash
    Preconditions: build-apple/whisper.xcframework exists (run ./build-xcframework.sh if not)
    Steps:
      1. swift package describe
      2. Assert output contains "OpenWhisperKit"
      3. Assert output contains "whisper" (binary target)
      4. Assert output contains "OpenWhisperKitTests"
    Expected Result: Package resolves with 3 targets
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(sdk): add Package.swift and OpenWhisperKit project structure`
  - Files: `Package.swift`, `Sources/OpenWhisperKit/`, `Tests/OpenWhisperKitTests/`

---

- [ ] 2. Implement core data types: Models, Configuration, Errors, Protocols

  **What to do**:

  Create 4 files defining the SDK's public type system:

  **`Sources/OpenWhisperKit/Models.swift`** — Result types:
  - `TranscriptionResult`: text, segments, language, timings (class, Sendable)
  - `TranscriptionSegment`: id, seek, start, end, text, tokens, temperature, avgLogprob, compressionRatio, noSpeechProb, words (struct, Codable, Sendable)
  - `WordTiming`: word, tokens, start, end, probability (struct, Codable, Sendable)
  - `TranscriptionTimings`: inputAudioSeconds, modelLoading, encoding, decoding, fullPipeline, tokensPerSecond, realTimeFactor (struct, Sendable)
  - `TranscriptionProgress`: progress (0.0-1.0), text, tokens, windowId (struct, Sendable)
  - `ModelState` enum: unloaded, loading, loaded (Sendable)
  - `ModelVariant` enum: tiny, tinyEn, base, baseEn, small, smallEn, medium, mediumEn, largeV1, largeV2, largeV3, largeV3Turbo (with `isMultilingual` computed property)
  - `SpeechSegment`: startTime, endTime, startSample, endSample (struct, for VAD results)

  **`Sources/OpenWhisperKit/Configuration.swift`** — Configuration types:
  - `OpenWhisperKitConfig`: modelPath, vadModelPath (optional), computeOptions (struct)
  - `ComputeOptions`: useGPU (default true), useCoreML (default true), flashAttention (default true), threadCount (optional — auto-detect) (struct)
  - `DecodingOptions`: mirroring WhisperKit's useful options — task (.transcribe/.translate), language, temperature, temperatureIncrementOnFallback, temperatureFallbackCount, topK, withoutTimestamps, wordTimestamps, maxInitialTimestamp, suppressBlank, compressionRatioThreshold, logProbThreshold, noSpeechThreshold, chunkingStrategy (.none/.vad), prompt (String? — converted to tokens internally) (struct, Sendable)
  - `DecodingTask` enum: transcribe, translate
  - `ChunkingStrategy` enum: none, vad

  **`Sources/OpenWhisperKit/WhisperError.swift`** — Error types:
  - `WhisperError` enum conforming to `LocalizedError`:
    - `.modelNotFound(String)` — model file doesn't exist at path
    - `.modelLoadFailed(String)` — whisper_init returned null
    - `.vadModelLoadFailed(String)` — VAD model load failed
    - `.transcriptionFailed(String)` — whisper_full returned non-zero
    - `.audioLoadFailed(String)` — couldn't load/convert audio file
    - `.invalidAudioFormat(String)` — wrong sample rate or format
    - `.cancelled` — user cancelled via callback
    - Each case should have a `errorDescription` providing a human-readable message

  **`Sources/OpenWhisperKit/Protocols.swift`** — Core protocols:
  - `Transcribing` protocol:
    ```swift
    public protocol Transcribing: Sendable {
        func transcribe(audioPath: String, options: DecodingOptions?) async throws -> TranscriptionResult
        func transcribe(audioSamples: [Float], options: DecodingOptions?) async throws -> TranscriptionResult
        func detectLanguage(audioPath: String) async throws -> (language: String, probabilities: [String: Float])
    }
    ```
  - `TranscriptionCallback` typealias: `((TranscriptionProgress) -> Bool?)?` — return false to stop

  **Must NOT do**:
  - Do NOT add any C interop code in these files — pure Swift types only
  - Do NOT import `whisper` module here — these are standalone types
  - Do NOT replicate WhisperKit's internal protocols (AudioEncoding, FeatureExtracting, etc.)
  - Do NOT make types overly complex — keep lean, add properties later as needed

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Defining data types is straightforward — well-specified structs/enums/protocols
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8
  - **Blocked By**: None (can write the files even before Package.swift — just source files)

  **References**:

  **Pattern References**:
  - `reference/WhisperKit/Sources/WhisperKit/Core/Models.swift` — WhisperKit's result types (TranscriptionResult, TranscriptionSegment, WordTiming) — use as API inspiration but simplify
  - `reference/WhisperKit/Sources/WhisperKit/Core/Configurations.swift` — WhisperKit's DecodingOptions — use as inspiration for what options to expose
  - `reference/WhisperKit/Sources/WhisperKit/Utilities/WhisperError.swift` — WhisperKit's error enum pattern

  **API/Type References**:
  - `include/whisper.h:460-520` — `whisper_full_params` struct — the C params that DecodingOptions must eventually map to
  - `include/whisper.h:380-420` — Segment/token accessor functions — what data is available to populate TranscriptionSegment

  **Acceptance Criteria**:
  - [ ] `Models.swift` compiles with all types public and Sendable
  - [ ] `Configuration.swift` compiles with `DecodingOptions` having sensible defaults
  - [ ] `WhisperError.swift` compiles with all error cases having `errorDescription`
  - [ ] `Protocols.swift` compiles with `Transcribing` protocol defined
  - [ ] No imports of `whisper` module in any of these files (pure Swift)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All type files compile as part of OpenWhisperKit target
    Tool: Bash
    Preconditions: Package.swift and XCFramework exist
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert exit code 0 or only warnings (no errors)
    Expected Result: Compilation succeeds
    Evidence: Terminal output captured
  ```

  **Commit**: YES (groups with Task 1)
  - Message: `feat(sdk): define core types — TranscriptionResult, DecodingOptions, WhisperError, Protocols`
  - Files: `Sources/OpenWhisperKit/Models.swift`, `Sources/OpenWhisperKit/Configuration.swift`, `Sources/OpenWhisperKit/WhisperError.swift`, `Sources/OpenWhisperKit/Protocols.swift`

---

- [ ] 3. Implement WhisperContext actor — C API bridge

  **What to do**:

  Create `Sources/OpenWhisperKit/WhisperContext.swift` — an actor that owns and manages the `whisper_context` C pointer:

  - **Actor design** (thread-safe access to C context):
    ```swift
    actor WhisperContext {
        private let context: OpaquePointer  // whisper_context*
        private let state: OpaquePointer?   // whisper_state* (optional, for parallel decoding)
        
        init(context: OpaquePointer) { ... }
        deinit { whisper_free(context) }
    }
    ```

  - **Static factory** — `createContext(config:)`:
    - Map `ComputeOptions` to `whisper_context_params`:
      - `use_gpu` ← `computeOptions.useGPU`
      - `flash_attn` ← `computeOptions.flashAttention`
      - `use_coreml` ← `computeOptions.useCoreML` (check if this param exists, may be automatic)
    - Call `whisper_init_from_file_with_params(path, params)`
    - Throw `WhisperError.modelLoadFailed` if null returned
    - On simulator, set `use_gpu = false` via `#if targetEnvironment(simulator)`

  - **Core transcription method** — `fullTranscribe(samples:options:callbacks:)`:
    - Map `DecodingOptions` to `whisper_full_params`:
      - `strategy` ← WHISPER_SAMPLING_GREEDY (default) or WHISPER_SAMPLING_BEAM_SEARCH (if topK > 1 could indicate beam)
      - `language` ← options.language (withCString)
      - `translate` ← (options.task == .translate)
      - `n_threads` ← options threadCount or auto-detect
      - `temperature` ← options.temperature
      - `temperature_inc` ← options.temperatureIncrementOnFallback
      - `no_timestamps` ← options.withoutTimestamps
      - `token_timestamps` ← options.wordTimestamps
      - `max_initial_ts` ← options.maxInitialTimestamp
      - `suppress_blank` ← options.suppressBlank
      - `thold_pt` ← options.logProbThreshold
      - `thold_ptsum` ← (not in DecodingOptions — use default)
      - `entropy_thold` ← options.compressionRatioThreshold
      - `no_speech_thold` ← options.noSpeechThreshold
      - `prompt_tokens` / `prompt_n_tokens` ← from options.prompt (tokenize internally)
    - Set up callbacks (progress, new_segment) via `CallbackBridge`
    - Call `whisper_full(context, params, samples, n_samples)`
    - Throw `WhisperError.transcriptionFailed` if return value != 0
    - Handle cancellation: if the progress callback signals stop, throw `WhisperError.cancelled`

  - **Result extraction methods**:
    - `getSegments() -> [TranscriptionSegment]` — iterate `whisper_full_n_segments()`, extract text, timestamps, token info
    - `getWordTimings(segmentIndex:) -> [WordTiming]` — if token timestamps enabled, extract per-token timing and group into words
    - `getTimings() -> TranscriptionTimings` — extract from `whisper_get_timings()` 
    - `getLanguage() -> String` — from `whisper_full_lang_id()` or auto-detect

  - **Language detection**:
    - `detectLanguage(samples:) -> (String, [String: Float])` — use `whisper_pcm_to_mel()` + `whisper_lang_auto_detect()` to get language probabilities

  - **Model info accessors**:
    - `isMultilingual: Bool` — `whisper_is_multilingual(context)`
    - `modelType: String` — `whisper_model_type_readable(context)` if available

  - **Memory safety rules** (CRITICAL):
    - All `[Float]` audio data must be passed via `withUnsafeBufferPointer` within the actor method — never capture buffer pointers across `await`
    - The actor ensures single-threaded access to the C context
    - `deinit` calls `whisper_free(context)` — never double-free

  **Must NOT do**:
  - Do NOT expose `OpaquePointer` publicly
  - Do NOT allow direct access to `whisper_full_params` from outside the actor
  - Do NOT implement parallel decoding via `whisper_full_parallel()` in initial version
  - Do NOT hold references to C strings beyond their `withCString` scope

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: C interop with memory safety is delicate work requiring careful attention
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `examples/whisper.swiftui/whisper.cpp.swift/LibWhisper.swift:10-153` — Existing actor pattern for WhisperContext — use as foundation but significantly expand
  - `examples/whisper.swiftui/whisper.cpp.swift/LibWhisper.swift:21-49` — `fullTranscribe()` method — shows how to map params and call `whisper_full()`
  - `examples/whisper.swiftui/whisper.cpp.swift/LibWhisper.swift:137-152` — `createContext()` factory — shows how to init with params

  **API/Type References**:
  - `include/whisper.h:440-570` — `whisper_full_params` struct — every field that can be mapped from DecodingOptions
  - `include/whisper.h:370-430` — Segment and token accessors: `whisper_full_n_segments()`, `whisper_full_get_segment_text()`, `whisper_full_get_segment_t0()`, `whisper_full_get_segment_t1()`, `whisper_full_n_tokens()`, `whisper_full_get_token_text()`, `whisper_full_get_token_p()`
  - `include/whisper.h:240-260` — Language functions: `whisper_lang_auto_detect()`, `whisper_lang_str()`
  - `include/whisper.h:280-300` — Timing functions: `whisper_get_timings()`, `whisper_timings` struct
  - `include/whisper.h:150-170` — Init functions: `whisper_init_from_file_with_params()`, `whisper_context_default_params()`, `whisper_context_params`
  - `include/whisper.h:190-200` — Free/state functions: `whisper_free()`, `whisper_free_state()`

  **Acceptance Criteria**:
  - [ ] `WhisperContext.swift` compiles without errors
  - [ ] Actor has `createContext(config:)` factory that takes `OpenWhisperKitConfig`
  - [ ] Actor has `fullTranscribe(samples:options:callbacks:)` method
  - [ ] Actor has `getSegments()` returning `[TranscriptionSegment]`
  - [ ] Actor has `detectLanguage(samples:)` returning language tuple
  - [ ] `deinit` calls `whisper_free`
  - [ ] No public exposure of `OpaquePointer`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: WhisperContext compiles as part of target
    Tool: Bash
    Preconditions: Tasks 1 and 2 complete, XCFramework exists
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert exit code 0
    Expected Result: Compilation succeeds
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(sdk): implement WhisperContext actor wrapping C API`
  - Files: `Sources/OpenWhisperKit/WhisperContext.swift`

---

- [ ] 4. Implement CallbackBridge — C-to-Swift callback bridging

  **What to do**:

  Create `Sources/OpenWhisperKit/CallbackBridge.swift` — handles bridging between whisper.cpp's C function pointer callbacks and Swift closures:

  - **Challenge**: C callbacks are `@convention(c)` function pointers that can't capture Swift closures. We must use the `user_data` void pointer to bridge.

  - **Design**:
    ```swift
    final class CallbackBridge: @unchecked Sendable {
        var progressCallback: ((TranscriptionProgress) -> Bool?)?
        var segmentCallback: (([TranscriptionSegment]) -> Void)?
        var shouldCancel: Bool = false
        
        // C-compatible function pointers (static)
        static let cProgressCallback: @convention(c) (OpaquePointer?, OpaquePointer?, Int32, UnsafeMutableRawPointer?) -> Void = { ... }
        static let cNewSegmentCallback: @convention(c) (OpaquePointer?, OpaquePointer?, Int32, UnsafeMutableRawPointer?) -> Void = { ... }
        static let cAbortCallback: @convention(c) (UnsafeMutableRawPointer?) -> Bool = { ... }
    }
    ```

  - **Progress callback**: 
    - Receives progress percentage (0-100) from whisper.cpp
    - Constructs a `TranscriptionProgress` (normalized to 0.0-1.0)
    - Calls Swift closure
    - If closure returns `false`, sets `shouldCancel = true`
  
  - **Abort callback**:
    - Checks `shouldCancel` flag
    - Returns `true` to abort if flag is set
    - This is how early stopping works — progress callback sets the flag, abort callback reads it

  - **New segment callback**:
    - Fires when whisper.cpp discovers a new segment
    - Extracts segment data from context
    - Calls Swift segment discovery closure

  - **Usage pattern** (how WhisperContext uses it):
    ```swift
    let bridge = CallbackBridge()
    bridge.progressCallback = userCallback
    
    // Pass bridge as user_data, set function pointers
    withExtendedLifetime(bridge) {
        let bridgePtr = Unmanaged.passUnretained(bridge).toOpaque()
        params.progress_callback = CallbackBridge.cProgressCallback
        params.progress_callback_user_data = UnsafeMutableRawPointer(bridgePtr)
        params.abort_callback = CallbackBridge.cAbortCallback
        params.abort_callback_user_data = UnsafeMutableRawPointer(bridgePtr)
        // ... call whisper_full
    }
    ```

  **Must NOT do**:
  - Do NOT use global mutable state for callback storage
  - Do NOT retain C pointers beyond the transcription call
  - Do NOT make CallbackBridge a singleton — one per transcription call
  - Do NOT use `Unmanaged.passRetained` (would leak if whisper_full throws)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: C callback bridging is tricky — memory safety, @convention(c), void pointer casting
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `examples/whisper.swiftui/whisper.cpp.swift/LibWhisper.swift:25-48` — Shows basic params setup (no callbacks used) — extend this with callback wiring

  **API/Type References**:
  - `include/whisper.h:460-475` — `whisper_full_params` callback fields: `new_segment_callback`, `new_segment_callback_user_data`, `progress_callback`, `progress_callback_user_data`, `abort_callback`, `abort_callback_user_data`
  - `include/whisper.h:440-445` — Callback typedefs: `whisper_new_segment_callback`, `whisper_progress_callback`, `ggml_abort_callback`

  **Acceptance Criteria**:
  - [ ] `CallbackBridge.swift` compiles
  - [ ] Has `@convention(c)` static function pointers for progress, new_segment, abort
  - [ ] Progress callback normalizes int percentage to Float 0.0-1.0
  - [ ] Abort callback reads cancellation flag
  - [ ] Uses `Unmanaged.passUnretained` + `withExtendedLifetime` for safe bridging

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: CallbackBridge compiles with correct function signatures
    Tool: Bash
    Preconditions: Tasks 1 and 2 complete
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert no errors related to CallbackBridge or @convention(c)
    Expected Result: Compilation succeeds
    Evidence: Terminal output captured
  ```

  **Commit**: YES (groups with Task 3)
  - Message: `feat(sdk): implement C-to-Swift callback bridge for progress and segments`
  - Files: `Sources/OpenWhisperKit/CallbackBridge.swift`

---

- [ ] 5. Implement AudioProcessor — file loading utility

  **What to do**:

  Create `Sources/OpenWhisperKit/AudioProcessor.swift` — utility for loading audio files and converting to the Float32 PCM @ 16kHz format that whisper.cpp requires:

  - **Static utility class** (no state needed):
    ```swift
    public enum AudioProcessor {
        public static func loadAudio(fromPath path: String) throws -> [Float]
        public static func loadAudio(fromURL url: URL) throws -> [Float]
        public static func convertToMono16kHz(_ buffer: AVAudioPCMBuffer) throws -> [Float]
    }
    ```

  - **`loadAudio(fromPath:)`**:
    - Use `AVAudioFile` to open the audio file
    - Supports wav, mp3, m4a, flac, caf (anything AVFoundation supports)
    - Read into `AVAudioPCMBuffer`
    - Convert to mono 16kHz Float32 via `AVAudioConverter`
    - Return `[Float]` array
    - Throw `WhisperError.audioLoadFailed` if file can't be opened
    - Throw `WhisperError.invalidAudioFormat` if conversion fails

  - **`convertToMono16kHz(_:)`**:
    - Takes an `AVAudioPCMBuffer` (any format)
    - Creates output format: mono, 16000 Hz, Float32
    - Uses `AVAudioConverter` to resample and downmix
    - Returns `[Float]` array

  - **Import requirements**: `import AVFoundation`

  **Must NOT do**:
  - Do NOT handle microphone input or recording
  - Do NOT implement streaming audio loading
  - Do NOT cache loaded audio
  - Do NOT add any external dependencies — AVFoundation only

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Standard AVFoundation audio loading — well-documented pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `reference/WhisperKit/Sources/WhisperKit/Core/AudioProcessor.swift` — WhisperKit's audio loading implementation — use as reference for AVAudioFile/AVAudioConverter patterns
  - `examples/whisper.swiftui/whisper.swiftui.demo/Models/WhisperState.swift` — Shows basic audio file reading pattern in the existing example

  **External References**:
  - AVAudioFile: https://developer.apple.com/documentation/avfaudio/avaudiofile
  - AVAudioConverter: https://developer.apple.com/documentation/avfaudio/avaudioconverter

  **Acceptance Criteria**:
  - [ ] `AudioProcessor.swift` compiles
  - [ ] `loadAudio(fromPath:)` is public and returns `[Float]`
  - [ ] Uses AVFoundation for format conversion (mono 16kHz Float32)
  - [ ] Throws `WhisperError.audioLoadFailed` for invalid paths
  - [ ] No external dependencies beyond AVFoundation

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: AudioProcessor compiles as part of target
    Tool: Bash
    Preconditions: Tasks 1 and 2 complete
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert no errors related to AudioProcessor
    Expected Result: Compilation succeeds
    Evidence: Terminal output captured
  ```

  **Commit**: YES (groups with Tasks 3, 4)
  - Message: `feat(sdk): add AudioProcessor for audio file loading and format conversion`
  - Files: `Sources/OpenWhisperKit/AudioProcessor.swift`

---

- [ ] 6. Implement VADProcessor — Silero VAD wrapper

  **What to do**:

  Create `Sources/OpenWhisperKit/VADProcessor.swift` — wraps whisper.cpp's built-in Silero VAD to detect speech segments and filter silence:

  - **Design**:
    ```swift
    public final class VADProcessor: Sendable {
        private let vadContext: OpaquePointer  // whisper_vad_context*
        
        public init(modelPath: String) throws
        deinit  // whisper_vad_free(vadContext)
        
        public func detectSpeech(in samples: [Float], sampleRate: Int = 16000) throws -> [SpeechSegment]
        public func filterSilence(from samples: [Float], sampleRate: Int = 16000) throws -> [Float]
    }
    ```

  - **Initialization**:
    - Load VAD model via `whisper_vad_init_from_file_with_params()` (check exact C API name — look in whisper.h for `whisper_vad_*` functions)
    - Throw `WhisperError.vadModelLoadFailed` if null returned

  - **`detectSpeech(in:)`**:
    - Call whisper_vad C API to get speech/silence segments
    - Map C results to `[SpeechSegment]` (startTime, endTime, startSample, endSample)
    - Return array of speech segments

  - **`filterSilence(from:)`**:
    - Call `detectSpeech()` to get segments
    - Extract and concatenate only the speech samples
    - Return filtered `[Float]` array with silence removed
    - Useful for long audio files where VAD significantly reduces processing time

  - **IMPORTANT**: Check `include/whisper.h` for the exact VAD API functions. The C API may use `whisper_vad_*` prefix. If the VAD API is not exposed in the public header, we may need to check if it's accessible through the XCFramework's module map.

  **Must NOT do**:
  - Do NOT implement custom VAD logic — use whisper.cpp's built-in Silero VAD
  - Do NOT make VAD mandatory — it's optional (enabled via `ChunkingStrategy.vad`)
  - Do NOT cache VAD results across calls

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Need to discover exact VAD C API surface and bridge it correctly
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential — depends on WhisperContext pattern from Task 3)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 3 (uses same C interop patterns)

  **References**:

  **API/Type References**:
  - `include/whisper.h` — Search for `whisper_vad` functions — these are the C API functions to wrap
  - `src/whisper.cpp` — If VAD functions aren't in the header, check the implementation for the internal API

  **Documentation References**:
  - Root `README.md` "Voice Activity Detection (VAD)" section — documents VAD usage with Silero model, `--vad` flag, and VAD options (threshold, min speech duration, min silence duration, etc.)

  **External References**:
  - Silero VAD model download: `./models/download-vad-model.sh silero-v6.2.0`

  **Acceptance Criteria**:
  - [ ] `VADProcessor.swift` compiles
  - [ ] Can initialize with a Silero VAD model path
  - [ ] `detectSpeech(in:)` returns `[SpeechSegment]` with start/end times
  - [ ] `filterSilence(from:)` returns filtered `[Float]` with silence removed
  - [ ] Properly frees VAD context in `deinit`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: VADProcessor compiles as part of target
    Tool: Bash
    Preconditions: Tasks 1-3 complete
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert no errors related to VADProcessor
    Expected Result: Compilation succeeds
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(sdk): add VADProcessor wrapping whisper.cpp's Silero VAD`
  - Files: `Sources/OpenWhisperKit/VADProcessor.swift`

---

- [ ] 7. Implement OpenWhisperKit main class + TranscriptionTask orchestration

  **What to do**:

  This is the centerpiece — the main public class that users interact with. Create two files:

  **`Sources/OpenWhisperKit/TranscriptionTask.swift`** — Internal orchestration:
  - Coordinates the transcription pipeline: audio loading → (optional VAD) → transcription → result extraction
  - Internal, not public

  **`Sources/OpenWhisperKit/OpenWhisperKit.swift`** — Main public class conforming to `Transcribing`:
  ```swift
  public final class OpenWhisperKit: Transcribing, Sendable {
      // Public read-only state
      public private(set) var modelState: ModelState
      public private(set) var currentTimings: TranscriptionTimings?
      
      // Configuration
      public let config: OpenWhisperKitConfig
      
      // Internal components
      private let whisperContext: WhisperContext
      private let vadProcessor: VADProcessor?
      
      // MARK: - Initialization
      
      /// Initialize with config. Loads model immediately.
      public init(_ config: OpenWhisperKitConfig) async throws
      
      /// Convenience: init with just a model path
      public init(modelPath: String, vadModelPath: String? = nil) async throws
      
      // MARK: - Transcription (Transcribing protocol)
      
      /// Transcribe from audio file path
      public func transcribe(
          audioPath: String,
          options: DecodingOptions? = nil,
          callback: TranscriptionCallback = nil
      ) async throws -> TranscriptionResult
      
      /// Transcribe from Float32 PCM samples (16kHz mono)
      public func transcribe(
          audioSamples: [Float],
          options: DecodingOptions? = nil,
          callback: TranscriptionCallback = nil
      ) async throws -> TranscriptionResult
      
      /// Batch transcribe multiple audio files
      public func transcribe(
          audioPaths: [String],
          options: DecodingOptions? = nil,
          callback: TranscriptionCallback = nil
      ) async throws -> [Result<TranscriptionResult, Error>]
      
      // MARK: - Language Detection
      
      /// Detect language from audio file
      public func detectLanguage(
          audioPath: String
      ) async throws -> (language: String, probabilities: [String: Float])
      
      /// Detect language from samples
      public func detectLanguage(
          audioSamples: [Float]
      ) async throws -> (language: String, probabilities: [String: Float])
  }
  ```

  **Transcription flow** (`transcribe(audioPath:)`):
  1. Load audio via `AudioProcessor.loadAudio(fromPath:)` → `[Float]`
  2. If `options.chunkingStrategy == .vad` and `vadProcessor != nil`:
     - Run VAD to detect speech segments
     - Process each speech segment separately
     - Merge results with correct timestamp offsets
  3. Else: pass full audio to `whisperContext.fullTranscribe()`
  4. Extract results: `whisperContext.getSegments()`, timings, language
  5. Build and return `TranscriptionResult`

  **Transcription flow** (`transcribe(audioSamples:)`):
  - Same as above but skip step 1 (samples already provided)
  - Validate: if samples is empty, return empty result

  **Batch transcription** (`transcribe(audioPaths:)`):
  - Iterate paths sequentially (whisper.cpp context can't run parallel transcriptions)
  - Return `[Result]` to handle per-file errors without failing the batch

  **Key design decisions**:
  - The class is `Sendable` because it delegates all mutable state to the actor
  - Default `DecodingOptions` uses greedy sampling, English, temperature 0.0
  - The `callback` fires during transcription for progress (delegated to CallbackBridge)

  **Must NOT do**:
  - Do NOT implement parallel transcription via `whisper_full_parallel()` — keep simple
  - Do NOT implement streaming
  - Do NOT add model downloading
  - Do NOT make the init failable (use throwing init instead)
  - Do NOT add any UI-related code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: This is the orchestration layer tying everything together — most complex task
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential — needs all previous tasks)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 3, 4, 5, 6

  **References**:

  **Pattern References**:
  - `reference/WhisperKit/Sources/WhisperKit/Core/WhisperKit.swift` — WhisperKit's main class — use as API inspiration (not implementation). Specifically look at:
    - Lines defining `transcribe(audioPath:)` and `transcribe(audioArray:)` signatures
    - How it delegates to internal components
    - How batch transcription returns `[Result]`
  - `reference/WhisperKit/Sources/WhisperKit/Core/TranscribeTask.swift` — WhisperKit's internal transcription orchestration — understand the flow but our version is simpler (whisper.cpp handles the decode loop)
  - `examples/whisper.swiftui/whisper.swiftui.demo/Models/WhisperState.swift` — Shows how the existing example orchestrates model loading → transcription → result display

  **API/Type References**:
  - All types from Task 2: `TranscriptionResult`, `TranscriptionSegment`, `DecodingOptions`, `WhisperError`
  - `WhisperContext` from Task 3: `createContext()`, `fullTranscribe()`, `getSegments()`
  - `CallbackBridge` from Task 4
  - `AudioProcessor` from Task 5
  - `VADProcessor` from Task 6

  **Acceptance Criteria**:
  - [ ] `OpenWhisperKit` class compiles and conforms to `Transcribing`
  - [ ] Has `init(_ config:)` and convenience `init(modelPath:)`
  - [ ] `transcribe(audioPath:)` loads audio and returns `TranscriptionResult`
  - [ ] `transcribe(audioSamples:)` accepts float array
  - [ ] `transcribe(audioPaths:)` handles batch with per-file error handling
  - [ ] `detectLanguage()` returns language string and probability dictionary
  - [ ] VAD integration works when `chunkingStrategy == .vad`
  - [ ] Progress callback fires during transcription
  - [ ] Early stopping works (callback returns false → transcription stops)
  - [ ] `swift build --target OpenWhisperKit` succeeds with zero errors

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Full target compiles
    Tool: Bash
    Preconditions: All previous tasks complete, XCFramework exists
    Steps:
      1. swift build --target OpenWhisperKit 2>&1
      2. Assert exit code 0
    Expected Result: Full SDK compiles
    Evidence: Terminal output captured

  Scenario: Public API surface is correct
    Tool: Bash
    Preconditions: Target compiles
    Steps:
      1. Create a temporary Swift file that imports OpenWhisperKit and exercises the public API:
         - let config = OpenWhisperKitConfig(modelPath: "/tmp/test.bin")
         - let kit = try await OpenWhisperKit(config)
         - let result = try await kit.transcribe(audioPath: "/tmp/test.wav")
         - let _ = result.text
         - let _ = result.segments
      2. swift build this file against OpenWhisperKit
      3. Assert compilation succeeds (we don't need it to run — just compile)
    Expected Result: Public API compiles correctly
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `feat(sdk): implement OpenWhisperKit main class and transcription orchestration`
  - Files: `Sources/OpenWhisperKit/OpenWhisperKit.swift`, `Sources/OpenWhisperKit/TranscriptionTask.swift`

---

- [ ] 8. Implement XCTest suite

  **What to do**:

  Create `Tests/OpenWhisperKitTests/` with test files covering the SDK's public API:

  **`Tests/OpenWhisperKitTests/ModelsTests.swift`** — Unit tests for data types:
  - Test `TranscriptionSegment` initialization and computed properties (duration)
  - Test `WordTiming` initialization and computed properties
  - Test `DecodingOptions` default values match expected defaults
  - Test `ModelVariant.isMultilingual` returns correct values
  - Test `WhisperError` error descriptions are non-empty
  - Test `TranscriptionResult` can be created and accessed
  - These tests require NO model files — pure Swift type tests

  **`Tests/OpenWhisperKitTests/ConfigurationTests.swift`** — Config tests:
  - Test `OpenWhisperKitConfig` initialization with defaults
  - Test `ComputeOptions` defaults (useGPU=true, useCoreML=true, flashAttention=true)
  - Test `DecodingOptions` custom initialization overrides defaults

  **`Tests/OpenWhisperKitTests/AudioProcessorTests.swift`** — Audio loading tests:
  - Test `loadAudio(fromPath:)` throws for nonexistent file
  - Test `loadAudio(fromPath:)` with a bundled test wav file (if available)
  - Test the thrown error is `WhisperError.audioLoadFailed`

  **`Tests/OpenWhisperKitTests/IntegrationTests.swift`** — Integration tests (require model):
  - These tests are conditionally skipped if no model is present at a known path
  - Test: Initialize `OpenWhisperKit` with model path → modelState == .loaded
  - Test: Transcribe a short wav file → result.text is non-empty
  - Test: Transcribe with word timestamps → segments have words
  - Test: Detect language on English audio → returns "en"
  - Test: Progress callback fires at least once during transcription
  - Test: Early stopping via callback → transcription stops (result may be partial)
  - Guard: `try XCTSkipUnless(FileManager.default.fileExists(atPath: testModelPath))`

  **Test resources**:
  - Bundle `samples/jfk.wav` (already in repo) as a test resource if feasible
  - Or reference it by path: `../../samples/jfk.wav`

  **Must NOT do**:
  - Do NOT make tests depend on model download — skip if model not present
  - Do NOT test whisper.cpp internals — only test the Swift SDK public API
  - Do NOT create mock/fake implementations of protocols for V1

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Tests need to cover both unit and integration scenarios, handle conditional skipping
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `reference/WhisperKit/Tests/` — WhisperKit's test suite structure (if it exists in the reference directory)
  - `samples/jfk.wav` — Existing test audio file in the repo (11 seconds, English, JFK speech)

  **API/Type References**:
  - All public types from Tasks 2, 5, 7 — `TranscriptionResult`, `DecodingOptions`, `OpenWhisperKit`, `AudioProcessor`, etc.

  **Acceptance Criteria**:
  - [ ] Test target compiles: `swift build --build-tests`
  - [ ] Unit tests (ModelsTests, ConfigurationTests) pass without any model file
  - [ ] AudioProcessor tests pass for error cases
  - [ ] Integration tests are properly guarded with `XCTSkipUnless`
  - [ ] `swift test --filter OpenWhisperKitTests` runs and reports results

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Unit tests pass without model
    Tool: Bash
    Preconditions: All SDK source files written, XCFramework exists
    Steps:
      1. swift test --filter ModelsTests 2>&1
      2. Assert "Test Suite 'ModelsTests' passed"
      3. swift test --filter ConfigurationTests 2>&1
      4. Assert "Test Suite 'ConfigurationTests' passed"
    Expected Result: Pure Swift type tests pass
    Evidence: Terminal output captured

  Scenario: Test target compiles
    Tool: Bash
    Preconditions: All SDK source files written
    Steps:
      1. swift build --build-tests 2>&1
      2. Assert exit code 0
    Expected Result: Test target compiles
    Evidence: Terminal output captured
  ```

  **Commit**: YES
  - Message: `test(sdk): add XCTest suite for OpenWhisperKit — unit and integration tests`
  - Files: `Tests/OpenWhisperKitTests/ModelsTests.swift`, `Tests/OpenWhisperKitTests/ConfigurationTests.swift`, `Tests/OpenWhisperKitTests/AudioProcessorTests.swift`, `Tests/OpenWhisperKitTests/IntegrationTests.swift`

---

- [ ] 9. Build verification and integration smoke test

  **What to do**:

  Final verification that everything works end-to-end:

  1. **Clean build**: `swift package clean && swift build --target OpenWhisperKit`
  2. **Run all tests**: `swift test --filter OpenWhisperKitTests`
  3. **Verify public API**: Create a small Swift script that exercises the full public API (imports, init, transcribe, results) and verify it compiles
  4. **Verify no warnings**: Check build output for warnings and fix any that appear
  5. **Verify XCFramework integration**: Ensure the binary target resolves correctly

  **Must NOT do**:
  - Do NOT skip this step
  - Do NOT ignore build warnings

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification step, running existing build/test commands
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Task 8

  **References**:
  - All files from Tasks 1-8

  **Acceptance Criteria**:
  - [ ] `swift build --target OpenWhisperKit` succeeds with zero errors
  - [ ] `swift test --filter OpenWhisperKitTests` reports all tests passing (integration tests may skip)
  - [ ] No compiler warnings in the OpenWhisperKit target
  - [ ] Public API surface matches the design (transcribe from path, from array, batch, detect language, VAD)

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Clean build succeeds
    Tool: Bash
    Steps:
      1. swift package clean
      2. swift build --target OpenWhisperKit 2>&1
      3. Assert exit code 0
      4. Assert no "error:" in output
    Expected Result: Clean build passes
    Evidence: Terminal output captured

  Scenario: All tests run
    Tool: Bash
    Steps:
      1. swift test --filter OpenWhisperKitTests 2>&1
      2. Assert "Test Suite 'OpenWhisperKitTests' passed" or individual suites passed
      3. Count passed/skipped/failed
    Expected Result: All pass or skip (no failures)
    Evidence: Terminal output captured
  ```

  **Commit**: NO (verification only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1+2 | `feat(sdk): add Package.swift, project structure, and core types` | Package.swift, Sources/OpenWhisperKit/*.swift | `swift package describe` |
| 3+4+5 | `feat(sdk): implement WhisperContext, CallbackBridge, and AudioProcessor` | Sources/OpenWhisperKit/{WhisperContext,CallbackBridge,AudioProcessor}.swift | `swift build --target OpenWhisperKit` |
| 6 | `feat(sdk): add VADProcessor wrapping Silero VAD` | Sources/OpenWhisperKit/VADProcessor.swift | `swift build` |
| 7 | `feat(sdk): implement OpenWhisperKit main class and transcription orchestration` | Sources/OpenWhisperKit/{OpenWhisperKit,TranscriptionTask}.swift | `swift build` |
| 8 | `test(sdk): add XCTest suite for OpenWhisperKit` | Tests/OpenWhisperKitTests/*.swift | `swift test` |

---

## Success Criteria

### Verification Commands
```bash
swift package describe          # Expected: Shows OpenWhisperKit, whisper, OpenWhisperKitTests targets
swift build --target OpenWhisperKit  # Expected: Build Succeeded
swift test --filter OpenWhisperKitTests  # Expected: All passed (integration may skip)
```

### Final Checklist
- [ ] All "Must Have" features present in public API
- [ ] All "Must NOT Have" guardrails respected (no streaming, no downloads, no C code changes)
- [ ] All tests pass or appropriately skip
- [ ] Clean build with zero errors and minimal warnings
- [ ] SDK can transcribe audio file → TranscriptionResult with segments, words, timings
- [ ] VAD works when enabled
- [ ] Progress callbacks fire
- [ ] Early stopping works
- [ ] CoreML + Metal acceleration enabled (passthrough from XCFramework)
