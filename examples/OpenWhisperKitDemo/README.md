# OpenWhisperKit Demo

A complete SwiftUI macOS application demonstrating the OpenWhisperKit SDK with model downloads, audio recording, and transcription.

## Features

- **Model Download**: Downloads Whisper large-v3-turbo model files from HuggingFace with progress tracking
- **CoreML Compilation**: Handles first-time CoreML model compilation with user feedback
- **Audio Recording**: Records audio from the microphone with duration display
- **Transcription**: Transcribes recorded audio with progress tracking and timing information
- **Modern UI**: Clean, gradient-based interface with state-specific views

## Requirements

- macOS 13.0 or later
- Xcode 15.0 or later
- Microphone access permission

## Building

From the `examples/OpenWhisperKitDemo` directory:

```bash
swift build
```

Or build in release mode:

```bash
swift build -c release
```

## Running

```bash
swift run
```

Or run the built executable:

```bash
.build/debug/OpenWhisperKitDemo
```

## Usage

1. **First Launch**: Click "Download Models" to download the required model files (~1.5 GB total)
2. **Model Loading**: Wait for the CoreML model to compile (2-5 minutes on first launch)
3. **Recording**: Click "Start Recording" and speak into your microphone
4. **Transcription**: Click "Stop Recording" to transcribe the audio
5. **Results**: View the transcription text and processing time

## Architecture

The app is structured with clean separation of concerns:

- **OpenWhisperKitDemoApp.swift**: App entry point
- **ContentView.swift**: SwiftUI UI with state-specific views
- **ViewModel.swift**: ObservableObject managing app state and orchestration
- **DownloadManager.swift**: URLSession-based downloads with progress tracking
- **AudioRecorder.swift**: AVAudioRecorder wrapper for microphone recording

## Model Files

The app downloads two files:

1. **CoreML Encoder** (~850 KB): `ggml-large-v3-turbo-encoder.mlmodelc.zip`
2. **GGML Model** (~1.5 GB): `ggml-large-v3-turbo-q5_0.bin`

Files are stored in the app's Documents directory and persist between launches.

## Notes

- The first model load triggers CoreML compilation which takes several minutes
- Subsequent launches are much faster as the compiled model is cached
- Microphone permission is requested automatically on first recording
- The app uses 16kHz mono PCM audio format for optimal compatibility
