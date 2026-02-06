import Foundation
import SwiftUI
import OpenWhisperKit

enum AppState: Equatable {
    case needsDownload
    case downloading
    case loadingModel
    case ready
    case recording
    case transcribing
    case result
    case error(String)
}

class ViewModel: ObservableObject {
    @Published var state: AppState = .needsDownload
    @Published var encoderProgress: Double = 0.0
    @Published var modelProgress: Double = 0.0
    @Published var encoderBytes: String = ""
    @Published var modelBytes: String = ""
    @Published var sortformerGGUFProgress: Double = 0.0
    @Published var sortformerCoreMLProgress: Double = 0.0
    @Published var sortformerGGUFBytes: String = ""
    @Published var sortformerCoreMLBytes: String = ""
    @Published var transcriptionText: String = ""
    @Published var recordingDuration: TimeInterval = 0
    @Published var transcriptionProgress: Double = 0.0
    @Published var transcriptionTime: String = ""
    @Published var diarizationText: String = ""
    @Published var diarizedUtterances: [DiarizedUtterance] = []
    @Published var isDiarizing: Bool = false
    
    private let downloadManager = DownloadManager()
    private let audioRecorder = AudioRecorder()
    private var whisperKit: OpenWhisperKit?
    private var sortformerContext: SortFormerContext?
    private var shouldStopTranscription = false
    
    private var recordingTimer: Timer?
    private var recordingStartTime: Date?
    
    private let encoderURL = URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-encoder.mlmodelc.zip")!
    private let modelURL = URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin")!
    private let sortformerGGUFURL = URL(string: "https://huggingface.co/andyye/streaming-sortformer/resolve/main/model.gguf")!
    private let sortformerCoreMLURL = URL(string: "https://huggingface.co/andyye/streaming-sortformer/resolve/main/model-coreml-head.mlmodelc.zip")!
    
    private var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    }
    
    private var modelPath: URL {
        documentsDirectory.appendingPathComponent("ggml-large-v3-turbo-q5_0.bin")
    }
    
    private var encoderPath: URL {
        documentsDirectory.appendingPathComponent("ggml-large-v3-turbo-encoder.mlmodelc")
    }
    
    private var sortformerGGUFPath: URL {
        documentsDirectory.appendingPathComponent("model.gguf")
    }
    
    private var sortformerCoreMLPath: URL {
        documentsDirectory.appendingPathComponent("model-coreml-head.mlmodelc")
    }
    
    init() {
        checkExistingModels()
    }
    
    func checkExistingModels() {
        let modelExists = FileManager.default.fileExists(atPath: modelPath.path)
        let encoderExists = FileManager.default.fileExists(atPath: encoderPath.path)
        let sfGGUFExists = FileManager.default.fileExists(atPath: sortformerGGUFPath.path)
        let sfCoreMLExists = FileManager.default.fileExists(atPath: sortformerCoreMLPath.path)
        
        if modelExists && encoderExists && sfGGUFExists && sfCoreMLExists {
            state = .ready
            Task {
                await loadModel()
            }
        } else {
            state = .needsDownload
        }
    }
    
    func startDownload() {
        state = .downloading
        encoderProgress = 0.0
        modelProgress = 0.0
        sortformerGGUFProgress = 0.0
        sortformerCoreMLProgress = 0.0
        
        Task {
            await withTaskGroup(of: Void.self) { group in
                group.addTask {
                    await self.downloadEncoder()
                }
                
                group.addTask {
                    await self.downloadModel()
                }
                
                group.addTask {
                    await self.downloadSortformerGGUF()
                }
                
                group.addTask {
                    await self.downloadSortformerCoreML()
                }
                
                await group.waitForAll()
                
                await MainActor.run {
                    if self.encoderProgress >= 1.0 && self.modelProgress >= 1.0
                        && self.sortformerGGUFProgress >= 1.0 && self.sortformerCoreMLProgress >= 1.0 {
                        self.state = .loadingModel
                        Task {
                            await self.loadModel()
                        }
                    }
                }
            }
        }
    }
    
    private func downloadEncoder() async {
        do {
            let stream = try await downloadManager.download(from: encoderURL)
            
            for await progress in stream {
                await MainActor.run {
                    self.encoderProgress = progress.progress
                    self.encoderBytes = self.formatBytes(progress.bytesDownloaded, total: progress.totalBytes)
                }
            }
        } catch {
            await MainActor.run {
                self.state = .error("Failed to download encoder: \(error.localizedDescription)")
            }
        }
    }
    
    private func downloadModel() async {
        do {
            let stream = try await downloadManager.download(from: modelURL)
            
            for await progress in stream {
                await MainActor.run {
                    self.modelProgress = progress.progress
                    self.modelBytes = self.formatBytes(progress.bytesDownloaded, total: progress.totalBytes)
                }
            }
        } catch {
            await MainActor.run {
                self.state = .error("Failed to download model: \(error.localizedDescription)")
            }
        }
    }
    
    private func downloadSortformerGGUF() async {
        do {
            let stream = try await downloadManager.download(from: sortformerGGUFURL)
            
            for await progress in stream {
                await MainActor.run {
                    self.sortformerGGUFProgress = progress.progress
                    self.sortformerGGUFBytes = self.formatBytes(progress.bytesDownloaded, total: progress.totalBytes)
                }
            }
        } catch {
            await MainActor.run {
                self.state = .error("Failed to download sortformer GGUF: \(error.localizedDescription)")
            }
        }
    }
    
    private func downloadSortformerCoreML() async {
        do {
            let stream = try await downloadManager.download(from: sortformerCoreMLURL)
            
            for await progress in stream {
                await MainActor.run {
                    self.sortformerCoreMLProgress = progress.progress
                    self.sortformerCoreMLBytes = self.formatBytes(progress.bytesDownloaded, total: progress.totalBytes)
                }
            }
        } catch {
            await MainActor.run {
                self.state = .error("Failed to download sortformer CoreML: \(error.localizedDescription)")
            }
        }
    }
    
    private func loadModel() async {
        await MainActor.run {
            self.state = .loadingModel
        }
        
        do {
            let config = OpenWhisperKitConfig(modelPath: modelPath.path)
            let kit = try await OpenWhisperKit(config)
            
            // Load sortformer context (graceful degradation if it fails)
            var sfContext: SortFormerContext?
            if FileManager.default.fileExists(atPath: sortformerGGUFPath.path) {
                do {
                    sfContext = try SortFormerContext.createContext(modelPath: sortformerGGUFPath.path)
                    print("[ViewModel] SortFormer model loaded successfully")
                } catch {
                    print("[ViewModel] WARNING: Failed to load sortformer model: \(error.localizedDescription)")
                }
            }
            
            await MainActor.run {
                self.whisperKit = kit
                self.sortformerContext = sfContext
                self.state = .ready
            }
        } catch {
            await MainActor.run {
                self.state = .error("Failed to load model: \(error.localizedDescription)")
            }
        }
    }
    
    func stopTranscription() {
        shouldStopTranscription = true
    }
    
    func dismissResult() {
        state = .ready
        diarizationText = ""
        diarizedUtterances = []
    }
    
    func transcribeFile(at url: URL) {
        guard state == .ready else { return }
        let accessing = url.startAccessingSecurityScopedResource()
        Task {
            await transcribe(audioURL: url)
            if accessing {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }
    
    func startRecording() {
        guard state == .ready else { return }
        
        let recordingURL = documentsDirectory.appendingPathComponent("recording.wav")
        
        Task {
            do {
                try await audioRecorder.startRecording(toOutputFile: recordingURL)
                
                await MainActor.run {
                    self.state = .recording
                    self.recordingStartTime = Date()
                    self.recordingDuration = 0
                    
                    self.recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                        guard let self = self, let startTime = self.recordingStartTime else { return }
                        self.recordingDuration = Date().timeIntervalSince(startTime)
                    }
                }
            } catch {
                await MainActor.run {
                    self.state = .error("Failed to start recording: \(error.localizedDescription)")
                }
            }
        }
    }
    
    func stopRecording() {
        guard state == .recording else { return }
        
        recordingTimer?.invalidate()
        recordingTimer = nil
        
        Task {
            guard let recordingURL = await audioRecorder.stopRecording() else {
                await MainActor.run {
                    self.state = .error("Failed to get recording URL")
                }
                return
            }
            
            await transcribe(audioURL: recordingURL)
        }
    }
    
    private func transcribe(audioURL: URL) async {
        await MainActor.run {
            self.state = .transcribing
            self.transcriptionProgress = 0.0
            self.transcriptionText = ""
            self.diarizationText = ""
            self.diarizedUtterances = []
            self.isDiarizing = false
            self.shouldStopTranscription = false
        }
        
        guard let whisperKit = whisperKit else {
            await MainActor.run {
                self.state = .error("Model not loaded")
            }
            return
        }
        
        let startTime = Date()
        
        do {
            let callback: TranscriptionCallback = { [weak self] progress in
                guard let self = self else { return false }
                Task { @MainActor in
                    self.transcriptionProgress = Double(progress.progress)
                    if !progress.text.isEmpty {
                        self.transcriptionText = progress.text
                    }
                }
                return self.shouldStopTranscription ? false : true
            }
            
            print("[Transcribe] Starting transcription for: \(audioURL.path)")
            let result = try await whisperKit.transcribe(
                audioPath: audioURL.path,
                options: DecodingOptions(wordTimestamps: true),
                callback: callback
            )
            
            let elapsed = Date().timeIntervalSince(startTime)
            print("[Transcribe] Done in \(String(format: "%.1fs", elapsed)). Segments: \(result.segments.count)")
            
            await MainActor.run {
                if !result.text.isEmpty {
                    self.transcriptionText = result.text
                }
                self.transcriptionTime = String(format: "%.1fs", elapsed)
            }
            
            // Run diarization if sortformer is available
            if let sfContext = sortformerContext {
                await MainActor.run { self.isDiarizing = true }
                
                do {
                    let samples = try AudioProcessor.loadAudio(fromPath: audioURL.path)
                    let diarResult = try await sfContext.diarize(samples: samples)
                    
                    let allWords = result.segments.flatMap { $0.words ?? [] }
                    
                    if !allWords.isEmpty {
                        let aligned = try DiarizationAligner.align(
                            words: allWords,
                            diarizationSegments: diarResult.segments,
                            options: DiarizationAligner.AlignmentOptions(fillNearest: true, sentenceSmoothing: true)
                        )
                        
                        await MainActor.run {
                            self.diarizedUtterances = aligned.segments
                            self.diarizationText = aligned.text
                        }
                        
                        print("[Diarize] Aligned \(allWords.count) words into \(aligned.segments.count) utterances")
                    } else {
                        print("[Diarize] No word timestamps available, skipping diarization")
                    }
                } catch {
                    print("[Diarize] WARNING: Diarization failed: \(error.localizedDescription)")
                }
                
                await MainActor.run { self.isDiarizing = false }
            }
            
            await MainActor.run {
                self.state = .result
            }
        } catch is WhisperError where shouldStopTranscription {
            let elapsed = Date().timeIntervalSince(startTime)
            print("[Transcribe] Stopped by user after \(String(format: "%.1fs", elapsed))")
            await MainActor.run {
                self.transcriptionTime = String(format: "%.1fs (stopped)", elapsed)
                self.state = .result
            }
        } catch {
            print("[Transcribe] ERROR: \(error)")
            await MainActor.run {
                self.state = .error("Transcription failed: \(error.localizedDescription)")
            }
        }
    }
    
    private func formatBytes(_ bytes: Int64, total: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        let downloaded = formatter.string(fromByteCount: bytes)
        let totalStr = formatter.string(fromByteCount: total)
        return "\(downloaded) / \(totalStr)"
    }
}
