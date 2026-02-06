import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var viewModel = ViewModel()
    @State private var isRecordingPulse = false
    @State private var isFilePickerPresented = false
    
    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(red: 0.05, green: 0.05, blue: 0.15), Color(red: 0.15, green: 0.05, blue: 0.25)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 0) {
                headerView
                
                Spacer()
                
                mainContentView
                
                Spacer()
                
                footerView
            }
            .padding(40)
        }
        .frame(minWidth: 600, minHeight: 700)
        .fileImporter(
            isPresented: $isFilePickerPresented,
            allowedContentTypes: [UTType.audio],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    viewModel.transcribeFile(at: url)
                }
            case .failure(let error):
                print("File picker error: \(error.localizedDescription)")
            }
        }
    }
    
    private var headerView: some View {
        VStack(spacing: 8) {
            HStack(spacing: 12) {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 48, weight: .light))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.white, Color(white: 0.7)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                
                Text("OpenWhisperKit")
                    .font(.system(size: 42, weight: .thin, design: .rounded))
                    .foregroundColor(.white)
            }
            
            Text(stateDescription)
                .font(.system(size: 14, weight: .medium, design: .monospaced))
                .foregroundColor(Color(white: 0.6))
                .textCase(.uppercase)
                .tracking(2)
        }
    }
    
    private var mainContentView: some View {
        Group {
            switch viewModel.state {
            case .needsDownload:
                downloadView
            case .downloading:
                downloadingView
            case .loadingModel:
                loadingView
            case .ready:
                readyView
            case .recording:
                recordingView
            case .transcribing:
                transcribingView
            case .result:
                resultView
            case .error(let message):
                errorView(message: message)
            }
        }
    }
    
    private var downloadView: some View {
        VStack(spacing: 32) {
            VStack(spacing: 16) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 72, weight: .ultraLight))
                    .foregroundColor(.white.opacity(0.9))
                
                Text("Models Required")
                    .font(.system(size: 28, weight: .light))
                    .foregroundColor(.white)
                
                Text("Download Whisper large-v3-turbo model\n~1.5 GB total")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.6))
                    .multilineTextAlignment(.center)
            }
            
            Button(action: { viewModel.startDownload() }) {
                HStack(spacing: 12) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 20))
                    Text("Download Models")
                        .font(.system(size: 18, weight: .medium))
                }
                .foregroundColor(.black)
                .frame(maxWidth: 300)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.white)
                )
            }
            .buttonStyle(.plain)
        }
    }
    
    private var downloadingView: some View {
        VStack(spacing: 40) {
            VStack(spacing: 24) {
                downloadProgressBar(
                    title: "CoreML Encoder",
                    progress: viewModel.encoderProgress,
                    bytes: viewModel.encoderBytes
                )
                
                downloadProgressBar(
                    title: "GGML Model",
                    progress: viewModel.modelProgress,
                    bytes: viewModel.modelBytes
                )
            }
            .frame(maxWidth: 500)
        }
    }
    
    private func downloadProgressBar(title: String, progress: Double, bytes: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.white)
                
                Spacer()
                
                Text("\(Int(progress * 100))%")
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.7))
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.1))
                        .frame(height: 8)
                    
                    RoundedRectangle(cornerRadius: 8)
                        .fill(
                            LinearGradient(
                                colors: [Color(red: 0.4, green: 0.8, blue: 1.0), Color(red: 0.6, green: 0.4, blue: 1.0)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * progress, height: 8)
                }
            }
            .frame(height: 8)
            
            if !bytes.isEmpty {
                Text(bytes)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.white.opacity(0.5))
            }
        }
    }
    
    private var loadingView: some View {
        VStack(spacing: 32) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                .scaleEffect(1.5)
            
            VStack(spacing: 12) {
                Text("Compiling CoreML Model")
                    .font(.system(size: 24, weight: .light))
                    .foregroundColor(.white)
                
                Text("This may take a few minutes on first launch")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.6))
            }
        }
    }
    
    private var readyView: some View {
        VStack(spacing: 40) {
            if !viewModel.transcriptionText.isEmpty {
                transcriptionResultView
            }
            
            Button(action: { viewModel.startRecording() }) {
                VStack(spacing: 16) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 64, weight: .light))
                    
                    Text("Start Recording")
                        .font(.system(size: 22, weight: .medium))
                }
                .foregroundColor(.white)
                .frame(width: 240, height: 240)
                .background(
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color(red: 0.3, green: 0.6, blue: 1.0), Color(red: 0.5, green: 0.3, blue: 1.0)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .shadow(color: Color(red: 0.4, green: 0.5, blue: 1.0).opacity(0.5), radius: 30, x: 0, y: 10)
                )
            }
            .buttonStyle(.plain)
            
            Button(action: { isFilePickerPresented = true }) {
                HStack(spacing: 12) {
                    Image(systemName: "doc.fill")
                        .font(.system(size: 18))
                    Text("Select Audio File")
                        .font(.system(size: 16, weight: .medium))
                }
                .foregroundColor(.white)
                .frame(maxWidth: 300)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.white.opacity(0.1))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.white.opacity(0.2), lineWidth: 1)
                        )
                )
            }
            .buttonStyle(.plain)
        }
    }
    
    private var recordingView: some View {
        VStack(spacing: 40) {
            VStack(spacing: 16) {
                Text(formatDuration(viewModel.recordingDuration))
                    .font(.system(size: 56, weight: .thin, design: .monospaced))
                    .foregroundColor(.white)
                
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 12, height: 12)
                        .opacity(isRecordingPulse ? 0.3 : 1.0)
                        .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: isRecordingPulse)
                        .onAppear { isRecordingPulse = true }
                        .onDisappear { isRecordingPulse = false }
                    
                    Text("RECORDING")
                        .font(.system(size: 14, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white.opacity(0.8))
                        .tracking(2)
                }
            }
            
            Button(action: { viewModel.stopRecording() }) {
                VStack(spacing: 16) {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 64, weight: .light))
                    
                    Text("Stop Recording")
                        .font(.system(size: 22, weight: .medium))
                }
                .foregroundColor(.white)
                .frame(width: 240, height: 240)
                .background(
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.red, Color(red: 0.8, green: 0.2, blue: 0.4)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .shadow(color: Color.red.opacity(0.5), radius: 30, x: 0, y: 10)
                )
            }
            .buttonStyle(.plain)
        }
    }
    
    private var transcribingView: some View {
        VStack(spacing: 24) {
            HStack(spacing: 16) {
                ProgressView(value: viewModel.transcriptionProgress)
                    .progressViewStyle(LinearProgressViewStyle(tint: .white))
                
                Text("\(Int(viewModel.transcriptionProgress * 100))%")
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(.white.opacity(0.7))
                    .frame(width: 40, alignment: .trailing)
            }
            .frame(maxWidth: 500)
            
            Text("Transcribing...")
                .font(.system(size: 20, weight: .light))
                .foregroundColor(.white)
            
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Live Transcription")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.white)
                    
                    Spacer()
                    
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.green)
                            .frame(width: 8, height: 8)
                        Text("LIVE")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(.green)
                    }
                }
                
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading) {
                            if viewModel.transcriptionText.isEmpty {
                                Text("Listening...")
                                    .font(.system(size: 16, weight: .regular))
                                    .foregroundColor(.white.opacity(0.4))
                                    .italic()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(20)
                            } else {
                                Text(viewModel.transcriptionText)
                                    .font(.system(size: 16, weight: .regular))
                                    .foregroundColor(.white.opacity(0.9))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(20)
                            }
                            
                            Color.clear
                                .frame(height: 1)
                                .id("transcriptionBottom")
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.white.opacity(0.05))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.1), lineWidth: 1)
                                )
                        )
                    }
                    .frame(maxHeight: 250)
                    .onChange(of: viewModel.transcriptionText) { _ in
                        withAnimation {
                            proxy.scrollTo("transcriptionBottom", anchor: .bottom)
                        }
                    }
                }
            }
            .frame(maxWidth: 600)
            
            Button(action: { viewModel.stopTranscription() }) {
                HStack(spacing: 8) {
                    Image(systemName: "stop.fill")
                        .font(.system(size: 14))
                    Text("Stop Transcription")
                        .font(.system(size: 16, weight: .medium))
                }
                .foregroundColor(.white)
                .padding(.vertical, 12)
                .padding(.horizontal, 24)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.red.opacity(0.3))
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(Color.red.opacity(0.5), lineWidth: 1)
                        )
                )
            }
            .buttonStyle(.plain)
        }
    }
    
    private var transcriptionResultView: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Transcription")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
                
                Spacer()
                
                if !viewModel.transcriptionTime.isEmpty {
                    Text(viewModel.transcriptionTime)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(.white.opacity(0.5))
                }
            }
            
            ScrollView {
                Text(viewModel.transcriptionText)
                    .font(.system(size: 16, weight: .regular))
                    .foregroundColor(.white.opacity(0.9))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(20)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white.opacity(0.05))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.white.opacity(0.1), lineWidth: 1)
                            )
                    )
            }
            .frame(maxHeight: 200)
        }
        .frame(maxWidth: 600)
    }
    
    private var resultView: some View {
        VStack(spacing: 32) {
            transcriptionResultView
            
            HStack(spacing: 16) {
                Button(action: { viewModel.dismissResult() }) {
                    HStack(spacing: 12) {
                        Image(systemName: "mic.fill")
                            .font(.system(size: 20))
                        Text("Record Again")
                            .font(.system(size: 18, weight: .medium))
                    }
                    .foregroundColor(.black)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(.white)
                    )
                }
                .buttonStyle(.plain)
                
                Button(action: {
                    viewModel.dismissResult()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        isFilePickerPresented = true
                    }
                }) {
                    HStack(spacing: 12) {
                        Image(systemName: "doc.fill")
                            .font(.system(size: 20))
                        Text("Select File")
                            .font(.system(size: 18, weight: .medium))
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.white.opacity(0.1))
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                            )
                    )
                }
                .buttonStyle(.plain)
            }
            .frame(maxWidth: 500)
        }
    }
    
    private func errorView(message: String) -> some View {
        VStack(spacing: 24) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 64, weight: .light))
                .foregroundColor(.red.opacity(0.8))
            
            Text("Error")
                .font(.system(size: 28, weight: .light))
                .foregroundColor(.white)
            
            Text(message)
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            Button(action: { viewModel.checkExistingModels() }) {
                Text("Retry")
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(.black)
                    .frame(width: 120)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(.white)
                    )
            }
            .buttonStyle(.plain)
        }
    }
    
    private var footerView: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .font(.system(size: 12))
            Text("Powered by whisper.cpp")
                .font(.system(size: 12, weight: .medium))
        }
        .foregroundColor(.white.opacity(0.4))
    }
    
    private var stateDescription: String {
        switch viewModel.state {
        case .needsDownload: return "Ready to Download"
        case .downloading: return "Downloading Models"
        case .loadingModel: return "Loading Model"
        case .ready: return "Ready"
        case .recording: return "Recording"
        case .transcribing: return "Transcribing"
        case .result: return "Complete"
        case .error: return "Error"
        }
    }
    
    private func formatDuration(_ duration: TimeInterval) -> String {
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60
        let milliseconds = Int((duration.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%02d:%02d.%01d", minutes, seconds, milliseconds)
    }
}
