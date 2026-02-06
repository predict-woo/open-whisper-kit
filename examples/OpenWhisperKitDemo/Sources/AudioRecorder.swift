import Foundation
import AVFoundation

actor AudioRecorder {
    private var recorder: AVAudioRecorder?
    private var recordingURL: URL?
    
    func startRecording(toOutputFile url: URL) throws {
        let recordSettings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000.0,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]
        
        #if !os(macOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default)
        try session.setActive(true)
        #endif
        
        let recorder = try AVAudioRecorder(url: url, settings: recordSettings)
        if !recorder.record() {
            throw RecorderError.failedToStart
        }
        
        self.recorder = recorder
        self.recordingURL = url
    }
    
    func stopRecording() -> URL? {
        recorder?.stop()
        recorder = nil
        
        #if !os(macOS)
        try? AVAudioSession.sharedInstance().setActive(false)
        #endif
        
        let url = recordingURL
        recordingURL = nil
        return url
    }
    
    func isRecording() -> Bool {
        return recorder?.isRecording ?? false
    }
}

enum RecorderError: Error {
    case failedToStart
}
