import Foundation

struct DownloadProgress: Sendable {
    let bytesDownloaded: Int64
    let totalBytes: Int64
    let progress: Double
}

// IMPORTANT: This MUST be a class, NOT an actor. URLSessionDownloadDelegate's
// didFinishDownloadingTo provides a temp file deleted when the callback returns.
// An actor hop would cause the temp file to vanish before we can move it.
final class DownloadManager: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var session: URLSession!
    private var progressContinuations: [URLSessionTask: AsyncStream<DownloadProgress>.Continuation] = [:]
    private let lock = NSLock()

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    // MARK: - Public API

    func download(from url: URL) async throws -> AsyncStream<DownloadProgress> {
        let (stream, continuation) = AsyncStream.makeStream(of: DownloadProgress.self)

        let task = session.downloadTask(with: url)
        storeContinuation(continuation, for: task)

        print("[DownloadManager] Starting download: \(url.lastPathComponent)")
        task.resume()

        return stream
    }

    private nonisolated func storeContinuation(
        _ continuation: AsyncStream<DownloadProgress>.Continuation,
        for task: URLSessionTask
    ) {
        lock.lock()
        progressContinuations[task] = continuation
        lock.unlock()
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let originalURL = downloadTask.originalRequest?.url,
              let destination = getDestination(for: originalURL) else {
            print("[DownloadManager] ERROR: Could not determine destination for download")
            lock.lock()
            let continuation = progressContinuations.removeValue(forKey: downloadTask)
            lock.unlock()
            continuation?.finish()
            return
        }

        print("[DownloadManager] Download complete: \(originalURL.lastPathComponent)")
        print("[DownloadManager] Temp file: \(location.path)")
        print("[DownloadManager] Destination: \(destination.path)")

        do {
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }

            try FileManager.default.moveItem(at: location, to: destination)
            print("[DownloadManager] File moved to: \(destination.path)")

            if destination.pathExtension == "zip" {
                print("[DownloadManager] Unzipping on background thread: \(destination.lastPathComponent)")
                lock.lock()
                let continuation = progressContinuations.removeValue(forKey: downloadTask)
                lock.unlock()
                let bytesReceived = downloadTask.countOfBytesReceived
                let bytesExpected = downloadTask.countOfBytesExpectedToReceive

                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try self.unzipFile(at: destination)
                        print("[DownloadManager] Unzip completed")
                    } catch {
                        print("[DownloadManager] Unzip ERROR: \(error)")
                    }
                    continuation?.yield(DownloadProgress(
                        bytesDownloaded: bytesReceived,
                        totalBytes: bytesExpected,
                        progress: 1.0
                    ))
                    continuation?.finish()
                    print("[DownloadManager] Stream finished: \(originalURL.lastPathComponent)")
                }
            } else {
                lock.lock()
                let continuation = progressContinuations.removeValue(forKey: downloadTask)
                lock.unlock()

                continuation?.yield(DownloadProgress(
                    bytesDownloaded: downloadTask.countOfBytesReceived,
                    totalBytes: downloadTask.countOfBytesExpectedToReceive,
                    progress: 1.0
                ))
                continuation?.finish()
                print("[DownloadManager] Stream finished: \(originalURL.lastPathComponent)")
            }

        } catch {
            print("[DownloadManager] ERROR moving file: \(error)")
            lock.lock()
            let continuation = progressContinuations.removeValue(forKey: downloadTask)
            lock.unlock()
            continuation?.finish()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        lock.lock()
        let continuation = progressContinuations[downloadTask]
        lock.unlock()

        guard let continuation = continuation else { return }

        let progress = totalBytesExpectedToWrite > 0
            ? Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
            : 0.0

        let percentage = Int(progress * 100)
        if percentage > 0 && percentage % 25 == 0 {
            let taskName = downloadTask.originalRequest?.url?.lastPathComponent ?? "unknown"
            print("[DownloadManager] \(taskName): \(percentage)%")
        }

        continuation.yield(DownloadProgress(
            bytesDownloaded: totalBytesWritten,
            totalBytes: totalBytesExpectedToWrite,
            progress: progress
        ))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            print("[DownloadManager] Task failed: \(error.localizedDescription)")
            lock.lock()
            let continuation = progressContinuations.removeValue(forKey: task)
            lock.unlock()
            continuation?.finish()
        }
    }

    // MARK: - Private

    private func getDestination(for url: URL) -> URL? {
        guard let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            print("[DownloadManager] ERROR: Could not find Documents directory")
            return nil
        }
        return documentsDir.appendingPathComponent(url.lastPathComponent)
    }

    private func unzipFile(at zipURL: URL) throws {
        let parentDir = zipURL.deletingLastPathComponent()
        let expectedDir = parentDir.appendingPathComponent(
            zipURL.deletingPathExtension().lastPathComponent
        )

        if FileManager.default.fileExists(atPath: expectedDir.path) {
            try FileManager.default.removeItem(at: expectedDir)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        process.arguments = ["-o", "-q", zipURL.path, "-d", parentDir.path]
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "DownloadManager",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: "unzip exited with status \(process.terminationStatus)"]
            )
        }

        try FileManager.default.removeItem(at: zipURL)
        print("[DownloadManager] Cleaned up zip: \(zipURL.lastPathComponent)")
    }
}
