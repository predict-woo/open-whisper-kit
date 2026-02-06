import Foundation
import zlib

/// A minimal pure-Swift zip extractor using zlib (available on iOS).
/// Handles standard ZIP files with Deflate or Stored compression.
enum ZipExtractor {
    
    enum ZipError: Error, LocalizedError {
        case invalidZipFile
        case unsupportedCompression(UInt16)
        case decompressionFailed
        case fileCreationFailed(String)
        
        var errorDescription: String? {
            switch self {
            case .invalidZipFile: return "Invalid or corrupt ZIP file"
            case .unsupportedCompression(let method): return "Unsupported compression method: \(method)"
            case .decompressionFailed: return "Failed to decompress ZIP entry"
            case .fileCreationFailed(let path): return "Failed to create file: \(path)"
            }
        }
    }
    
    /// Extract a ZIP file to its parent directory, then remove the ZIP.
    static func extract(zipAt zipURL: URL) throws {
        let parentDir = zipURL.deletingLastPathComponent()
        let zipData = try Data(contentsOf: zipURL)
        
        var offset = 0
        let bytes = [UInt8](zipData)
        let count = bytes.count
        
        while offset + 30 <= count {
            // Check for local file header signature: PK\x03\x04
            guard bytes[offset] == 0x50,
                  bytes[offset + 1] == 0x4B,
                  bytes[offset + 2] == 0x03,
                  bytes[offset + 3] == 0x04 else {
                break // No more local file headers
            }
            
            let compressionMethod = UInt16(bytes[offset + 8]) | (UInt16(bytes[offset + 9]) << 8)
            let compressedSize = UInt32(bytes[offset + 18]) | (UInt32(bytes[offset + 19]) << 8) | (UInt32(bytes[offset + 20]) << 16) | (UInt32(bytes[offset + 21]) << 24)
            let uncompressedSize = UInt32(bytes[offset + 22]) | (UInt32(bytes[offset + 23]) << 8) | (UInt32(bytes[offset + 24]) << 16) | (UInt32(bytes[offset + 25]) << 24)
            let fileNameLength = Int(UInt16(bytes[offset + 26]) | (UInt16(bytes[offset + 27]) << 8))
            let extraFieldLength = Int(UInt16(bytes[offset + 28]) | (UInt16(bytes[offset + 29]) << 8))
            
            let fileNameStart = offset + 30
            let fileNameEnd = fileNameStart + fileNameLength
            
            guard fileNameEnd <= count else { throw ZipError.invalidZipFile }
            
            let fileNameBytes = Array(bytes[fileNameStart..<fileNameEnd])
            guard let fileName = String(bytes: fileNameBytes, encoding: .utf8) else {
                throw ZipError.invalidZipFile
            }
            
            let dataStart = fileNameEnd + extraFieldLength
            let dataEnd = dataStart + Int(compressedSize)
            
            guard dataEnd <= count else { throw ZipError.invalidZipFile }
            
            let filePath = parentDir.appendingPathComponent(fileName)
            
            if fileName.hasSuffix("/") {
                // Directory entry
                try FileManager.default.createDirectory(at: filePath, withIntermediateDirectories: true)
            } else {
                // File entry — ensure parent directory exists
                let fileDir = filePath.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)
                
                let compressedData = Data(bytes[dataStart..<dataEnd])
                let fileData: Data
                
                switch compressionMethod {
                case 0: // Stored
                    fileData = compressedData
                case 8: // Deflate
                    fileData = try inflate(compressedData, expectedSize: Int(uncompressedSize))
                default:
                    throw ZipError.unsupportedCompression(compressionMethod)
                }
                
                guard FileManager.default.createFile(atPath: filePath.path, contents: fileData) else {
                    throw ZipError.fileCreationFailed(filePath.path)
                }
            }
            
            offset = dataEnd
        }
        
        // Clean up the zip file
        try FileManager.default.removeItem(at: zipURL)
        print("[ZipExtractor] Cleaned up zip: \(zipURL.lastPathComponent)")
    }
    
    /// Decompress raw deflate data using zlib.
    private static func inflate(_ data: Data, expectedSize: Int) throws -> Data {
        var stream = z_stream()
        
        let inputBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: data.count)
        data.copyBytes(to: inputBuffer, count: data.count)
        defer { inputBuffer.deallocate() }
        
        let outputBufferSize = max(expectedSize, 4096)
        let outputBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: outputBufferSize)
        defer { outputBuffer.deallocate() }
        
        stream.next_in = inputBuffer
        stream.avail_in = uInt(data.count)
        stream.next_out = outputBuffer
        stream.avail_out = uInt(outputBufferSize)
        
        // -MAX_WBITS for raw deflate (no zlib/gzip header)
        guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw ZipError.decompressionFailed
        }
        defer { inflateEnd(&stream) }
        
        var result = Data()
        
        while true {
            stream.next_out = outputBuffer
            stream.avail_out = uInt(outputBufferSize)
            
            let ret = zlib.inflate(&stream, Z_NO_FLUSH)
            
            let bytesProduced = outputBufferSize - Int(stream.avail_out)
            if bytesProduced > 0 {
                result.append(outputBuffer, count: bytesProduced)
            }
            
            if ret == Z_STREAM_END {
                break
            }
            
            guard ret == Z_OK || ret == Z_BUF_ERROR else {
                throw ZipError.decompressionFailed
            }
        }
        
        return result
    }
}
