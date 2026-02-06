import Foundation
import zlib

/// A minimal pure-Swift zip extractor using zlib (available on iOS).
/// Uses a two-pass approach: first parses the Central Directory for accurate sizes,
/// then walks local file headers to extract. Handles ZIPs with data descriptors
/// (general purpose flag bit 3) where local headers have compressedSize=0.
///
/// Memory-efficient: uses memory-mapped file access and streams decompression
/// directly to disk, keeping peak memory usage at ~2-5MB regardless of file size.
enum ZipExtractor {
    
    enum ZipError: Error, LocalizedError {
        case invalidZipFile
        case unsupportedCompression(UInt16)
        case decompressionFailed
        case fileCreationFailed(String)
        case centralDirectoryNotFound
        
        var errorDescription: String? {
            switch self {
            case .invalidZipFile: return "Invalid or corrupt ZIP file"
            case .unsupportedCompression(let method): return "Unsupported compression method: \(method)"
            case .decompressionFailed: return "Failed to decompress ZIP entry"
            case .fileCreationFailed(let path): return "Failed to create file: \(path)"
            case .centralDirectoryNotFound: return "ZIP central directory not found"
            }
        }
    }
    
    private struct CentralDirectoryEntry {
        let compressionMethod: UInt16
        let compressedSize: UInt64
        let uncompressedSize: UInt64
    }
    
    /// Extract a ZIP file to its parent directory, then remove the ZIP.
    /// Reports extraction progress (0.0–1.0) via the optional callback.
    static func extract(zipAt zipURL: URL, progress: ((Double) -> Void)? = nil) throws {
        let parentDir = zipURL.deletingLastPathComponent()
        let zipData = try Data(contentsOf: zipURL, options: .alwaysMapped)
        let count = zipData.count
        
        // Pass 1: Parse Central Directory for accurate sizes
        let cdEntries = try parseCentralDirectory(data: zipData, count: count)
        
        // Calculate total uncompressed size for progress reporting
        let totalSize = cdEntries.values.reduce(UInt64(0)) { $0 + $1.uncompressedSize }
        var extractedSize: UInt64 = 0
        
        // Pass 2: Walk local file headers, use CD sizes
        var offset = 0
        while offset + 30 <= count {
            // Check for local file header signature: PK\x03\x04
            guard zipData[offset] == 0x50,
                  zipData[offset + 1] == 0x4B,
                  zipData[offset + 2] == 0x03,
                  zipData[offset + 3] == 0x04 else {
                break // No more local file headers
            }
            
            // Read flags from local header (offset + 6, 2 bytes LE)
            let flags = readUInt16(zipData, offset + 6)
            let localCompressedSize = readUInt32(zipData, offset + 18)
            let fileNameLength = Int(readUInt16(zipData, offset + 26))
            let extraFieldLength = Int(readUInt16(zipData, offset + 28))
            
            let fileNameStart = offset + 30
            let fileNameEnd = fileNameStart + fileNameLength
            guard fileNameEnd <= count else { throw ZipError.invalidZipFile }
            
            guard let fileName = String(data: zipData[fileNameStart..<fileNameEnd], encoding: .utf8) else {
                throw ZipError.invalidZipFile
            }
            
            // Look up real sizes from Central Directory
            let cdEntry = cdEntries[fileName]
            let compressedSize = cdEntry?.compressedSize ?? UInt64(localCompressedSize)
            let uncompressedSize = cdEntry?.uncompressedSize ?? 0
            let compressionMethod = cdEntry?.compressionMethod ?? readUInt16(zipData, offset + 8)
            
            let dataStart = fileNameEnd + extraFieldLength
            let dataEnd = dataStart + Int(compressedSize)
            guard dataEnd <= count else { throw ZipError.invalidZipFile }
            
            // Skip __MACOSX resource fork entries
            if fileName.hasPrefix("__MACOSX/") {
                offset = dataEnd
                // Skip data descriptor if bit 3 set
                if flags & 0x0008 != 0 {
                    if offset + 4 <= count && readUInt32(zipData, offset) == 0x08074b50 {
                        offset += 16 // sig(4) + crc(4) + compSize(4) + uncompSize(4)
                    } else {
                        offset += 12 // crc(4) + compSize(4) + uncompSize(4)
                    }
                }
                continue
            }
            
            let filePath = parentDir.appendingPathComponent(fileName)
            
            if fileName.hasSuffix("/") {
                // Directory entry
                try FileManager.default.createDirectory(at: filePath, withIntermediateDirectories: true)
            } else {
                // File entry — ensure parent directory exists
                let fileDir = filePath.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)
                
                switch compressionMethod {
                case 0: // Stored — write directly from mapped data (no decompression)
                    guard FileManager.default.createFile(atPath: filePath.path, contents: zipData[dataStart..<dataEnd]) else {
                        throw ZipError.fileCreationFailed(filePath.path)
                    }
                case 8: // Deflate — stream decompress directly to disk
                    try inflateToFile(
                        from: zipData,
                        dataOffset: dataStart,
                        compressedSize: Int(compressedSize),
                        to: filePath.path
                    )
                default:
                    throw ZipError.unsupportedCompression(compressionMethod)
                }
                
                extractedSize += uncompressedSize
                if totalSize > 0 {
                    progress?(Double(extractedSize) / Double(totalSize))
                }
            }
            
            offset = dataEnd
            
            // Skip data descriptor if bit 3 set
            if flags & 0x0008 != 0 {
                if offset + 4 <= count && readUInt32(zipData, offset) == 0x08074b50 {
                    offset += 16 // sig(4) + crc(4) + compSize(4) + uncompSize(4)
                } else {
                    offset += 12 // crc(4) + compSize(4) + uncompSize(4)
                }
            }
        }
        
        // Clean up the zip file
        try FileManager.default.removeItem(at: zipURL)
        progress?(1.0)
        print("[ZipExtractor] Cleaned up zip: \(zipURL.lastPathComponent)")
    }
    
    // MARK: - Central Directory Parser
    
    private static func parseCentralDirectory(data: Data, count: Int) throws -> [String: CentralDirectoryEntry] {
        // Find End of Central Directory record (search backwards for 0x06054b50)
        var eocdOffset = -1
        for i in stride(from: count - 22, through: max(0, count - 65557), by: -1) {
            if data[i] == 0x50 && data[i + 1] == 0x4B && data[i + 2] == 0x05 && data[i + 3] == 0x06 {
                eocdOffset = i
                break
            }
        }
        guard eocdOffset >= 0 else { throw ZipError.centralDirectoryNotFound }
        
        let cdOffset = Int(readUInt32(data, eocdOffset + 16))
        let cdEntryCount = Int(readUInt16(data, eocdOffset + 10))
        
        var entries: [String: CentralDirectoryEntry] = [:]
        var pos = cdOffset
        
        for _ in 0..<cdEntryCount {
            guard pos + 46 <= count else { break }
            guard data[pos] == 0x50 && data[pos + 1] == 0x4B && data[pos + 2] == 0x01 && data[pos + 3] == 0x02 else { break }
            
            let method = readUInt16(data, pos + 10)
            let cSize = UInt64(readUInt32(data, pos + 20))
            let uSize = UInt64(readUInt32(data, pos + 24))
            let fnLen = Int(readUInt16(data, pos + 28))
            let exLen = Int(readUInt16(data, pos + 30))
            let cmLen = Int(readUInt16(data, pos + 32))
            
            let fnStart = pos + 46
            guard fnStart + fnLen <= count else { break }
            if let name = String(data: data[fnStart..<fnStart + fnLen], encoding: .utf8) {
                entries[name] = CentralDirectoryEntry(compressionMethod: method, compressedSize: cSize, uncompressedSize: uSize)
            }
            
            pos = fnStart + fnLen + exLen + cmLen
        }
        
        return entries
    }
    
    // MARK: - Binary Helpers
    
    private static func readUInt16(_ data: Data, _ offset: Int) -> UInt16 {
        UInt16(data[offset]) | (UInt16(data[offset + 1]) << 8)
    }
    
    private static func readUInt32(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(data[offset]) | (UInt32(data[offset + 1]) << 8) | (UInt32(data[offset + 2]) << 16) | (UInt32(data[offset + 3]) << 24)
    }
    
    // MARK: - Streaming Decompression
    
    /// Stream-decompress deflated data directly to a file on disk.
    /// Uses constant ~2MB of memory regardless of file size.
    private static func inflateToFile(
        from zipData: Data,
        dataOffset: Int,
        compressedSize: Int,
        to filePath: String
    ) throws {
        guard FileManager.default.createFile(atPath: filePath, contents: nil) else {
            throw ZipError.fileCreationFailed(filePath)
        }
        guard let fileHandle = FileHandle(forWritingAtPath: filePath) else {
            throw ZipError.fileCreationFailed(filePath)
        }
        defer { try? fileHandle.close() }
        
        var stream = z_stream()
        
        // -MAX_WBITS for raw deflate (no zlib/gzip header)
        guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else {
            throw ZipError.decompressionFailed
        }
        defer { inflateEnd(&stream) }
        
        let outputChunkSize = 1024 * 1024 // 1MB output buffer
        let outputBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: outputChunkSize)
        defer { outputBuffer.deallocate() }
        
        let inputChunkSize = 1024 * 1024 // 1MB input chunks
        var inputOffset = dataOffset
        let inputEnd = dataOffset + compressedSize
        
        while inputOffset < inputEnd {
            let thisInputSize = min(inputChunkSize, inputEnd - inputOffset)
            
            try zipData.withUnsafeBytes { rawPtr in
                let basePtr = rawPtr.baseAddress!.assumingMemoryBound(to: UInt8.self)
                stream.next_in = UnsafeMutablePointer(mutating: basePtr + inputOffset)
                stream.avail_in = uInt(thisInputSize)
                
                repeat {
                    stream.next_out = outputBuffer
                    stream.avail_out = uInt(outputChunkSize)
                    
                    let ret = zlib.inflate(&stream, Z_NO_FLUSH)
                    
                    let bytesProduced = outputChunkSize - Int(stream.avail_out)
                    if bytesProduced > 0 {
                        fileHandle.write(Data(bytesNoCopy: outputBuffer, count: bytesProduced, deallocator: .none))
                    }
                    
                    if ret == Z_STREAM_END {
                        return
                    }
                    guard ret == Z_OK || ret == Z_BUF_ERROR else {
                        throw ZipError.decompressionFailed
                    }
                } while stream.avail_in > 0
            }
            
            inputOffset += thisInputSize
        }
    }
}
