// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenWhisperKit",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "OpenWhisperKit",
            targets: ["OpenWhisperKit"]
        )
    ],
    targets: [
        .binaryTarget(
            name: "whisper",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.3.0/whisper.xcframework.zip",
            checksum: "053f5efe4ba582bca15ffbb14b9027bd9fba93f23cc8842ee42c84814da9e49e"
        ),
        .binaryTarget(
            name: "sortformer",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.3.0/sortformer.xcframework.zip",
            checksum: "60076cfca6e906fb449adbaadc8c2d0a0784c66b2bd90587c52fee97a941a5ad"
        ),
        .target(
            name: "OpenWhisperKit",
            dependencies: ["whisper", "sortformer"],
            path: "Sources/OpenWhisperKit"
        ),
        .executableTarget(
            name: "diarize-cli",
            dependencies: ["OpenWhisperKit"],
            path: "Sources/diarize-cli"
        ),
        .testTarget(
            name: "OpenWhisperKitTests",
            dependencies: ["OpenWhisperKit"],
            path: "Tests/OpenWhisperKitTests"
        )
    ]
)
