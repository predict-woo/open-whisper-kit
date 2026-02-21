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
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.3.1/whisper.xcframework.zip",
            checksum: "ce41d6bc2e8ea29915da7f0539091bd28c2af339e88170d1153c8ff587f14b80"
        ),
        .binaryTarget(
            name: "sortformer",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.3.1/sortformer.xcframework.zip",
            checksum: "493f09806a48d8c28bba9a612eea75d369c3a594717f0f603d35910aad135ae6"
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
