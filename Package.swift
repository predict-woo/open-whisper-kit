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
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.1.0/whisper.xcframework.zip",
            checksum: "97e7a8eee754cf2a7daa6c07d49899abe1831eb884572dd17c58b114173331c9"
        ),
        .binaryTarget(
            name: "sortformer",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.1.0/sortformer.xcframework.zip",
            checksum: "1d656a03fe3693551c2ade36a5b0ce0ee159d063b26993718377173440d6f9d1"
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
