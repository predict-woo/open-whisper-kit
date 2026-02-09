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
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.0.0/whisper.xcframework.zip",
            checksum: "1000123f4f9c06069c968b2d8b78dd0049dc612564819b7196b93047db365158"
        ),
        .binaryTarget(
            name: "sortformer",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.0.0/sortformer.xcframework.zip",
            checksum: "3601ac8800fe300e3df4c78f405ab90f5219bd973bb2b9ab72f905269aedad8b"
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
