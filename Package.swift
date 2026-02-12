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
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.2.0/whisper.xcframework.zip",
            checksum: "246a48e15a0d3e6303e33907814af9fc783db6af7a88aa6ed03077cb44eb2955"
        ),
        .binaryTarget(
            name: "sortformer",
            url: "https://github.com/predict-woo/open-whisper-kit/releases/download/v1.2.0/sortformer.xcframework.zip",
            checksum: "e4da36052bead5b0aaff11885810f5a016ad315d67aa92001f7bbe5b7e021c09"
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
