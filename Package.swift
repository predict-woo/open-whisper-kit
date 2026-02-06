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
            path: "build-apple/whisper.xcframework"
        ),
        .binaryTarget(
            name: "sortformer",
            path: "build-apple/sortformer.xcframework"
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
