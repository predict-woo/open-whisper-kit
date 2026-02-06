// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OpenWhisperKitDemo",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(name: "open-whisper-kit", path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "OpenWhisperKitDemo",
            dependencies: [
                .product(name: "OpenWhisperKit", package: "open-whisper-kit")
            ],
            path: "Sources"
        )
    ]
)
