// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VoiceAgentOrb",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
        .macCatalyst(.v18),
        .tvOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(
            name: "VoiceAgentOrb",
            targets: ["VoiceAgentOrb"]
        ),
    ],
    targets: [
        .target(
            name: "VoiceAgentOrb",
            resources: [
                .process("Resources"),
                .process("Shaders")
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6)
            ]
        ),
    ]
)
