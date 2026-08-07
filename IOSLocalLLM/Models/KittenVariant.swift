import Foundation

enum KittenPrecision: String, Codable, Sendable {
    case fp32
    case int8
}

enum KittenVariant: String, CaseIterable, Codable, Identifiable, Sendable {
    case micro08 = "tts.kitten.micro.0_8"
    case nano08Int8 = "tts.kitten.nano.0_8.int8"
    case mini08 = "tts.kitten.mini.0_8"

    var id: String { rawValue }

    var directoryName: String {
        switch self {
        case .micro08: "micro-0.8"
        case .nano08Int8: "nano-0.8-int8"
        case .mini08: "mini-0.8"
        }
    }
}

struct KittenArtifact: Codable, Equatable, Sendable {
    let relativePath: String
    let expectedBytes: Int64
    let sha256: String
    let required: Bool
}

struct KittenVariantConfiguration: Codable, Equatable, Identifiable, Sendable {
    let id: KittenVariant
    let familyID: String
    let repositoryID: String
    let revision: String
    let modelFileName: String
    let voiceFileName: String
    let precision: KittenPrecision
    let sampleRate: Double
    let minimumRuntimeVersion: String
    let artifacts: [KittenArtifact]

    var installedDirectory: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documents.appendingPathComponent("VoiceModels", isDirectory: true)
            .appendingPathComponent("KittenTTS", isDirectory: true)
            .appendingPathComponent(id.directoryName, isDirectory: true)
    }

    var modelURL: URL { installedDirectory.appendingPathComponent(modelFileName) }
    var voiceDataURL: URL { installedDirectory.appendingPathComponent(voiceFileName) }
}

enum KittenManifest {
    static let configurations: [KittenVariantConfiguration] = [
        .init(
            id: .micro08,
            familyID: "tts.kitten",
            repositoryID: "KittenML/kitten-tts-micro-0.8",
            revision: "1ccf72b2c2048fd17efac7de2fab32d10e225084",
            modelFileName: "kitten_tts_micro_v0_8.onnx",
            voiceFileName: "voices.npz",
            precision: .fp32,
            sampleRate: 24_000,
            minimumRuntimeVersion: "1.23.0",
            artifacts: [
                .init(relativePath: "kitten_tts_micro_v0_8.onnx", expectedBytes: 41_384_970, sha256: "95481626fee1ba70ce683e69c534fc7cb38433c46ce42d3abbeafb4b9f1a4123", required: true),
                .init(relativePath: "voices.npz", expectedBytes: 3_278_902, sha256: "112710c1be8ad0e967c190fb0fd95cbe5848ec4791b93209f20b28b7da20dac1", required: true),
            ]
        ),
        .init(
            id: .nano08Int8,
            familyID: "tts.kitten",
            repositoryID: "KittenML/kitten-tts-nano-0.8-int8",
            revision: "84781d74e29ee25217551556398b42f80593a813",
            modelFileName: "kitten_tts_nano_v0_8.onnx",
            voiceFileName: "voices.npz",
            precision: .int8,
            sampleRate: 24_000,
            minimumRuntimeVersion: "1.23.0",
            artifacts: [
                .init(relativePath: "kitten_tts_nano_v0_8.onnx", expectedBytes: 24_369_971, sha256: "f7b0afcbee92870b32b8e0276d855b954dc25470c9f051b376ac7eee537c76fc", required: true),
                .init(relativePath: "voices.npz", expectedBytes: 3_278_902, sha256: "8aa7cee235abb0739cb51e6559685f65a4dacd95568833d05699b1633f519b3f", required: true),
            ]
        ),
        .init(
            id: .mini08,
            familyID: "tts.kitten",
            repositoryID: "KittenML/kitten-tts-mini-0.8",
            revision: "c02725660cea441db4c383af69f1f26f5cd00947",
            modelFileName: "kitten_tts_mini_v0_8.onnx",
            voiceFileName: "voices.npz",
            precision: .fp32,
            sampleRate: 24_000,
            minimumRuntimeVersion: "1.23.0",
            artifacts: [
                .init(relativePath: "kitten_tts_mini_v0_8.onnx", expectedBytes: 78_268_016, sha256: "0f5bbae4fc4800c98dbc544a87ecfa79510de2fb8222db30d12e5bfe9177df91", required: true),
                .init(relativePath: "voices.npz", expectedBytes: 3_278_902, sha256: "40ad2638952b77b7b2f30127e2608e169fc69dd256b53bd8aaa3409a33193c42", required: true),
            ]
        ),
    ]

    static func configuration(for variant: KittenVariant) -> KittenVariantConfiguration {
        configurations.first { $0.id == variant }!
    }
}
