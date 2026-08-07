import Foundation

enum ModelToolParser: String, Codable, CaseIterable, Sendable {
    case none
    case hermes
    case qwen3Coder = "qwen3_coder"
    case qwen3XML = "qwen3_xml"
    case foundationModels = "foundation_models"

    var displayName: String {
        switch self {
        case .none: return "None"
        case .hermes: return "Hermes"
        case .qwen3Coder: return "Qwen3 Coder"
        case .qwen3XML: return "Qwen3 XML"
        case .foundationModels: return "Foundation Models"
        }
    }
}

enum ModelReasoningParser: String, Codable, CaseIterable, Sendable {
    case none
    case deepSeekR1 = "deepseek_r1"
    case qwen3

    var displayName: String {
        switch self {
        case .none: return "None"
        case .deepSeekR1: return "DeepSeek R1"
        case .qwen3: return "Qwen3"
        }
    }

    /// Chat templates for these parsers typically pre-fill the opening
    /// reasoning marker, so the model only emits a closing tag.
    var assumesPrefilledOpening: Bool {
        switch self {
        case .none: return false
        case .deepSeekR1, .qwen3: return true
        }
    }
}

enum ModelParallelToolPolicy: Equatable, Sendable {
    case sequential
    case automatic(maximumCalls: Int)

    var maximumCalls: Int {
        switch self {
        case .sequential: return 1
        case .automatic(let maximumCalls): return maximumCalls
        }
    }

    var displayName: String {
        switch self {
        case .sequential: return "Sequential"
        case .automatic(let maximumCalls): return "Auto · up to \(maximumCalls)"
        }
    }
}

enum LocalAPIParallelToolCallsSetting: Int, CaseIterable, Identifiable, Equatable, Sendable {
    case sequential = 1
    case automaticTwo = 2
    case automaticThree = 3
    case automaticFour = 4

    var id: Int { rawValue }

    var maximumCalls: Int { rawValue }

    var displayName: String {
        switch self {
        case .sequential: return "Sequential"
        case .automaticTwo: return "Auto · up to 2"
        case .automaticThree: return "Auto · up to 3"
        case .automaticFour: return "Auto · up to 4"
        }
    }

    var allowsParallelCalls: Bool {
        self != .sequential
    }
}

/// One source of truth for the model-facing context, parser, and tool policy.
/// The same profile drives MLX generation, `/v1/models`, request validation,
/// and the Home compatibility card so the server never advertises a context
/// window that its active KV cache cannot retain.
struct ModelCapabilityProfile: Equatable, Sendable {
    static let hermesMinimumContextTokens = 65_536

    enum Family: String, Sendable {
        case qwen3
        case qwen35 = "qwen3.5"
        case ornith
        case bonsai
        case generic

        var displayName: String {
            switch self {
            case .qwen3: return "Qwen3"
            case .qwen35: return "Qwen3.5"
            case .ornith: return "Ornith"
            case .bonsai: return "Bonsai"
            case .generic: return "Generic"
            }
        }
    }

    let family: Family
    /// Published architecture limit. This can be larger than the practical
    /// on-device allocation.
    let modelContextLength: Int
    /// Context the app really retains for this runtime.
    let configuredContextLength: Int
    let maximumKVCacheTokens: Int
    let maximumOutputTokens: Int
    let kvBits: Int
    let prefillStepSize: Int
    let toolParser: ModelToolParser
    let reasoningParser: ModelReasoningParser
    let parallelTools: ModelParallelToolPolicy
    let supportsThinkingToggle: Bool
    let usesChatTemplateOverride: Bool
    let requiresYaRNContextExtension: Bool
    /// Conservative estimate used for an understandable UI warning. MLX
    /// allocates lazily, so this is the fully-populated cache estimate rather
    /// than memory charged immediately when the model loads.
    let estimatedKVCacheBytes: Int64

    /// Whether the OpenAI streaming path should start inside a reasoning
    /// block (template-prefilled open). Ornith uses an explicit
    /// `Thinking Process:` marker instead, so it stays on the implicit path.
    var streamAssumesPrefilledReasoningOpen: Bool {
        guard reasoningParser.assumesPrefilledOpening else { return false }
        return family != .ornith
    }

    var isHermesContextCompatible: Bool {
        configuredContextLength >= Self.hermesMinimumContextTokens
            && maximumKVCacheTokens >= Self.hermesMinimumContextTokens
            && toolParser != .none
    }

    var hermesWarning: String? {
        guard !isHermesContextCompatible else { return nil }
        return "Hermes agents require at least 65,536 real context tokens. This profile is configured for \(configuredContextLength.formatted())."
    }

    func effectiveParallelLimit(
        requestAllowsParallel: Bool,
        globalEnabled: Bool,
        configuredMaximumCalls: Int? = nil
    ) -> Int {
        guard requestAllowsParallel, globalEnabled else { return 1 }
        let configuredLimit = max(1, configuredMaximumCalls ?? parallelTools.maximumCalls)
        return min(parallelTools.maximumCalls, configuredLimit)
    }

    static func resolve(for model: AssistantModel) -> Self {
        resolve(
            repoID: model.repoID,
            catalogContextLength: model.contextWindowTokens,
            supportsThinking: model.supportsThinking
        )
    }

    static func resolve(
        repoID: String,
        catalogContextLength: Int = 32_768,
        supportsThinking: Bool = false
    ) -> Self {
        let identity = repoID.lowercased()

        // Bonsai must be checked before Qwen3.5 because some imported folders
        // expose the base architecture in their local path. Its smaller and
        // ternary variants have not been reliable with parallel agent calls.
        if identity.contains("bonsai") {
            let context = min(max(2_048, catalogContextLength), 32_768)
            return .init(
                family: .bonsai,
                modelContextLength: max(context, catalogContextLength),
                configuredContextLength: context,
                maximumKVCacheTokens: context,
                maximumOutputTokens: 2_048,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3Coder,
                reasoningParser: .qwen3,
                parallelTools: .sequential,
                supportsThinkingToggle: supportsThinking,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: Int64(context) * 16_384
            )
        }

        if identity.contains("ornith") {
            return .init(
                family: .ornith,
                modelContextLength: 262_144,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3XML,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: 1_100_000_000
            )
        }

        if identity.contains("qwen3.5") || identity.contains("qwen3_5") {
            return .init(
                family: .qwen35,
                modelContextLength: 262_144,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .qwen3Coder,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: false,
                estimatedKVCacheBytes: 1_100_000_000
            )
        }

        if identity.contains("qwen3") {
            let bytesPerToken: Int64
            if identity.contains("8b") {
                bytesPerToken = 65_536
            } else if identity.contains("4b") {
                bytesPerToken = 32_768
            } else {
                bytesPerToken = 16_384
            }
            return .init(
                family: .qwen3,
                modelContextLength: 131_072,
                configuredContextLength: 65_536,
                maximumKVCacheTokens: 65_536,
                maximumOutputTokens: 4_096,
                kvBits: 4,
                prefillStepSize: 128,
                toolParser: .hermes,
                reasoningParser: .qwen3,
                parallelTools: .automatic(maximumCalls: 2),
                supportsThinkingToggle: true,
                usesChatTemplateOverride: true,
                requiresYaRNContextExtension: true,
                estimatedKVCacheBytes: Int64(65_536) * bytesPerToken
            )
        }

        let context = max(2_048, catalogContextLength)
        return .init(
            family: .generic,
            modelContextLength: context,
            configuredContextLength: context,
            maximumKVCacheTokens: context,
            maximumOutputTokens: min(4_096, max(512, context / 4)),
            kvBits: 8,
            prefillStepSize: 256,
            toolParser: .none,
            reasoningParser: supportsThinking ? .qwen3 : .none,
            parallelTools: .sequential,
            supportsThinkingToggle: supportsThinking,
            usesChatTemplateOverride: false,
            requiresYaRNContextExtension: false,
            estimatedKVCacheBytes: Int64(context) * 32_768
        )
    }
}

enum ModelRuntimeContextConfigurator {
    private static let backupName = ".on-device-las-config-original.json"

    /// Qwen3 is native 32K. The official Qwen guidance requires a YaRN factor
    /// of 2 for a real 65,536-token runtime. Apply that override to the local
    /// model configuration before MLX decodes it, while keeping a private
    /// original beside the model for recovery/export transparency.
    static func applyIfNeeded(directory: URL, profile: ModelCapabilityProfile) throws {
        guard profile.requiresYaRNContextExtension else { return }
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else { return }
        let data = try Data(contentsOf: configURL)
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let existingLength = (object["max_position_embeddings"] as? NSNumber)?.intValue ?? 0
        let existingScaling = object["rope_scaling"] as? [String: Any]
        let existingType = (existingScaling?["rope_type"] ?? existingScaling?["type"]) as? String
        let existingFactor = (existingScaling?["factor"] as? NSNumber)?.doubleValue ?? 0
        if existingLength >= profile.configuredContextLength,
           existingType?.lowercased() == "yarn",
           existingFactor >= 2 {
            return
        }

        let backupURL = directory.appendingPathComponent(backupName)
        if !FileManager.default.fileExists(atPath: backupURL.path) {
            try FileManager.default.copyItem(at: configURL, to: backupURL)
        }
        object["max_position_embeddings"] = profile.configuredContextLength
        object["rope_scaling"] = [
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32_768
        ]
        let updated = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        try updated.write(to: configURL, options: .atomic)
    }
}
