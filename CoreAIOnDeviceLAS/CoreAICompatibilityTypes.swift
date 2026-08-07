import Foundation
import SwiftUI

public enum ModelExecutionLocation: String, Codable, Hashable, Sendable {
    case localDownloaded
    case localSystem
    case applePrivateCloud
    case externalCloud
}

/// The Core AI target owns these small API-facing types instead of importing
/// the reference app's MLX/llama catalog.  The wire layer needs stable model
/// metadata, but it must not know how an Apple model is implemented.
enum ModelRuntime: String, Codable, Hashable, CaseIterable, Sendable {
    case coreAI
    case mlx
    case llamaCpp
}

enum ModelCapability: String, CaseIterable, Codable, Hashable, Sendable {
    case recommended
    case best
    case newRelease
    case vision
    case thinking
    case tools
    case multilingual
}

struct AssistantModel: Identifiable, Hashable, Codable, Sendable {
    let id: String
    let repoID: String
    let displayName: String
    let subtitle: String
    let approxRAMBytes: Int64
    let tags: [String]
    let contextWindowTokens: Int
    var downloadSizeBytes: Int64?
    var capabilities: Set<ModelCapability>
    var supportsTools: Bool
    var runtime: ModelRuntime

    var supportsVision: Bool { capabilities.contains(.vision) }
    var supportsThinking: Bool { capabilities.contains(.thinking) }

    init(
        id: String,
        repoID: String,
        displayName: String,
        subtitle: String,
        approxRAMBytes: Int64,
        tags: [String] = [],
        contextWindowTokens: Int,
        downloadSizeBytes: Int64? = nil,
        capabilities: Set<ModelCapability> = [],
        supportsTools: Bool = false,
        runtime: ModelRuntime = .coreAI
    ) {
        self.id = id
        self.repoID = repoID
        self.displayName = displayName
        self.subtitle = subtitle
        self.approxRAMBytes = approxRAMBytes
        self.tags = tags
        self.contextWindowTokens = contextWindowTokens
        self.downloadSizeBytes = downloadSizeBytes
        self.capabilities = capabilities
        self.supportsTools = supportsTools
        self.runtime = runtime
    }
}

enum ModelToolParser: String, Codable, CaseIterable, Sendable {
    case none
    case hermes
    case qwen3Coder = "qwen3_coder"
    case qwen3XML = "qwen3_xml"
    case foundationModels

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
    case foundationModels

    var assumesPrefilledOpening: Bool {
        switch self {
        case .none: return false
        case .foundationModels: return true
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
}

enum LocalAPIParallelToolCallsSetting: Int, CaseIterable, Identifiable, Equatable, Sendable {
    case sequential = 1
    case automaticTwo = 2
    case automaticThree = 3
    case automaticFour = 4

    var id: Int { rawValue }
    var maximumCalls: Int { rawValue }
    var allowsParallelCalls: Bool { self != .sequential }
}

struct ModelCapabilityProfile: Equatable, Sendable {
    let modelContextLength: Int
    let configuredContextLength: Int
    let maximumKVCacheTokens: Int
    let maximumOutputTokens: Int
    let toolParser: ModelToolParser
    let reasoningParser: ModelReasoningParser
    let parallelTools: ModelParallelToolPolicy

    var streamAssumesPrefilledReasoningOpen: Bool {
        reasoningParser.assumesPrefilledOpening
    }

    static func resolve(for model: AssistantModel) -> Self {
        let context = max(512, model.contextWindowTokens)
        return Self(
            modelContextLength: context,
            configuredContextLength: context,
            maximumKVCacheTokens: context,
            maximumOutputTokens: 4_096,
            toolParser: model.supportsTools ? .foundationModels : .none,
            reasoningParser: model.supportsThinking ? .foundationModels : .none,
            parallelTools: model.supportsTools
                ? .automatic(maximumCalls: 2)
                : .sequential
        )
    }

    func effectiveParallelLimit(
        requestAllowsParallel: Bool,
        globalEnabled: Bool,
        configuredMaximumCalls: Int? = nil
    ) -> Int {
        guard requestAllowsParallel, globalEnabled else { return 1 }
        return min(
            parallelTools.maximumCalls,
            max(1, configuredMaximumCalls ?? parallelTools.maximumCalls)
        )
    }
}

struct AssistantGenerationResult: Sendable, Equatable {
    enum StopReason: Sendable, Equatable {
        case stop
        case length
        case cancelled
    }

    let promptTokenCount: Int
    let completionTokenCount: Int
    let stopReason: StopReason
}

enum CoreAIAssistantCatalog {
    static let defaultModel = AssistantModel(
        id: CoreAIModelStore.defaultModelID,
        repoID: "coreai/qwen3-0.6b",
        displayName: "Core AI Qwen 0.6B",
        subtitle: "Core AI · model resource directory",
        approxRAMBytes: 1_000_000_000,
        tags: ["chat", "core-ai"],
        contextWindowTokens: 32_768,
        capabilities: [.recommended],
        supportsTools: false,
        runtime: .coreAI
    )
}

/// Compatibility façade retained for the existing HTTP response encoders.
/// The implementation is Core AI-only; these names deliberately contain no
/// runtime package types.
struct AssistantNativeToolDefinition: Equatable, Sendable {
    let name: String
    let description: String?
    let parametersJSON: String
}

enum AssistantNativeToolFormat: Sendable {
    case hermesJSON
    case qwenXML
    case foundationModels
}

struct MLXToolCallConstraintConfiguration: Equatable, Sendable {
    enum Decision: Equatable, Sendable {
        case automatic
        case required
    }

    let toolNames: [String]
    let decision: Decision
    let allowParallelCalls: Bool
    let allowReasoningPrefixes: Bool

    var isSuitableForNativeConstraint: Bool {
        !toolNames.isEmpty
            && toolNames.count <= 128
            && toolNames.allSatisfy { !$0.isEmpty && $0.count <= 128 }
    }
}

struct ConversationContextMemory: Codable, Equatable, Sendable {
    let summary: String
}

struct ConversationContextPreparation: Sendable {
    let messages: [ChatMessage]
    let memory: ConversationContextMemory?
    let didCompact: Bool
}

enum ConversationContextCompactor {
    static let triggerFraction = 0.72

    static func prepare(
        messages: [ChatMessage],
        existingMemory: ConversationContextMemory?,
        maxTokens: Int
    ) -> ConversationContextPreparation {
        // Keep the API boundary bounded without importing the reference
        // app's persistent conversation store into this independent target.
        let budgetCharacters = max(1, maxTokens * 4)
        var usedCharacters = 0
        let boundedMessages = messages.filter { message in
            guard usedCharacters + message.contentForModel.count <= budgetCharacters else { return false }
            usedCharacters += message.contentForModel.count
            return true
        }
        return .init(
            messages: boundedMessages,
            memory: existingMemory,
            didCompact: boundedMessages.count != messages.count
        )
    }
}

/// Settings for the server-only Core AI product. This is intentionally a
/// separate settings surface; it does not carry voice, speech, camera, or
/// legacy model-runtime preferences from the reference app.
@MainActor
final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    @AppStorage("coreAI.localAPIEnabled") var localAPIEnabled = true
    @AppStorage("coreAI.localAPIPort") var localAPIPort = 11_434
    @AppStorage("coreAI.localAPIKeepScreenAwake") var localAPIKeepScreenAwake = true
    @AppStorage("coreAI.localAPIAutoLoadModel") var localAPIAutoLoadModel = false
    @AppStorage("coreAI.localAPIToolCallingEnabled") var localAPIToolCallingEnabled = false
    @AppStorage("coreAI.localAPIReasoningEnabled") var localAPIReasoningEnabled = false
    @AppStorage("coreAI.localAPIParallelToolCallsEnabled") var localAPIParallelToolCallsEnabled = false
    @AppStorage("coreAI.localAPIParallelToolCallsLimit") var localAPIParallelToolCallsLimit = 1
    @AppStorage("coreAI.localAPIStrictToolSchemasEnabled") var localAPIStrictToolSchemasEnabled = true
    @AppStorage("useHFToken") var useHFToken = true
    @AppStorage("coreAI.wifiOnlyDownloads") var wifiOnlyDownloads = false
    @AppStorage("coreAI.assistantMaxTokens") var assistantMaxTokens = 2_048
    @AppStorage("coreAI.assistantTemperature") var assistantTemperature = 0.7
    @AppStorage("coreAI.assistantTopP") var assistantTopP = 0.95
    @AppStorage("coreAI.hasSeenOnboarding") var hasSeenOnboarding = false
    @AppStorage("coreAI.thermalWarningsEnabled") var thermalWarningsEnabled = true
    @AppStorage("coreAI.appearance") var appearance: String = "light"
    @AppStorage("coreAI.themeAccent") var themeAccent: String = KoduTheme.appAccent.rawValue

    var resolvedColorScheme: ColorScheme {
        appearance == "light" ? .light : .dark
    }
}
