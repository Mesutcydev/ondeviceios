import Foundation

enum CoreAIInferenceError: LocalizedError, Equatable {
    case unavailable(String)
    case modelMissing
    case suspended
    case unsupportedCapability(String)
    case invalidImage(String)
    case generationFailed

    var errorDescription: String? {
        switch self {
        case .unavailable(let message): return message
        case .modelMissing: return "Download or import a Core AI .aimodel before starting the server."
        case .suspended: return "Core AI inference is suspended until the app returns to the foreground."
        case .unsupportedCapability(let capability): return "The selected Core AI model does not support \(capability)."
        case .invalidImage(let message): return message
        case .generationFailed: return "Core AI could not complete the local generation."
        }
    }
}

/// Small app-facing facade. The Core AI SDK is isolated behind this type so
/// the existing API server and safety/lifecycle code do not import SDK types.
@MainActor
final class CoreAIInferenceService {
    static let shared = CoreAIInferenceService()

    private(set) var isReady = false
    private(set) var isSuspended = false
    private var runtime: CoreAIRuntime?

    private init() {}

    func load() async throws {
        guard #available(iOS 27.0, *) else {
            throw CoreAIInferenceError.unavailable("Core AI requires iOS 27 or later.")
        }
        guard let url = CoreAIModelStore.shared.modelResourcesURL,
              let manifest = CoreAIModelStore.shared.manifest else {
            throw CoreAIInferenceError.modelMissing
        }
        guard manifest.capabilities.textGeneration else {
            throw CoreAIInferenceError.unsupportedCapability("text generation")
        }
        Diagnostics.shared.breadcrumb(
            "Core AI runtime load start · \(manifest.id) · v\(manifest.version) · resources=\(url.lastPathComponent)",
            category: "coreai"
        )
        let runtime = makeRuntime()
        do {
            try await runtime.load(resourcesAt: url)
        } catch {
            Diagnostics.shared.error(
                "Core AI runtime load failed · \(manifest.id) · \(error.localizedDescription)",
                category: "coreai"
            )
            throw error
        }
        self.runtime = runtime
        isReady = true
        isSuspended = false
        Diagnostics.shared.breadcrumb("Core AI runtime ready · \(manifest.id)", category: "coreai")
    }

    func generate(
        messages: [ChatMessage],
        maxTokens: Int,
        temperature: Double,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws {
        guard !isSuspended else { throw CoreAIInferenceError.suspended }
        guard isReady, let runtime else { throw CoreAIInferenceError.modelMissing }
        try await runtime.generate(
            prompt: messages.map { "\($0.role.rawValue): \($0.contentForModel)" }.joined(separator: "\n"),
            maxTokens: maxTokens,
            temperature: temperature,
            onToken: onToken
        )
    }

    func cancel() {
        runtime?.cancel()
    }

    func unload() async {
        runtime?.cancel()
        runtime = nil
        isReady = false
    }

    func suspend() async {
        isSuspended = true
        await unload()
        isSuspended = true
    }
}

private protocol CoreAIRuntime: AnyObject {
    func load(resourcesAt url: URL) async throws
    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws
    func cancel()
}

#if canImport(FoundationModels) && canImport(CoreAILanguageModels)
import FoundationModels
import CoreAILanguageModels

@available(iOS 27.0, *)
private final class CoreAIAvailableRuntime: CoreAIRuntime {
    private var model: CoreAILanguageModel?
    private var task: Task<Void, Never>?

    func load(resourcesAt url: URL) async throws {
        model = try await CoreAILanguageModel(resourcesAt: url)
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws {
        guard let model else { throw CoreAIInferenceError.modelMissing }
        let session = LanguageModelSession(model: model)
        var previous = ""
        for try await snapshot in session.streamResponse(
            to: prompt,
            options: FoundationModels.GenerationOptions(
                temperature: temperature,
                maximumResponseTokens: maxTokens
            )
        ) {
            try Task.checkCancellation()
            let full = snapshot.content
            let delta = String(full.dropFirst(previous.count))
            previous = full
            if !delta.isEmpty { onToken(delta) }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }
}
#endif

private final class CoreAIUnavailableRuntime: CoreAIRuntime {
    func load(resourcesAt url: URL) async throws {
        throw CoreAIInferenceError.unavailable("Build this target with the Xcode 27 SDK to use Core AI.")
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        temperature: Double,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws {
        throw CoreAIInferenceError.unavailable("Core AI is unavailable in this build.")
    }

    func cancel() {}
}

private extension CoreAIInferenceService {
    func makeRuntime() -> CoreAIRuntime {
        #if canImport(FoundationModels) && canImport(CoreAILanguageModels)
        if #available(iOS 27.0, *) { return CoreAIAvailableRuntime() }
        #endif
        return CoreAIUnavailableRuntime()
    }
}
