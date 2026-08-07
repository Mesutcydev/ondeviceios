import Foundation
import UIKit

/// Core AI implementation of the reference app's assistant façade. Keeping
/// the façade name lets the existing HTTP compatibility layer remain stable,
/// while this target compiles no MLX or llama.cpp source.
@MainActor
final class CodingAssistantService: ObservableObject {
    static let shared = CodingAssistantService()

    enum ServiceState: Equatable {
        case unloaded
        case loading(String)
        case ready
        case generating
        case failed(String)
    }

    /// Live load telemetry shown by the on-device debugger. Mirrors the
    /// reference app's snapshot shape so the shared debugger view works
    /// unchanged. No prompts, generated content, credentials, or user files
    /// are stored here.
    struct LoadDebugSnapshot: Equatable, Sendable {
        enum Phase: String, Equatable, Sendable {
            case idle = "Idle"
            case preparing = "Preparing"
            case downloading = "Downloading"
            case loading = "Loading"
            case ready = "Ready"
            case failed = "Failed"
            case cancelled = "Cancelled"
        }

        var phase: Phase = .idle
        var modelID = ""
        var displayName = ""
        var repoID = ""
        var operation = "Idle"
        var detail = ""
        var lastEvent = ""
        var progress: Double? = nil
        var startedAt: Date? = nil
        var lastProgressAt: Date? = nil
        var lastEventAt = Date()
        var isStalled = false
    }

    @Published private(set) var state: ServiceState = .unloaded
    @Published private(set) var tokenRate: Double = 0
    @Published private(set) var estimatedInputTokens = 0
    @Published private(set) var activeModel = CoreAIAssistantCatalog.defaultModel
    @Published private(set) var isVisionChatCapable = false
    @Published private(set) var loadDebug = LoadDebugSnapshot()

    private var generationTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var loadWatchdogTask: Task<Void, Never>?
    private var lastLoadWatchdogLogAt: Date?
    private var generationStartedAt: Date?
    private var generatedCharacters = 0

    var activeDisplayName: String { activeModel.displayName }
    var activeModelRepoID: String? { activeModel.repoID }
    var hasResidentRuntime: Bool { CoreAIInferenceService.shared.isReady }

    /// Cap used by LocalAPIServer when clamping client `max_tokens`.
    var localAPIEffectiveMaximumOutputTokens: Int {
        max(
            1,
            min(
                AppSettings.shared.assistantMaxTokens,
                CoreAIModelStore.shared.manifest?.maximumOutputTokens ?? 4_096
            )
        )
    }

    private init() {}

    func startLoad() {
        guard loadTask == nil else { return }
        loadTask = Task { @MainActor [weak self] in
            guard let self else { return }
            state = .loading("Preparing \(activeModel.displayName)…")
            beginLoadDebug(for: activeModel)
            do {
                CoreAIModelStore.shared.refresh()
                guard let manifest = CoreAIModelStore.shared.manifest else {
                    throw CoreAIInferenceError.modelMissing
                }
                updateLoadDebug(
                    operation: "Loading Core AI model",
                    detail: "Core AI is specializing \(manifest.displayName) (\(manifest.version)).",
                    event: "Manifest validated"
                )
                try await CoreAIInferenceService.shared.load()
                try Task.checkCancellation()
                isVisionChatCapable = manifest.capabilities.imageInput
                activeModel = AssistantModel(
                    id: manifest.id,
                    repoID: "coreai/\(manifest.modelFamily)",
                    displayName: manifest.displayName,
                    subtitle: "Core AI · \(manifest.version)",
                    approxRAMBytes: max(1, manifest.totalDownloadBytes),
                    tags: ["chat", "core-ai"],
                    contextWindowTokens: manifest.contextWindow,
                    downloadSizeBytes: manifest.totalDownloadBytes,
                    capabilities: Self.capabilities(from: manifest.capabilities),
                    supportsTools: manifest.capabilities.toolCalling,
                    runtime: .coreAI
                )
                state = .ready
                finishLoadDebug(
                    phase: .ready,
                    operation: "Model ready",
                    detail: "Core AI model is resident and serving requests."
                )
                RuntimeLogCenter.emit(
                    "Core AI model ready · \(activeModel.displayName)",
                    subsystem: "coreai"
                )
            } catch is CancellationError {
                state = .unloaded
                finishLoadDebug(
                    phase: .cancelled,
                    operation: "Load cancelled",
                    detail: "The model load was stopped."
                )
            } catch {
                state = .failed(error.localizedDescription)
                finishLoadDebug(
                    phase: .failed,
                    operation: "Load failed",
                    detail: error.localizedDescription,
                    error: "Core AI load failed · \(error.localizedDescription)"
                )
                RuntimeLogCenter.emit(
                    "Core AI load failed · \(error.localizedDescription)",
                    level: .error,
                    subsystem: "coreai"
                )
            }
            loadTask = nil
        }
    }

    func cancelLoad() {
        loadTask?.cancel()
        loadTask = nil
        if case .loading = state { state = .unloaded }
        if loadDebug.phase == .preparing || loadDebug.phase == .loading || loadDebug.phase == .downloading {
            finishLoadDebug(
                phase: .cancelled,
                operation: "Load cancelled",
                detail: "The model load was stopped."
            )
        }
    }

    /// Synchronous unload entry used by the debugger's Stop button and by
    /// lifecycle transitions. Cancels any in-flight load and generation, then
    /// tears the runtime down.
    func unload() {
        stopGeneration()
        cancelLoad()
        Task { @MainActor [weak self] in
            await CoreAIInferenceService.shared.unload()
            guard let self else { return }
            isVisionChatCapable = false
            if state == .ready || state == .generating {
                state = .unloaded
                Diagnostics.shared.breadcrumb("Core AI model unloaded", category: "coreai")
            }
        }
    }

    func startSwitchTo(_ model: AssistantModel) {
        activeModel = model
        state = .unloaded
        startLoad()
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        CoreAIInferenceService.shared.cancel()
        if state == .generating { state = .ready }
    }

    func waitForGenerationToFinish(timeout: Duration = .seconds(5)) async -> Bool {
        let deadline = ContinuousClock.now.advanced(by: timeout)
        while generationTask != nil {
            if ContinuousClock.now >= deadline { return false }
            do {
                try await Task.sleep(for: .milliseconds(50))
            } catch {
                return false
            }
        }
        return true
    }

    func unloadAndWaitForCleanup() async {
        stopGeneration()
        cancelLoad()
        await CoreAIInferenceService.shared.unload()
        isVisionChatCapable = false
        state = .unloaded
    }

    // MARK: - Load telemetry

    private func beginLoadDebug(for model: AssistantModel) {
        loadWatchdogTask?.cancel()
        lastLoadWatchdogLogAt = nil

        let now = Date()
        loadDebug = LoadDebugSnapshot(
            phase: .preparing,
            modelID: model.id,
            displayName: model.displayName,
            repoID: model.repoID,
            operation: "Preparing Core AI model",
            detail: "Waiting for the Core AI runtime.",
            lastEvent: "Load started",
            progress: nil,
            startedAt: now,
            lastProgressAt: now,
            lastEventAt: now,
            isStalled: false
        )
        RuntimeLogCenter.emit(
            "Load started · \(model.displayName) · repo=\(model.repoID)",
            subsystem: "coreai"
        )
        Diagnostics.shared.breadcrumb(
            "Core AI load started · \(model.id) · footprint=\(MemoryAdvisor.physFootprint.formattedBytes) · available=\(MemoryAdvisor.availableMemoryForModel.formattedBytes)",
            category: "coreai"
        )

        // The Core AI SDK load is opaque — no progress callbacks. The
        // watchdog flags callback-silence so a hung specialization shows up
        // in the debugger instead of looking like a frozen screen, and logs
        // memory/thermal context that explains a jetsam kill after the fact.
        loadWatchdogTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled else { return }

                let now = Date()
                let current = self.loadDebug
                guard current.startedAt != nil,
                      current.phase == .preparing || current.phase == .loading else {
                    return
                }

                let elapsed = now.timeIntervalSince(current.startedAt ?? now)
                let sinceProgress = now.timeIntervalSince(
                    current.lastProgressAt ?? current.startedAt ?? now
                )
                var next = current
                next.phase = .loading
                next.isStalled = sinceProgress >= 12
                next.detail = next.isStalled
                    ? "Core AI has been silent for \(Int(sinceProgress))s while specializing the model. Check memory and thermal readings before stopping."
                    : "Core AI is specializing the model. Native callbacks are not exposed during this phase."
                self.loadDebug = next

                if next.isStalled {
                    let shouldLog = self.lastLoadWatchdogLogAt.map {
                        now.timeIntervalSince($0) >= 5
                    } ?? true
                    if shouldLog {
                        self.lastLoadWatchdogLogAt = now
                        let message = "Core AI load silent · \(current.displayName) · elapsed=\(Int(elapsed))s · gap=\(Int(sinceProgress))s · footprint=\(MemoryAdvisor.physFootprint.formattedBytes) · available=\(MemoryAdvisor.availableMemoryForModel.formattedBytes) · thermal=\(SystemSnapshot.thermalState())"
                        RuntimeLogCenter.emit(message, level: .warning, subsystem: "coreai")
                        Diagnostics.shared.warning(message, category: "coreai")
                    }
                }
            }
        }
    }

    private func updateLoadDebug(
        phase: LoadDebugSnapshot.Phase? = nil,
        operation: String? = nil,
        progress: Double? = nil,
        detail: String? = nil,
        event: String? = nil
    ) {
        let now = Date()
        var next = loadDebug
        if let phase { next.phase = phase }
        if let operation { next.operation = operation }
        if let progress {
            let clamped = min(1, max(0, progress))
            if next.progress == nil || clamped >= (next.progress ?? 0) {
                next.progress = clamped
            }
            next.lastProgressAt = now
            next.isStalled = false
        }
        if let detail { next.detail = detail }
        if let event {
            next.lastEvent = event
            next.lastEventAt = now
        }
        loadDebug = next
    }

    private func finishLoadDebug(
        phase: LoadDebugSnapshot.Phase,
        operation: String,
        detail: String,
        error: String? = nil
    ) {
        loadWatchdogTask?.cancel()
        loadWatchdogTask = nil

        var next = loadDebug
        next.phase = phase
        next.operation = operation
        next.detail = detail
        next.lastEvent = error ?? operation
        next.lastEventAt = Date()
        next.lastProgressAt = Date()
        next.isStalled = false
        if phase == .ready { next.progress = 1 }
        loadDebug = next

        let level: RuntimeLogCenter.Level = phase == .failed ? .error : .info
        let message = error ?? "\(next.displayName) · \(operation) · \(detail)"
        RuntimeLogCenter.emit(message, level: level, subsystem: "coreai")
        Diagnostics.shared.breadcrumb(
            "Core AI load \(phase.rawValue.lowercased()) · \(message)",
            category: "coreai"
        )
    }

    /// Signature matches `LocalAPIServer` so the shared HTTP compatibility
    /// layer can stay identical across OnDeviceLAS and CoreAIOnDeviceLAS.
    func generate(
        messages: [ChatMessage],
        maxTokensOverride: Int? = nil,
        temperatureOverride: Double? = nil,
        topPOverride: Double? = nil,
        jsonMode: Bool = false,
        toolConstraint: MLXToolCallConstraintConfiguration? = nil,
        nativeTools: [AssistantNativeToolDefinition] = [],
        nativeToolFormat: AssistantNativeToolFormat? = nil,
        forceNoThinking: Bool = false,
        onToken: @escaping @Sendable (String) -> Void,
        onGenerationResult: (@Sendable (AssistantGenerationResult) -> Void)? = nil,
        onComplete: @escaping @Sendable (Double) -> Void
    ) {
        _ = topPOverride
        _ = jsonMode
        _ = toolConstraint
        _ = nativeTools
        _ = nativeToolFormat
        _ = forceNoThinking

        guard generationTask == nil else {
            onComplete(0)
            return
        }
        guard CoreAIInferenceService.shared.isReady else {
            onComplete(0)
            return
        }

        let maxTokens = min(
            maxTokensOverride ?? AppSettings.shared.assistantMaxTokens,
            localAPIEffectiveMaximumOutputTokens
        )
        let temperature = temperatureOverride ?? AppSettings.shared.assistantTemperature
        let prompt = messages.map { message in
            "\(message.role.rawValue): \(message.contentForModel)"
        }.joined(separator: "\n")
        estimatedInputTokens = max(1, prompt.count / 4)
        generatedCharacters = 0
        generationStartedAt = Date()
        state = .generating

        generationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let started = Date()
            var stopReason: AssistantGenerationResult.StopReason = .stop
            do {
                try await CoreAIInferenceService.shared.generate(
                    messages: messages,
                    maxTokens: maxTokens,
                    temperature: temperature,
                    onToken: { [weak self] token in
                        Task { @MainActor in
                            guard let self else { return }
                            self.generatedCharacters += token.count
                            if let startedAt = self.generationStartedAt {
                                let seconds = max(0.001, Date().timeIntervalSince(startedAt))
                                self.tokenRate = Double(self.generatedCharacters) / 4.0 / seconds
                            }
                            onToken(token)
                        }
                    }
                )
                let completionTokens = max(1, generatedCharacters / 4)
                if completionTokens >= maxTokens {
                    stopReason = .length
                }
                onGenerationResult?(
                    AssistantGenerationResult(
                        promptTokenCount: estimatedInputTokens,
                        completionTokenCount: completionTokens,
                        stopReason: stopReason
                    )
                )
                state = .ready
                onComplete(Date().timeIntervalSince(started))
            } catch is CancellationError {
                stopReason = .cancelled
                onGenerationResult?(
                    AssistantGenerationResult(
                        promptTokenCount: estimatedInputTokens,
                        completionTokenCount: max(0, generatedCharacters / 4),
                        stopReason: stopReason
                    )
                )
                state = .ready
                onComplete(0)
            } catch {
                state = .failed(error.localizedDescription)
                onComplete(0)
            }
            generationTask = nil
        }
    }

    private static func capabilities(
        from modelCapabilities: CoreAIModelCapabilities
    ) -> Set<ModelCapability> {
        var caps: Set<ModelCapability> = [.recommended]
        if modelCapabilities.imageInput { caps.insert(.vision) }
        if modelCapabilities.reasoning { caps.insert(.thinking) }
        if modelCapabilities.toolCalling { caps.insert(.tools) }
        return caps
    }
}
