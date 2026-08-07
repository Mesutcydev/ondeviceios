import Foundation
import UIKit
import MLX
import MLXLLM
import MLXVLM
import CoreImage
import MLXLMCommon
import MLXLMTransformers  // default TransformersLoader for loadContainer(from:configuration:)
import os                  // OSAllocatedUnfairLock

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

// MARK: - Per-family Assistant execution policy

/// Text generation has a very different memory envelope from image analysis,
/// even when both roles come from one unified model package. Keep the large
/// families useful in Assistant by bounding their context/KV cache instead of
/// forcing every text load through Lens's much larger vision admission gate.
struct MLXAssistantExecutionProfile: Equatable, Sendable {
    let maxContextTokens: Int
    let maxOutputTokens: Int?
    let maxKVSize: Int?
    let kvBits: Int
    let prefillStepSize: Int
    let cacheLimitBytes: Int

    /// Prompt budget for this execution policy.
    ///
    /// A rotating KV cache bounds memory independently of total generation
    /// length. For profiles that intentionally leave output uncapped, do not
    /// subtract the user's entire response allowance from the prompt window;
    /// doing so reduced Qwen 3.5 to the 512-token floor and discarded the
    /// immediately preceding turn on ordinary follow-ups.
    func inputBudget(
        modelContextWindowTokens: Int,
        deviceContextCap: Int,
        requestedOutputTokens: Int
    ) -> Int {
        let contextLimit = min(
            modelContextWindowTokens,
            min(deviceContextCap, maxContextTokens)
        )
        if maxKVSize != nil, maxOutputTokens == nil {
            return max(512, contextLimit - 512)
        }
        let safeOutput = maxOutputTokens.map {
            min(requestedOutputTokens, $0)
        } ?? requestedOutputTokens
        return max(512, contextLimit - safeOutput - 256)
    }

    static func resolve(
        repoID: String,
        architecture: String? = nil,
        catalogContextLength: Int = 32_768,
        supportsThinking: Bool = false
    ) -> Self {
        let capability = ModelCapabilityProfile.resolve(
            repoID: "\(repoID) \(architecture ?? "")",
            catalogContextLength: catalogContextLength,
            supportsThinking: supportsThinking
        )
        return .init(
            maxContextTokens: capability.configuredContextLength,
            maxOutputTokens: capability.maximumOutputTokens,
            maxKVSize: capability.maximumKVCacheTokens,
            kvBits: capability.kvBits,
            prefillStepSize: capability.prefillStepSize,
            cacheLimitBytes: capability.family == .generic
                ? 64 * 1_024 * 1_024
                : 0
        )
    }
}

/// TokenAI-style allocator policy for large MLX models. This is deliberately
/// separate from GGUF paging: MLX still loads every weight, but its allocator
/// is prevented from retaining a large reusable cache or exceeding the
/// entitlement-aware process ceiling.
struct MLXLowMemoryPolicy: Equatable, Sendable {
    let memoryLimitBytes: Int?
    let cacheLimitBytes: Int?

    static func resolve(
        enabled: Bool,
        physicalMemoryBytes: UInt64,
        processCeilingBytes: Int64
    ) -> Self {
        // Standard models use MLX's native allocator policy. Applying a
        // synthetic process limit and
        // cache cap to every load changed the residency characteristics of
        // otherwise identical models and produced idle exits after a
        // successful load. Keep the strict policy as an explicit option.
        guard enabled else {
            return .init(memoryLimitBytes: nil, cacheLimitBytes: nil)
        }
        let conservativePhysicalLimit = Int(Double(physicalMemoryBytes) * 0.74)
        let appCeiling = processCeilingBytes > 0
            ? Int(min(Int64(Int.max), processCeilingBytes))
            : conservativePhysicalLimit
        return .init(
            memoryLimitBytes: max(0, min(conservativePhysicalLimit, appCeiling)),
            cacheLimitBytes: 0
        )
    }
}

/// Chooses between the normal accelerated GGUF path and the opt-in,
/// storage-backed path for models larger than the process memory ceiling.
///
/// llama.cpp memory-maps GGUF weights in both modes. Paging mode additionally
/// keeps every transformer layer off Metal, allowing iOS to reclaim clean
/// file-backed pages instead of retaining a second GPU copy of the weights.
struct GGUFLoadPolicy: Equatable, Sendable {
    let minimumAvailableBytes: Int64
    let gpuLayers: Int32
    let storageBacked: Bool

    /// A freshly-unloaded mmap-backed GGUF can leave a few reclaimable pages
    /// charged to the process briefly. The entitled build may admit that
    /// narrow gap; larger deficits still fail before llama.cpp allocates.
    static let entitledNearFitGraceBytes: Int64 = 128 * 1_024 * 1_024

    /// Runtime, KV cache, sampler, and a bounded window of file-backed pages.
    /// The full GGUF size is deliberately excluded in storage-backed mode:
    /// clean mmap pages are reclaimable and do not need to remain resident.
    static let storageBackedHeadroom: Int64 = 1_500_000_000

    static func resolve(fileBytes: Int64, pagingEnabled: Bool) -> Self {
        if pagingEnabled {
            return .init(
                minimumAvailableBytes: storageBackedHeadroom,
                gpuLayers: 0,
                storageBacked: true
            )
        }

        let minimum = Int64(Double(fileBytes) * 1.15) + 500_000_000
        let threeGiB = Int64(3 * 1_024 * 1_024 * 1_024)
        return .init(
            minimumAvailableBytes: minimum,
            gpuLayers: fileBytes > threeGiB ? 12 : 999,
            storageBacked: false
        )
    }

    func canAdmit(
        availableBytes: Int64,
        hasIncreasedMemoryEntitlement: Bool
    ) -> Bool {
        guard availableBytes > 0 else { return true }
        let grace = hasIncreasedMemoryEntitlement
            ? Self.entitledNearFitGraceBytes
            : 0
        return availableBytes >= minimumAvailableBytes - grace
    }
}

/// Describes which service-owned task handles an unload operation may await.
/// A task must never await its own handle: model switches and direct loads
/// both call into the shared unload path before installing a new runtime.
struct AssistantUnloadDrainPolicy: Equatable, Sendable {
    let loadTask: Bool
    let transitionTask: Bool

    /// Unload initiated by lifecycle/UI code outside a service-owned load.
    static let external = Self(loadTask: true, transitionTask: true)
    /// Cleanup entered by `startLoad` (which can itself run inside a switch).
    static let loadOwned = Self(loadTask: false, transitionTask: false)
    /// Cleanup entered by the transition task before loading its new model.
    static let transitionOwned = Self(loadTask: true, transitionTask: false)
}

enum AssistantGenerationOwnership {
    static func isCurrent(activeID: UUID?, completingID: UUID) -> Bool {
        activeID == completingID
    }
}

/// Model-agnostic tool information passed from the local API boundary to the
/// MLX chat-template processor. This deliberately contains JSON text rather
/// than Foundation `Any`, keeping the value safe across generation actors.
struct AssistantNativeToolDefinition: Equatable, Sendable {
    let name: String
    let description: String?
    let parametersJSON: String
}

enum AssistantNativeToolFormat: Sendable {
    case hermesJSON
    case qwenXML
}

// MARK: - CodingAssistantService
// Runs whatever AssistantModel is selected (default: AssistantModelCatalog
// .presets[0]) on-device via MLX Swift. On first use the model is loaded
// from one of three on-disk locations, in priority order:
//   1. Documents/huggingface/models/<repoID>/   — HubApi lazy cache
//   2. Documents/LLMModels/<dirName>/           — Download Center pre-stage
//   3. Documents/HFModels/<repoID>/             — legacy generic HF pre-stage
// If none exist, MLX falls back to HubApi which downloads into (1). See
// `modelConfig(for:)` for the bridge that picks between directory- and
// repo-id-based ModelConfiguration.

@MainActor
final class CodingAssistantService: ObservableObject {
    static let shared = CodingAssistantService()

    /// Transfers the final strong reference to a loaded MLX container away
    /// from the main actor. Access is serialized by `MLXGenerationGate`; the
    /// unchecked conformance only documents that single-owner hand-off.
    private final class MLXContainerReleaseBox: @unchecked Sendable {
        private var container: ModelContainer?

        init(_ container: ModelContainer?) {
            self.container = container
        }

        func release() {
            container = nil
        }
    }

    /// Like the MLX hand-off above, this keeps llama.cpp's synchronous native
    /// destructor and Metal synchronization away from the main actor while
    /// preserving one final, explicitly-awaited owner.
    private final class GGUFRuntimeReleaseBox: @unchecked Sendable {
        private var runtime: LlamaCppVLM?

        init(_ runtime: LlamaCppVLM?) {
            self.runtime = runtime
        }

        func release() {
            runtime = nil
        }
    }

    /// Recurrent Gemma 4 remains on its proven compact path. Other imported
    /// GGUFs — especially Qwen/Qwopus VLMs with a paired projector — use the
    /// bounded llama.cpp family profile instead of inheriting Gemma's 512-token
    /// emergency limit, which was too small for agent schemas and image turns.
    private static func importedGGUFContextTokens(repoID: String) -> UInt32 {
        let id = repoID.lowercased()
        if id.contains("gemma-4") || id.contains("gemma4") { return 512 }
        return LlamaCppVLMExecutionProfile.resolve(repoID: repoID).contextSize
    }

    private static func importedGGUFMaxOutputTokens(repoID: String) -> Int {
        let id = repoID.lowercased()
        if id.contains("gemma-4") || id.contains("gemma4") { return 128 }
        return LlamaCppVLMExecutionProfile.resolve(repoID: repoID).maxOutputTokens
    }

    private static func importedGGUFInputBudget(repoID: String) -> Int {
        let context = Int(importedGGUFContextTokens(repoID: repoID))
        let output = importedGGUFMaxOutputTokens(repoID: repoID)
        return max(320, context - output - 256)
    }

    @Published private(set) var state: ServiceState = .unloaded
    @Published private(set) var tokenRate: Double = 0
    /// Estimated token count of the trimmed input sent on the last generate().
    /// Used by the UI to render a context-window progress bar.
    @Published private(set) var estimatedInputTokens: Int = 0
    /// Cloud execution is deliberately tracked separately from local model
    /// residency. A PCC request can fail or be cancelled without making a
    /// downloaded MLX / llama.cpp model appear failed.
    @Published private(set) var activeExecutionLocation: ModelExecutionLocation
    @Published private(set) var applePrivateCloudStatus: ApplePCCStatus = .unsupportedOS
    @Published private(set) var applePrivateCloudGenerationState: ApplePCCGenerationState = .idle
    @Published private(set) var applePrivateCloudContextSize: Int?

    /// Practical input budget used by the chat context compactor. This mirrors
    /// the runtime's device/model cap so compaction happens before the final
    /// hard safety trim in `generate()`.
    var effectiveGenerationSettings: AssistantModelGenerationSettings {
        AssistantModelSettingsStore.shared.effectiveSettings(
            for: activeModel.repoID,
            supportsThinking: activeModel.supportsThinking,
            appSettings: AppSettings.shared
        )
    }

    var currentInputBudget: Int {
        if activeExecutionLocation == .applePrivateCloud {
            guard let contextSize = applePrivateCloudContextSize else {
                // No guessed PCC window: retain only the newest prompt until
                // the SDK reports its actual context size.
                return 512
            }
            return ApplePrivateCloud.inputBudget(
                contextSize: contextSize,
                maximumResponseTokens: AppSettings.shared.assistantMaxTokens
            )
        }
        if activeModel.runtime == .llamaCpp {
            return Self.importedGGUFInputBudget(repoID: activeModel.repoID)
        }
        let executionProfile = MLXAssistantExecutionProfile.resolve(
            repoID: activeModel.repoID,
            catalogContextLength: activeModel.contextWindowTokens,
            supportsThinking: activeModel.supportsThinking
        )
        let requestedOutput = min(
            effectiveGenerationSettings.maxTokens,
            DeviceSafetyMonitor.shared.recommendedMaxTokens
        )
        return executionProfile.inputBudget(
            modelContextWindowTokens: executionProfile.maxContextTokens,
            deviceContextCap: executionProfile.maxContextTokens,
            requestedOutputTokens: requestedOutput
        )
    }

    /// Maximum visible completion budget the currently loaded local runtime
    /// can honor right now. The API publishes and clamps against this value so
    /// `/v1/models` never promises more than generation will actually accept.
    var localAPIEffectiveMaximumOutputTokens: Int {
        if activeModel.runtime == .llamaCpp {
            return max(
                1,
                min(
                    Self.importedGGUFMaxOutputTokens(repoID: activeModel.repoID),
                    DeviceSafetyMonitor.shared.recommendedMaxTokens
                )
            )
        }
        let executionProfile = MLXAssistantExecutionProfile.resolve(
            repoID: activeModel.repoID,
            catalogContextLength: activeModel.contextWindowTokens,
            supportsThinking: activeModel.supportsThinking
        )
        return max(
            1,
            min(
                executionProfile.maxOutputTokens ?? Int.max,
                DeviceSafetyMonitor.shared.recommendedMaxTokens
            )
        )
    }

    enum ServiceState: Equatable {
        case unloaded
        case loading(String)
        case ready
        case generating
        case failed(String)
    }

    /// A small, safe snapshot of the active model load. This is intentionally
    /// separate from `ServiceState`: the latter is user-facing, while this
    /// records the native operation and the last progress callback needed to
    /// diagnose a load that appears to stop at "Preparing…".
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

    /// Live load telemetry shown by the on-device debugger. No prompts,
    /// generated content, credentials, or user files are stored here.
    @Published private(set) var loadDebug = LoadDebugSnapshot()

    /// True when the model weights are resident in memory and able to
    /// serve a generate() call — either idle (`.ready`) or busy
    /// (`.generating`). Used by the Mac-pairing UI to decide whether
    /// to show the "load a model first" banner; a generation in
    /// flight is NOT a reason to scare the user into thinking
    /// nothing is loaded. Loading/failed/unloaded all read false.
    var isModelLoaded: Bool {
        if resolvedMLXContainer != nil { return true }
        switch state {
        case .ready, .generating: return true
        case .unloaded, .loading, .failed: return false
        }
    }

    private var container: ModelContainer?

    /// True when the resident MLX runtime is the **dual-role shared vision
    /// container** — the only Assistant path that can route image inputs
    /// through the MLXVLM processor. Bounded-text containers (text-only
    /// MLXLLM load) and GGUF assistants loaded without an mmproj projector
    /// cannot process images, so they stay `false` and any attached image
    /// thumbnails are rendered in the chat bubble only, never sent to the
    /// model. Mirrors how `resolvedMLXContainer`'s fallback only consults
    /// the shared Lens runtime when our own `container` is nil. The chat
    /// UI (photo-picker prefill, send-button gating) and `generate()`'s
    /// chat-array construction both branch on this flag, so a vision-
    /// capable catalogue model that happened to load as a bounded text
    /// runtime doesn't claim image viewing it can't honour.
    @Published private(set) var isVisionChatCapable: Bool = false

    /// Re-evaluate `isVisionChatCapable` against the currently resident
    /// runtime. Called at every load / unload / model-switch state
    /// transition site so the flag tracks what is actually in memory.
    /// Identity-style check: a supportsVision catalogue entry whose MLX
    /// container was loaded locally (text-only) returns `false` because
    /// that container has no vision processor, even though the package is
    /// nominally a VLM.
    @MainActor
    private func refreshVisionChatCapability() {
        if activeModel.runtime == .llamaCpp {
            isVisionChatCapable = ggufModel != nil && ggufVisionProjectorPath != nil
        } else {
            isVisionChatCapable =
                activeModel.supportsVision
                && container == nil
                && LensInferenceLoop.shared.sharedContainer(for: activeModel.repoID) != nil
        }
    }

    /// Text-only runtimes are owned here. A unified text+vision runtime is
    /// borrowed from Lens only when its full vision envelope fits; otherwise
    /// Assistant owns a smaller, bounded text container for the same package.
    private var resolvedMLXContainer: ModelContainer? {
        container ?? LensInferenceLoop.shared.sharedContainer(for: activeModel.repoID)
    }

    private var hasResidentRuntime: Bool {
#if CORE_AI_SERVER_APP
        return CoreAIInferenceService.shared.isReady
#else
        activeModel.runtime == .llamaCpp
            ? ggufModel != nil
            : resolvedMLXContainer != nil
#endif
    }

    private var ggufModel: LlamaCppVLM?
    /// Non-nil only when the resident GGUF was loaded with a validated mtmd
    /// projector. The path is diagnostic state only; it is never logged.
    private var ggufVisionProjectorPath: String?
    private var generateTask: Task<Void, Never>?
    private var pccGenerateTask: Task<Void, Never>?
    private var activePCCRequestID: UUID?
    /// Identity of the MLX load currently allowed to publish a container.
    /// Model loading itself cannot be interrupted safely, so an unload/model
    /// switch invalidates this token and lets the stale load finish without
    /// committing its result over the newly-selected model.
    private var activeLoadID: UUID?
    /// Identifies the GGUF task currently allowed to mutate service state.
    /// A late completion from a cancelled/unloaded task must never mark a
    /// newer generation ready.
    private var activeGGUFGenerationID: UUID?
    /// Identity of the MLX generation currently allowed to publish service
    /// state. A cancelled generation may retire after a newer request starts;
    /// its completion must not clear the newer task handle or mark it ready.
    private var activeMLXGenerationID: UUID?
    private var memoryWarningObserver: NSObjectProtocol?
    private var isTransitioning = false
    /// The server-only UI owns its load task through `startLoad()`, allowing
    /// the Stop action to cancel HubApi/MLX cooperatively instead of merely
    /// hiding the loading state while native work continues in the background.
    private var loadTask: Task<Void, Never>?
    private var loadTaskID: UUID?
    private var transitionTask: Task<Void, Never>?
    /// Single-flight native teardown. `unload()` is a synchronous UI action,
    /// so a following Load tap must await this task rather than allocating a
    /// second GGUF while the first model is still being destroyed.
    private var unloadTask: Task<Void, Never>?
    private var unloadTaskID: UUID?
    private var loadWatchdogTask: Task<Void, Never>?
    private var lastLoadWatchdogLogAt: Date?
    private var lastLoadProgressLogAt: Date?
    private var lastLoadProgressLogFraction: Double = -1
    /// mlx-swift-lm reports download progress up to roughly 80–83%, then
    /// parses safetensors and uploads the weights without invoking the
    /// callback again. Keep that native finalization phase visible instead
    /// of treating a normal callback gap as a deadlock.
    nonisolated private static let nativeWeightFinalizationThreshold = 0.80
    private static let nativeWeightCallbackWarningAfter: TimeInterval = 120
    nonisolated private static let interruptedMLXLoadKey = "OnDeviceLAS.interruptedMLXLoad"
    /// Highest load-progress fraction seen during the active load().
    /// Progress callbacks hop to the MainActor via unordered `Task`s,
    /// so without this monotonic guard percentages can render out of order.
    private var loadProgressFraction: Double = 0

    // MARK: - Init / Deinit

    private init() {
        activeExecutionLocation =
            ApplePrivateCloud.isSupportedOnCurrentOS
            && AppSettings.shared.assistantModelID == ApplePrivateCloud.modelID
                ? .applePrivateCloud
                : .localDownloaded
        if let interruptedLoad = Self.consumeInterruptedMLXLoadMarker() {
            let message = "Previous MLX load ended before cleanup · \(interruptedLoad)"
            RuntimeLogCenter.emit(message, level: .error, subsystem: "model")
            Diagnostics.shared.fault(message, category: "model")
        }
        // Listen for global memory warnings so we can shed the model
        // proactively before iOS Jetsam-kills us.
        memoryWarningObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil, queue: .main
        ) { _ in
            // Queue a GPU cache clear only — do NOT unload the model. Proactive
            // unloading on every memory warning is too aggressive: iOS fires
            // these warnings regularly during inference even when the process
            // has plenty of room left. Let Jetsam handle a true OOM; our job
            // is to free the easiest-to-reclaim GPU buffers and keep running.
            Task { await MLXGenerationGate.shared.clearCacheWhenIdle() }
        }
    }

    /// Persists the last safe checkpoint before entering native MLX code. A
    /// Jetsam/watchdog kill cannot execute Swift cleanup, so this small marker
    /// is the only reliable way to explain the exact phase on next launch.
    /// It deliberately excludes prompts, generated text, paths, and secrets.
    nonisolated private static func markMLXLoadInFlight(
        _ model: AssistantModel,
        sourceDirectory: URL? = nil,
        weightBytes: UInt64 = 0,
        estimatedPeakBytes: Int64? = nil,
        allocatorLimitBytes: Int? = nil,
        cacheLimitBytes: Int? = nil,
        phase: String = "admitted"
    ) {
        let now = Date().timeIntervalSince1970
        let metadata = mlxModelMetadata(in: sourceDirectory)
        let mlx = MLX.Memory.snapshot()
        var marker: [String: String] = [
            "modelID": model.id,
            "repoID": model.repoID,
            "displayName": model.displayName,
            "phase": phase,
            "startedAt": String(now),
            "checkpointAt": String(now),
            "weightBytes": String(weightBytes),
            "estimatedPeakBytes": String(estimatedPeakBytes ?? 0),
            "kernelHeadroom": String(MemoryAdvisor.processAvailableMemory),
            "availableForModel": String(MemoryAdvisor.availableMemoryForModel),
            "processCeiling": String(MemoryAdvisor.processMemoryCeiling),
            "processFootprint": String(MemoryAdvisor.physFootprint),
            "entitled": String(MemoryAdvisor.hasIncreasedMemoryLimitEntitlement),
            "mlxActive": String(mlx.activeMemory),
            "mlxCache": String(mlx.cacheMemory),
            "mlxPeak": String(mlx.peakMemory),
        ]
        if let allocatorLimitBytes {
            marker["allocatorLimit"] = String(allocatorLimitBytes)
        }
        if let cacheLimitBytes {
            marker["cacheLimit"] = String(cacheLimitBytes)
        }
        if !metadata.modelType.isEmpty { marker["modelType"] = metadata.modelType }
        if !metadata.architecture.isEmpty { marker["architecture"] = metadata.architecture }
        if !metadata.quantization.isEmpty { marker["quantization"] = metadata.quantization }
        UserDefaults.standard.set(marker, forKey: interruptedMLXLoadKey)
        UserDefaults.standard.synchronize()
    }

    nonisolated private static func updateMLXLoadCheckpoint(
        phase: String,
        progress: Double? = nil,
        allocatorLimitBytes: Int? = nil,
        cacheLimitBytes: Int? = nil
    ) {
        guard var marker = UserDefaults.standard.dictionary(
            forKey: interruptedMLXLoadKey
        ) as? [String: String] else { return }

        // Native progress can be very chatty. Persist only meaningful 5%
        // advances unless the phase itself changed.
        let previousPhase = marker["phase"]
        if previousPhase == phase,
           !phase.hasSuffix("callback-silent"),
           let progress {
            let old = Double(marker["progress"] ?? "") ?? -1
            if progress < old + 0.05 { return }
        }

        let mlx = MLX.Memory.snapshot()
        marker["phase"] = phase
        marker["checkpointAt"] = String(Date().timeIntervalSince1970)
        marker["processFootprint"] = String(MemoryAdvisor.physFootprint)
        marker["kernelHeadroom"] = String(MemoryAdvisor.processAvailableMemory)
        marker["availableForModel"] = String(MemoryAdvisor.availableMemoryForModel)
        marker["mlxActive"] = String(mlx.activeMemory)
        marker["mlxCache"] = String(mlx.cacheMemory)
        marker["mlxPeak"] = String(mlx.peakMemory)
        if let progress { marker["progress"] = String(progress) }
        if let allocatorLimitBytes {
            marker["allocatorLimit"] = String(allocatorLimitBytes)
        }
        if let cacheLimitBytes { marker["cacheLimit"] = String(cacheLimitBytes) }
        UserDefaults.standard.set(marker, forKey: interruptedMLXLoadKey)
        UserDefaults.standard.synchronize()
    }

    nonisolated private static func mlxModelMetadata(
        in directory: URL?
    ) -> (modelType: String, architecture: String, quantization: String) {
        guard let directory,
              let data = try? Data(
                contentsOf: directory.appendingPathComponent("config.json")
              ),
              let json = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any] else {
            return ("", "", "")
        }
        let modelType = json["model_type"] as? String ?? ""
        let architecture = (json["architectures"] as? [String])?.joined(
            separator: ","
        ) ?? ""
        let quantizationObject = json["quantization"] as? [String: Any]
            ?? json["quantization_config"] as? [String: Any]
        let method = quantizationObject?["method"] as? String
            ?? quantizationObject?["quant_method"] as? String
            ?? ""
        let bits = (quantizationObject?["bits"] as? NSNumber)?.intValue
        let quantization = [
            method,
            bits.map { "\($0)-bit" } ?? ""
        ].filter { !$0.isEmpty }.joined(separator: " ")
        return (modelType, architecture, quantization)
    }

    private static func clearMLXLoadInFlightMarker() {
        UserDefaults.standard.removeObject(forKey: interruptedMLXLoadKey)
        UserDefaults.standard.synchronize()
    }

    private static func consumeInterruptedMLXLoadMarker() -> String? {
        guard let marker = UserDefaults.standard.dictionary(
            forKey: interruptedMLXLoadKey
        ) as? [String: String] else {
            return nil
        }
        clearMLXLoadInFlightMarker()
        let modelID = marker["modelID"] ?? "unknown"
        let repoID = marker["repoID"] ?? "unknown"
        let phase = marker["phase"] ?? "unknown"
        let startedAt = Double(marker["startedAt"] ?? "") ?? 0
        let checkpointAt = Double(marker["checkpointAt"] ?? "") ?? startedAt
        let elapsed = max(0, checkpointAt - startedAt)
        let progress = Double(marker["progress"] ?? "").map {
            "\(Int($0 * 100))%"
        } ?? "none"
        let weights = Int64(marker["weightBytes"] ?? "")?.formattedBytes ?? "unknown"
        let estimatedPeak = Int64(marker["estimatedPeakBytes"] ?? "")?.formattedBytes ?? "unknown"
        let headroom = Int64(marker["kernelHeadroom"] ?? "")?.formattedBytes ?? "unknown"
        let available = Int64(marker["availableForModel"] ?? "")?.formattedBytes ?? "unknown"
        let ceiling = Int64(marker["processCeiling"] ?? "")?.formattedBytes ?? "unknown"
        let footprint = Int64(marker["processFootprint"] ?? "")?.formattedBytes ?? "unknown"
        let allocator = Int64(marker["allocatorLimit"] ?? "")?.formattedBytes ?? "unknown"
        let cacheLimit = Int64(marker["cacheLimit"] ?? "")?.formattedBytes ?? "unknown"
        let mlxActive = Int64(marker["mlxActive"] ?? "")?.formattedBytes ?? "unknown"
        let mlxCache = Int64(marker["mlxCache"] ?? "")?.formattedBytes ?? "unknown"
        let mlxPeak = Int64(marker["mlxPeak"] ?? "")?.formattedBytes ?? "unknown"
        let modelType = marker["modelType"] ?? "unknown"
        let architecture = marker["architecture"] ?? "unknown"
        let quantization = marker["quantization"] ?? "unknown"
        let entitled = marker["entitled"] ?? "unknown"
        return "model=\(modelID) · repo=\(repoID) · phase=\(phase) · elapsed=\(String(format: "%.1fs", elapsed)) · progress=\(progress) · weights=\(weights) · estimatedPeak=\(estimatedPeak) · modelType=\(modelType) · architecture=\(architecture) · quantization=\(quantization) · processFootprint=\(footprint) · available=\(available) · kernelHeadroom=\(headroom) · processCeiling=\(ceiling) · allocatorLimit=\(allocator) · cacheLimit=\(cacheLimit) · mlxActive=\(mlxActive) · mlxCache=\(mlxCache) · mlxPeak=\(mlxPeak) · highMemoryEntitlement=\(entitled)"
    }

    /// Starts a load owned by the service so the on-device Stop action can
    /// cancel the actual async task, not just reset the published state.
    func startLoad(
        allowStorageFallback: Bool = true,
        reselectFromSettings: Bool = true,
        allowUnsafeMemoryLoad: Bool = false
    ) {
#if CORE_AI_SERVER_APP
        guard loadTask == nil, !CoreAIInferenceService.shared.isReady else { return }
        state = .loading("Preparing Core AI model…")
        loadTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await CoreAIInferenceService.shared.load()
                state = .ready
                loadTask = nil
            } catch {
                state = .failed(error.localizedDescription)
                loadTask = nil
            }
        }
        return
#endif
        guard loadTask == nil, transitionTask == nil else { return }
        if case .loading = state { return }

        let taskID = UUID()
        loadTaskID = taskID
        loadTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.load(
                allowStorageFallback: allowStorageFallback,
                reselectFromSettings: reselectFromSettings,
                allowUnsafeMemoryLoad: allowUnsafeMemoryLoad
            )
            if self.loadTaskID == taskID {
                self.loadTaskID = nil
            }
            // Cancellation invalidates the publication ID but deliberately
            // retains this handle while native work drains. Clear the handle
            // when the owning task itself has actually returned.
            self.loadTask = nil
        }
    }

    /// Starts a model switch with the same cancellation ownership as a normal
    /// load. The server target only needs one transition at a time, but keeping
    /// the handle here makes a stuck native load stoppable from the debugger.
    func startSwitchTo(
        _ model: AssistantModel,
        persistAsDefault: Bool = true
    ) {
        guard transitionTask == nil, loadTask == nil else {
            ToastCenter.shared.info(
                "Model still loading",
                detail: "Wait for the current native load to finish before starting another one."
            )
            return
        }
        let taskID = UUID()
        transitionTaskID = taskID
        transitionTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.switchTo(model, persistAsDefault: persistAsDefault)
            if self.transitionTaskID == taskID {
                self.transitionTaskID = nil
            }
            self.transitionTask = nil
        }
    }

    /// Cancels an in-flight load/switch. Native MLX work remains cooperative,
    /// so `activeLoadID` is invalidated immediately and task cancellation
    /// reaches HubApi/MLX at its next suspension point.
    func cancelLoad() {
        let loading = isLoadInFlight
        loadTaskID = nil
        loadTask?.cancel()
        // Keep the handle until the native loader has actually unwound.
        // Dropping it here let the following unload clear MLX/Metal state
        // while `loadContainer` was still running on a large sharded model.
        transitionTaskID = nil
        transitionTask?.cancel()
        // Keep the handle until unload cleanup has awaited the native loader.
        // Dropping this task allowed a cancelled GGUF switch to finish in the
        // background while a replacement load allocated a second runtime.

        guard loading else { return }
        activeLoadID = nil
        finishLoadDebug(
            phase: .cancelled,
            operation: "Load cancelled",
            detail: "Stopped by the user. Native cleanup is still draining."
        )
        state = .unloaded
        RuntimeLogCenter.emit("Model load cancellation requested", subsystem: "model")
    }

    private var transitionTaskID: UUID?

    private var isLoadInFlight: Bool {
        if case .loading = state { return true }
        return false
    }

    private func beginLoadDebug(for model: AssistantModel, operation: String) {
        loadWatchdogTask?.cancel()
        lastLoadWatchdogLogAt = nil
        lastLoadProgressLogAt = nil
        lastLoadProgressLogFraction = -1

        let now = Date()
        loadDebug = LoadDebugSnapshot(
            phase: .preparing,
            modelID: model.id,
            displayName: model.displayName,
            repoID: model.repoID,
            operation: operation,
            detail: "Waiting for the native model loader.",
            lastEvent: "Load started",
            progress: nil,
            startedAt: now,
            lastProgressAt: now,
            lastEventAt: now,
            isStalled: false
        )
        RuntimeLogCenter.emit(
            "Load started · \(model.displayName) · repo=\(model.repoID)",
            subsystem: "model"
        )
        Diagnostics.shared.notice(
            "assistant load started · \(model.id) · repo=\(model.repoID)",
            category: "assistant"
        )

        loadWatchdogTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard !Task.isCancelled else { return }

                let now = Date()
                let current = self.loadDebug
                guard current.startedAt != nil,
                      current.phase == .preparing
                        || current.phase == .downloading
                        || current.phase == .loading else {
                    return
                }

                let elapsed = now.timeIntervalSince(current.startedAt ?? now)
                let sinceProgress = now.timeIntervalSince(
                    current.lastProgressAt ?? current.startedAt ?? now
                )
                var next = current
                let isNativeWeightFinalization = current.operation == "Loading MLX weights"
                    || (current.phase == .loading
                        && (current.progress ?? 0) >= Self.nativeWeightFinalizationThreshold)
                let warningAfter = isNativeWeightFinalization
                    ? Self.nativeWeightCallbackWarningAfter
                    : 12
                next.isStalled = sinceProgress >= warningAfter
                if isNativeWeightFinalization {
                    next.operation = "Loading MLX weights"
                    next.detail = next.isStalled
                        ? "MLX has been callback-silent for \(Int(sinceProgress))s while parsing and uploading weights. Check memory and thermal readings before stopping."
                        : "MLX is parsing safetensors and uploading weights. Native callbacks pause during this phase."
                    if next.isStalled {
                        let shouldLog = self.lastLoadWatchdogLogAt.map {
                            now.timeIntervalSince($0) >= 5
                        } ?? true
                        if shouldLog {
                            self.lastLoadWatchdogLogAt = now
                            Self.updateMLXLoadCheckpoint(
                                phase: "native-finalization-callback-silent",
                                progress: current.progress
                            )
                            let message = "No native weight progress · \(current.displayName) · elapsed=\(Int(elapsed))s · callbackGap=\(Int(sinceProgress))s · footprint=\(MemoryAdvisor.physFootprint.formattedBytes) · available=\(MemoryAdvisor.availableMemoryForModel.formattedBytes)"
                            RuntimeLogCenter.emit(message, level: .warning, subsystem: "model")
                            Diagnostics.shared.warning(message, category: "assistant")
                        }
                    }
                } else if next.isStalled {
                    next.operation = "Waiting for native loader"
                    next.detail = "No progress callback for \(Int(sinceProgress))s · MLX may be parsing or uploading weights."
                    let shouldLog = self.lastLoadWatchdogLogAt.map {
                        now.timeIntervalSince($0) >= 5
                    } ?? true
                    if shouldLog {
                        self.lastLoadWatchdogLogAt = now
                        Self.updateMLXLoadCheckpoint(
                            phase: "loader-callback-silent",
                            progress: current.progress
                        )
                        let message = "No loader progress · \(current.displayName) · elapsed=\(Int(elapsed))s · callbackGap=\(Int(sinceProgress))s · footprint=\(MemoryAdvisor.physFootprint.formattedBytes) · available=\(MemoryAdvisor.availableMemoryForModel.formattedBytes)"
                        RuntimeLogCenter.emit(message, level: .warning, subsystem: "model")
                        Diagnostics.shared.warning(message, category: "assistant")
                    }
                } else {
                    next.detail = "Native loader active · last callback \(Int(sinceProgress))s ago."
                }
                self.loadDebug = next

                let timeout = AppSettings.shared.modelLoadTimeoutSeconds
                guard timeout > 0, elapsed >= Double(timeout) else { continue }

                let message = "Model load timed out after \(timeout)s · \(current.displayName)."
                RuntimeLogCenter.emit(message, level: .error, subsystem: "model")
                Diagnostics.shared.error(message, category: "assistant")
                self.finishLoadDebug(
                    phase: .cancelled,
                    operation: "Load timed out",
                    detail: "The load was stopped after the configured timeout."
                )
                // Use the same drain path as the Stop button. The previous
                // branch discarded the task handle before cleanup, allowing
                // cache/runtime teardown to overlap native MLX work.
                self.cancelLoad()
                ToastCenter.shared.error(
                    "Model load timed out",
                    detail: "\(current.displayName) did not report progress within \(timeout)s. Open the debugger for the last loader phase."
                )
                Task { @MainActor [weak self] in
                    await self?.unloadAndWaitForCleanup()
                }
                return
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

            let shouldLog = clamped >= lastLoadProgressLogFraction + 0.05
                || lastLoadProgressLogAt.map { now.timeIntervalSince($0) >= 5 } ?? true
            if shouldLog {
                lastLoadProgressLogFraction = max(lastLoadProgressLogFraction, clamped)
                lastLoadProgressLogAt = now
                RuntimeLogCenter.emit(
                    "\(loadDebug.displayName) · \(next.operation) · \(Int(clamped * 100))%",
                    subsystem: "model"
                )
            }
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
        RuntimeLogCenter.emit(message, level: level, subsystem: "model")
    }

    deinit {
        if let o = memoryWarningObserver {
            NotificationCenter.default.removeObserver(o)
        }
        generateTask?.cancel()
        pccGenerateTask?.cancel()
        loadTask?.cancel()
        transitionTask?.cancel()
        loadWatchdogTask?.cancel()
        if let activePCCRequestID {
            ApplePrivateCloud.cancel(activePCCRequestID)
        }
    }

    // MARK: - Model config
    // HubApi downloads to Documents/huggingface/models/<repoID>/ on first
    // use (see swift-transformers HubApi.swift — downloadBase defaults to
    // Documents/huggingface, not Caches/). The active model is chosen by
    // the user via AssistantModelCatalog and persisted in
    // AppSettings.assistantModelID.

    /// Currently active model — exposed so the picker UI can show it.
    @Published private(set) var activeModel: AssistantModel = AssistantModelCatalog.currentSelection()

    var activeSelectionID: String {
        activeExecutionLocation == .applePrivateCloud
            ? ApplePrivateCloud.modelID
            : activeModel.id
    }

    var activeDisplayName: String {
        activeExecutionLocation == .applePrivateCloud
            ? ApplePrivateCloud.displayName
            : activeModel.displayName
    }

    var canGenerateSelectedTarget: Bool {
        if activeExecutionLocation == .applePrivateCloud {
            return AppSettings.shared.hasCurrentApplePCCPrivacyConsent
                && applePrivateCloudStatus.canSend
                && applePrivateCloudGenerationState != .generating
        }
        return state == .ready
    }

    var isGeneratingSelectedTarget: Bool {
        if activeExecutionLocation == .applePrivateCloud {
            return applePrivateCloudGenerationState == .generating
        }
        return state == .generating
    }

    var selectedContextWindowTokens: Int {
        if activeExecutionLocation == .applePrivateCloud {
            return applePrivateCloudContextSize ?? 0
        }
        return activeModel.contextWindowTokens
    }

    /// True only when the user had explicitly started a cold model load and
    /// iOS backgrounded the app before it finished. Foregrounding may resume
    /// that load; ordinary unloaded models remain lazy and are not auto-loaded.
    private var resumeLoadAfterLifecycleInterruption = false

    /// The repo ID of the currently loaded model, or nil when nothing is
    /// resident. Used by LifecycleController for crash breadcrumbs.
    var activeModelRepoID: String? {
        switch state {
        case .ready, .generating:
            return activeModel.repoID
        case .unloaded, .loading, .failed:
            return nil
        }
    }

    /// Builds the MLX configuration for whatever the user has selected.
    ///
    /// If the user pre-staged weights via the Download Center, use them
    /// directly via a directory-based ModelConfiguration. Otherwise fall
    /// back to a repo-id config — MLX will lazy-download into HubApi's
    /// cache on first use.
    ///
    /// This is the bridge between two parallel storage layouts:
    ///   • Download Center writes to Documents/LLMModels/<dirName>/
    ///   • HubApi reads from Documents/huggingface/models/<repoID>/
    /// Without this preference check, a Download-Center-staged copy sat
    /// idle on disk while MLX silently re-downloaded its own copy via
    /// HubApi — doubling disk use and confusing users who'd already
    /// "downloaded" the model.
    private static func modelConfig(
        for model: AssistantModel,
        stagedDirectory: URL? = nil
    ) -> ModelConfiguration {
        // mlx-swift-lm 3.x dropped `overrideTokenizer:` (which forced the
        // tokenizer *class*). Class selection is now handled automatically by
        // AutoTokenizer in the TransformersLoader, so it's simply omitted.
        if let staged = stagedDirectory ?? preStagedDirectory(for: model) {
            return ModelConfiguration(
                directory: staged,
                defaultPrompt: "Hello",
                extraEOSTokens: ["<|im_end|>"]
            )
        }
        return ModelConfiguration(
            id: model.repoID,
            defaultPrompt: "Hello",
            extraEOSTokens: ["<|im_end|>"]
        )
    }

    /// Legacy default — used by anything still calling the old API.
    /// Resolves the directory at access time so the first-launch path
    /// can short-circuit HubApi too.
    static var modelConfig: ModelConfiguration {
        modelConfig(for: AssistantModelCatalog.presets[0])
    }

    /// Returns the on-disk directory where pre-staged weights for `model`
    /// live, if any. Walks the layouts the app writes to in priority order:
    /// HubApi cache first (so a previously-used model wins), then the
    /// Download Center destinations. A directory only counts as "staged"
    /// when it passes `ModelCacheProbe.isUsableModelDirectory` (config +
    /// weights both present) — a half-finished download would otherwise
    /// trick MLX into trying to load incomplete state.
    static nonisolated func preStagedDirectory(for model: AssistantModel) -> URL? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dirName = model.repoID.split(separator: "/").last.map(String.init) ?? model.repoID
        // HF Search downloads land at `HFModels/<author>_<name>/` (slash
        // flattened to underscore) — see ModelsManagerView.registerAndDownload
        // and HFSearchRow.init. LocalModelImport drops to `HFModels/<tail>/`
        // where tail is the repoID's last path component. Without both
        // variants, every launch fell through to the HubApi fetch path and
        // re-downloaded the weights even though they were already on disk.
        let flattened = model.repoID.replacingOccurrences(of: "/", with: "_")
        // HuggingFace Hub canonical cache layout: `models--<author>--<name>`
        // with weights inside a `snapshots/<rev-hash>/` subdir. mlx-swift's
        // HubApi follows this convention when it's not pointed at a custom
        // cache root. Probe missed this entirely before — every cold launch
        // of an MLX-loaded model that wasn't ALSO catalog-downloaded would
        // mislabel as "Downloading" until the speed latch flipped.
        let dashedRepo = model.repoID.replacingOccurrences(of: "/", with: "--")
        let hfHubBase = docs.appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--\(dashedRepo)")
        var candidates: [URL] = [
            docs.appendingPathComponent("huggingface")
                .appendingPathComponent("models")
                .appendingPathComponent(model.repoID),
            docs.appendingPathComponent("LLMModels").appendingPathComponent(dirName),
            docs.appendingPathComponent("HFModels").appendingPathComponent(model.repoID),
            docs.appendingPathComponent("HFModels").appendingPathComponent(flattened),
            docs.appendingPathComponent("HFModels").appendingPathComponent(dirName),
            // HF Hub-compatible layout — base dir AND each snapshot subdir.
            // ModelCacheProbe recurses, so the base entry catches it; the
            // explicit snapshot enumeration below is belt-and-suspenders
            // for cases where only one snapshot is complete and another
            // is mid-download.
            hfHubBase,
        ]
        let snapshotsDir = hfHubBase.appendingPathComponent("snapshots")
        if let entries = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path) {
            for snap in entries {
                candidates.append(snapshotsDir.appendingPathComponent(snap))
            }
        }
        // Require tokenizer.json (not just any tokenizer companion) — the MLX
        // text loader throws configurationMissing("tokenizer.json") otherwise.
        // A dir missing only tokenizer.json is rejected here so modelConfig
        // falls back to the repo-id config and HubApi re-fetches the snapshot,
        // skipping the weights already on disk. See
        // LocalModelFileValidator.isUsableMLXTextModelDirectory.
        if let direct = candidates.first(where: LocalModelFileValidator.isUsableMLXTextModelDirectory) {
            return direct
        }

        // A model imported from Files is deliberately stored under a local
        // synthetic repo id (`local/local_<name>`). If the user selects the
        // matching catalog preset afterward, that synthetic id is not an exact
        // match for the Hugging Face repo id even though the files are the
        // correct model. Resolve that alias by its small `.repoID` sidecar and
        // folder name before falling back to HubApi. This prevents a second
        // multi-GB fetch at load time and is what the debugger's `cached=false`
        // trace was exposing for Ornith.
        let roots = [
            docs.appendingPathComponent("LLMModels", isDirectory: true),
            docs.appendingPathComponent("HFModels", isDirectory: true)
        ]
        var aliasCandidates: [URL] = []
        for root in roots {
            guard let entries = try? FileManager.default.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            for entry in entries {
                var isDirectory: ObjCBool = false
                guard FileManager.default.fileExists(
                    atPath: entry.path,
                    isDirectory: &isDirectory
                ), isDirectory.boolValue else { continue }
                if Self.isLocalAliasDirectory(
                    entry,
                    for: model.repoID,
                    dirName: dirName,
                    flattenedRepo: flattened
                ) {
                    aliasCandidates.append(entry)
                }

                // Keep the scan bounded. Imports may contain one wrapper
                // directory, but a recursive walk here would duplicate the
                // multi-GB model validation work on every load attempt.
                guard let children = try? FileManager.default.contentsOfDirectory(
                    at: entry,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                ) else { continue }
                for child in children {
                    var childIsDirectory: ObjCBool = false
                    guard FileManager.default.fileExists(
                        atPath: child.path,
                        isDirectory: &childIsDirectory
                    ), childIsDirectory.boolValue else { continue }
                    if Self.isLocalAliasDirectory(
                        child,
                        for: model.repoID,
                        dirName: dirName,
                        flattenedRepo: flattened
                    ) {
                        aliasCandidates.append(child)
                    }
                }
            }
        }
        return aliasCandidates.first(where: LocalModelFileValidator.isUsableMLXTextModelDirectory)
    }

    private static nonisolated func isLocalAliasDirectory(
        _ directory: URL,
        for repoID: String,
        dirName: String,
        flattenedRepo: String
    ) -> Bool {
        let sidecar = directory.appendingPathComponent(".repoID")
        let storedID = (try? String(contentsOf: sidecar, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let requested = repoID.lowercased()
        if storedID == requested { return true }

        // LocalModelImportService names these folders `local_<source-name>`
        // and records `local/<folder-name>`. Accept the optional collision
        // suffix used by repeated imports, but never match an unrelated
        // generic directory solely because it happens to contain weights.
        guard storedID?.hasPrefix("local/") == true else { return false }
        let folder = directory.lastPathComponent.lowercased()
        let aliases = [
            "local_\(flattenedRepo.lowercased())",
            "local_\(dirName.lowercased())"
        ]
        return aliases.contains { alias in
            folder == alias || folder.hasPrefix(alias + "-")
        }
    }

    /// Returns true when the model's weights are already cached and look
    /// complete (both config.json and at least one weights file). Drives
    /// the "Preparing" vs "Downloading" label in the load progress pill.
    /// Mirrors the logic in `preStagedDirectory` — anything we'd accept
    /// as a load source counts as "cached" here too.
    private static func isModelCachedLocally(_ model: AssistantModel) -> Bool {
        preStagedDirectory(for: model) != nil
    }

    /// First built-in preset whose weights are already fully staged on disk
    /// AND that passes the device safety gate, excluding `excludedID`. Used as
    /// the storage-fallback target: a model that needs no download (so it can't
    /// re-trigger the out-of-space failure) and fits memory. Returns nil when
    /// nothing usable is cached.
    private static func firstReadyOnDiskAssistantModel(excluding excludedID: String) -> AssistantModel? {
        for model in AssistantModelCatalog.presets where model.id != excludedID {
            guard isModelCachedLocally(model) else { continue }
            guard MemoryAdvisor.safetyBlocker(for: model.id) == nil else { continue }
            return model
        }
        return nil
    }

    /// Best-effort weight-size estimate. If weights are pre-staged, sum
    /// the actual safetensors / .npz / .bin files; otherwise fall back
    /// to `AssistantModel.approxRAMBytes × 0.6` (4-bit weights are
    /// roughly half a byte per parameter, so disk ≈ RAM × 0.5–0.6).
    /// Used by ModelResidency to decide whether the assistant LLM and
    /// the visual VLM can both live in memory at once on this device.
    ///
    /// `static nonisolated` so it can be called from outside the actor.
    static nonisolated func estimatedWeightBytes(for model: AssistantModel) -> UInt64 {
        if let dir = preStagedDirectory(for: model) {
            let measured = sumWeightFiles(in: dir)
            if measured > 0 { return measured }
        }
        if let published = model.downloadSizeBytes {
            return UInt64(max(0, published))
        }
        // RAM estimate × 0.6 ≈ 4-bit disk size. Conservative when no
        // weights are on disk yet (predicts the eventual download size
        // for the memory-budget math).
        return UInt64(Double(model.approxRAMBytes) * 0.6)
    }

    /// Sum the weight-file sizes under `dir`. Mirrors the helper in
    /// LensInferenceLoop; kept local here so this file doesn't depend
    /// on the lens module. Returns 0 if no recognizable weight files
    /// are present.
    private static nonisolated func sumWeightFiles(in dir: URL) -> UInt64 {
        guard let enumerator = FileManager.default.enumerator(
            at: dir, includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: UInt64 = 0
        for case let url as URL in enumerator {
            let name = url.lastPathComponent
            let isWeight = name.hasSuffix(".safetensors")
                        || name.hasSuffix(".npz")
                        || name.hasSuffix(".bin")
            guard isWeight else { continue }
            if let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize {
                total += UInt64(size)
            }
        }
        return total
    }

    // MARK: - System prompt

    nonisolated static let systemPrompt = """
    You are an expert coding assistant running entirely on-device in IOSLocalLLM.

    Your capabilities:
    • Code completion, generation, and refactoring across all languages
    • Bug detection, root cause analysis, and fixes with explanations
    • Architecture review: design patterns, SOLID, performance trade-offs
    • Security audit: OWASP Top 10, injection, auth flaws, data exposure
    • Code explanation at any depth
    • Test generation (unit, integration, property-based)
    • Documentation and inline comment writing

    Rules:
    - Always output working, runnable code
    - Prefer clarity over cleverness; name things explicitly
    - Cite the specific line/pattern when reviewing
    - Use markdown with fenced code blocks and language tags
    - Put a blank line between paragraphs and use Markdown bullets or numbered
      lists for multiple items. Never run headings, labels, or list items
      together without whitespace.
    - Be concise — no filler phrases
    """

    nonisolated static let responseFormattingPrompt = """
    Format every answer for a narrow mobile screen. Put a blank line between
    paragraphs. When presenting multiple fields, choices, steps, or facts, use
    a Markdown bullet or numbered list with one item per line. Never concatenate
    a heading, label, sentence, or list item directly into the next one.
    """

    // MARK: - Load

    func load(
        allowStorageFallback: Bool = true,
        reselectFromSettings: Bool = true,
        allowUnsafeMemoryLoad: Bool = false
    ) async {
        if let unloadTask {
            Diagnostics.shared.breadcrumb(
                "assistant load waiting for pending unload",
                category: "assistant"
            )
            await unloadTask.value
        }
        guard !Task.isCancelled else { return }
        if activeExecutionLocation == .applePrivateCloud {
            await refreshApplePrivateCloudStatus()
            return
        }

        // Resolve `model_type: nanbeige` before MLX reads config.json.
        // Registration is actor-isolated and idempotent.
        await NanbeigeMLXRegistration.register()

        // Loading/generation cannot be re-entered. A `.ready` state is only a
        // no-op when its actual runtime still exists; a shared Lens container
        // may have been reclaimed independently on memory pressure.
        switch state {
        case .loading, .generating: return
        default: break
        }

        // Pick up the user's current model selection FIRST so the safety
        // checks reflect what the user actually picked (not just qwen3-4b).
        // Skipped on a storage-fallback re-entry, which has already pointed
        // activeModel at an on-disk model without disturbing the saved choice.
        activeModel = Self.loadTarget(
            activeModel: activeModel,
            savedDefault: AssistantModelCatalog.currentSelection(),
            reselectFromSettings: reselectFromSettings
        )

        let runtimeIsResident = activeModel.runtime == .llamaCpp
            ? ggufModel != nil
            : resolvedMLXContainer != nil
        if case .ready = state, runtimeIsResident { return }
        if case .ready = state { state = .unloaded }

        if let compatibility = activeModel.platformCompatibility,
           !compatibility.supportsCurrentPlatform {
            state = .failed(compatibility.detail)
            RuntimeLogCenter.emit(
                "Model rejected by platform compatibility · \(activeModel.displayName) · \(compatibility.detail)",
                level: .error,
                subsystem: "model"
            )
            ToastCenter.shared.error("Can't load \(activeModel.displayName)",
                                     detail: compatibility.detail)
            return
        }

        // A failed/cancelled generation can leave its model resident while the
        // service state allows `load()` to be entered again. Never begin a
        // retry or a newly-selected runtime on top of those weights: a 4.7 GB
        // GGUF followed by Qwen briefly exceeds the iOS process limit and is
        // terminated by Jetsam. `switchTo` already drains explicitly; this
        // closes the direct retry/load path as well.
        if ggufModel != nil || container != nil || generateTask != nil {
            Diagnostics.shared.breadcrumb(
                "assistant reload draining resident runtime · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel)",
                category: "assistant"
            )
            // This call is made from inside the service's own load task. It
            // must release an old resident runtime without awaiting the
            // current load task itself.
            await unloadAndWaitForCleanup(policy: .loadOwned)
            guard !Task.isCancelled else { return }
        }

        // The Lens runtime is the neutral owner for unified text+vision
        // packages. If it already has this model, attach instantly instead of
        // unloading it and loading identical weights through MLXLLM.
        if let _ = LensInferenceLoop.shared.sharedContainer(for: activeModel.repoID) {
            FastVLMService.shared.unload()
            await LlamaCppVLMService.shared.unloadAndWaitForCleanup()
            state = .ready
            refreshVisionChatCapability()
            Diagnostics.shared.breadcrumb(
                "assistant attached to shared dual-role runtime · \(activeModel.id) · visionChat=\(isVisionChatCapable)",
                category: "assistant"
            )
            return
        }

        // Assistant is a single-runtime surface. Drain every visual backend,
        // including llama.cpp/mtmd (which does not share the MLX gate), before
        // measuring or loading the chat model. This makes rapid Lens →
        // Assistant and Visual → Code-mode changes deterministic.
        MLXVisionService.shared.unload()
        FastVLMService.shared.unload()
        await LlamaCppVLMService.shared.unloadAndWaitForCleanup()
        await MLXGenerationGate.shared.clearCacheWhenIdle()

        // Resolve and measure the exact directory the native loader will use
        // before admission. Imported models live under a synthetic
        // `local/<folder>` repo id; the older generic lookup missed that path
        // and admitted every import as an unknown 3.5 GB model. A 16 GB MLX
        // checkpoint could therefore pass the gate and be Jetsam-killed while
        // materializing weights. Weight bytes × workingSetOverhead is the same
        // single peak estimate used by MemoryAdvisor for known local models.
        let stagedDirectory = Self.preStagedDirectory(for: activeModel)
        let probedWarm = stagedDirectory != nil
        let measuredWeightBytes = stagedDirectory.map {
            Self.sumWeightFiles(in: $0)
        } ?? 0
        let measuredFootprintBytes: Int64? = measuredWeightBytes > 0
            ? Int64(Double(measuredWeightBytes) * MemoryAdvisor.workingSetOverhead)
            : nil

        // Combined device-safety gate — covers RAM, live free memory, thermal
        // state, and low-power mode. Refuses outright when unsafe.
        // GGUF uses mmap and can fall back to CPU layers when Metal headroom
        // is tight. The generic gate estimates every imported directory like
        // an eagerly-materialized MLX model (disk × 1.6), which rejected valid
        // large GGUFs before llama.cpp was even attempted.
        if activeModel.runtime != .llamaCpp,
           let block = MemoryAdvisor.safetyBlocker(
                for: activeModel.id,
                allowTightFit: AppSettings.shared.largeModelLowMemoryEnabled,
                runtime: activeModel.runtime,
                allowUnsafeMemoryLoad: allowUnsafeMemoryLoad,
                measuredFootprintBytes: measuredFootprintBytes
            ) {
            state = .failed(block)
            RuntimeLogCenter.emit(
                "Memory/safety gate blocked \(activeModel.displayName) · weights=\(Int64(measuredWeightBytes).formattedBytes) · estimatedPeak=\(measuredFootprintBytes?.formattedBytes ?? "unknown") · \(block)",
                level: .error,
                subsystem: "model"
            )
            ToastCenter.shared.error("Can't load selected model",
                                      detail: block)
            return
        }
        if allowUnsafeMemoryLoad {
            Diagnostics.shared.breadcrumb(
                "user-confirmed unsafe memory load · \(activeModel.id)",
                category: "assistant"
            )
            ToastCenter.shared.info(
                "Experimental load started",
                detail: "iOS may close the app if this model exceeds its memory limit."
            )
        }

        // Open with "Preparing" by default — the verb users want for a
        // warm cache. The on-disk probe (preStagedDirectory) tries hard
        // but can miss HubApi cache variants. If the load turns out to
        // be a real network download, the latch below flips to
        // "Downloading" once we observe genuine network-slow progress
        // (≥3 s elapsed AND <20% complete). This way a cached model
        // never shows "Downloading" by accident — worst case is a
        // brief "Preparing" before "Downloading" on a true cold fetch,
        // which reads correctly because preparation IS the first phase
        // of any network download.
        // Pre-flight DISK check for the download path. When nothing usable is
        // staged, the id-based load fetches the weights via HubApi. On a
        // near-full device that write fails late with a raw ENOSPC AFTER
        // writing a multi-GB partial snapshot that then squats on disk across
        // retries — making "no space" worse each attempt. Refuse early with the
        // honest free-vs-needed message instead, writing nothing.
        //
        // CRITICAL: credit bytes ALREADY on disk. HubApi resumes — it skips
        // files already downloaded — so the real requirement is the REMAINING
        // bytes, not the full model. Without this, a model that's 90% fetched
        // (e.g. interrupted by a previous ENOSPC, or a partial the shard-probe
        // rejected) is wrongly refused for space it doesn't actually need.
        if !probedWarm {
            let estimatedTotal = Int64(Self.estimatedWeightBytes(for: activeModel))
            let alreadyOnDisk = Self.hubCacheBytesOnDisk(for: activeModel)
            let needBytes = max(0, estimatedTotal - alreadyOnDisk) + 300_000_000
            if let free = HFModelDownloadManager.freeDiskBytes(), free < needBytes {
                let msg = Self.outOfStorageMessage(for: activeModel, remainingBytes: needBytes)
                state = .failed(msg)
                RuntimeLogCenter.emit(msg, level: .error, subsystem: "model")
                ToastCenter.shared.error("\(activeModel.displayName) needs more storage", detail: msg)
                return
            }
        }

        state = .loading("Preparing \(activeModel.displayName)…")
        loadProgressFraction = 0
        let loadStart = Date()
        beginLoadDebug(for: activeModel, operation: "Starting model load")

        // Record the selected MLX model before entering the package loader.
        // MLX promotes low-level validation failures to fatalError/SIGTRAP,
        // so a post-load breadcrumb is never reached in exactly the failure
        // mode we most need to diagnose (for example an unsupported weight
        // quantization kernel).
        if activeModel.runtime == .mlx {
            Diagnostics.shared.breadcrumb(
                "MLX assistant load · \(activeModel.id) · repo=\(activeModel.repoID) · cached=\(probedWarm)",
                category: "assistant"
            )
        }

        if activeModel.runtime == .llamaCpp {
            updateLoadDebug(
                operation: "Validating imported GGUF",
                detail: "Checking the model file and selected memory policy."
            )
            guard let directory = Self.preStagedDirectory(for: activeModel),
                  let modelURL = LocalModelFileValidator.ggufLLM(in: directory) else {
                let message = "The imported GGUF file could not be found on disk."
                state = .failed(message)
                finishLoadDebug(phase: .failed, operation: "GGUF validation failed", detail: message, error: message)
                return
            }
            if let reason = DeviceSafetyMonitor.shared.stopReason {
                state = .failed(reason.detail)
                finishLoadDebug(phase: .failed, operation: "Safety gate blocked load", detail: reason.detail, error: reason.detail)
                return
            }
            do {
                try Task.checkCancellation()
                // Gemma 4 keeps its proven 512-token recurrent profile; Qwen /
                // Qwopus and other imported GGUFs receive the bounded 4K VLM
                // profile so image prompts and tool schemas are not truncated.
                let context = Self.importedGGUFContextTokens(repoID: activeModel.repoID)
                let projectorURL = LocalModelFileValidator.ggufProjector(in: directory)
                let modelBytes = ((try? FileManager.default.attributesOfItem(atPath: modelURL.path))?[.size] as? NSNumber)?.int64Value ?? 0
                let projectorBytes = projectorURL.flatMap {
                    ((try? FileManager.default.attributesOfItem(atPath: $0.path))?[.size] as? NSNumber)?.int64Value
                } ?? 0
                // The projector is a real resident component, not optional
                // metadata. Include it in admission or a 922 MB F32 mmproj can
                // turn an apparently safe Q3 model into a jetsam load.
                let fileBytes = modelBytes + projectorBytes
                let available = MemoryAdvisor.availableMemoryForModel
                let loadPolicy = GGUFLoadPolicy.resolve(
                    fileBytes: fileBytes,
                    pagingEnabled: AppSettings.shared.largeModelLowMemoryEnabled
                )
                let minimumNeeded = loadPolicy.minimumAvailableBytes
                guard fileBytes == 0 || loadPolicy.canAdmit(
                    availableBytes: available,
                    hasIncreasedMemoryEntitlement: MemoryAdvisor.hasIncreasedMemoryLimitEntitlement
                ) else {
                    throw RuntimeError.modelLoadBlocked(reason: String(
                        format: "This GGUF needs about %.1f GB available in the selected loading mode; the app currently has %.1f GB. Close other apps, enable storage-backed paging, or use a smaller quantization.",
                        Double(minimumNeeded) / 1_000_000_000,
                        Double(available) / 1_000_000_000
                    ))
                }
                let gpuLayers = loadPolicy.gpuLayers
                Diagnostics.shared.breadcrumb(
                    "GGUF assistant load · bytes=\(fileBytes) · multimodal=\(projectorURL != nil) · context=\(context) · available=\(available) · gpuLayers=\(gpuLayers) · storageBacked=\(loadPolicy.storageBacked)",
                    category: "assistant"
                )
                var loadedRuntime: LlamaCppVLM? = try await Task.detached(priority: .userInitiated) {
                    try LlamaCppVLM(
                        llmPath: modelURL.path,
                        mmprojPath: projectorURL?.path,
                        nThreads: 4,
                        contextSize: context,
                        gpuLayers: gpuLayers
                    )
                }.value
                if Task.isCancelled {
                    Diagnostics.shared.breadcrumb(
                        "GGUF load completed after cancellation; releasing stale runtime",
                        category: "assistant"
                    )
                    let staleRuntime = GGUFRuntimeReleaseBox(loadedRuntime)
                    loadedRuntime = nil
                    await Task.detached(priority: .utility) {
                        staleRuntime.release()
                        autoreleasepool { }
                    }.value
                    throw CancellationError()
                }
                guard let loaded = loadedRuntime else {
                    throw LlamaCppError.contextInitFailed
                }
                ggufModel = loaded
                loadedRuntime = nil
                ggufVisionProjectorPath = projectorURL?.path
                Diagnostics.shared.breadcrumb(
                    "GGUF assistant loaded · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel) · gpuLayers=\(gpuLayers)",
                    category: "assistant"
                )
                state = .ready
                refreshVisionChatCapability()
                let elapsedMs = Date().timeIntervalSince(loadStart) * 1_000
                finishLoadDebug(
                    phase: .ready,
                    operation: "GGUF runtime resident",
                    detail: "Loaded in \(String(format: "%.1fs", elapsedMs / 1_000))."
                )
                ModelUsageTracker.shared.recordLoadTime(modelID: activeModel.id, ms: elapsedMs)
                // Readiness is persistent state; the Assistant's inline
                // status row reflects `.ready` without covering the chat.
            } catch {
                ggufModel = nil
                ggufVisionProjectorPath = nil
                if Self.isCancellationError(error) || Task.isCancelled {
                    state = .unloaded
                    finishLoadDebug(
                        phase: .cancelled,
                        operation: "GGUF load cancelled",
                        detail: "Native loading finished and its temporary runtime was released."
                    )
                    return
                }
                state = .failed(error.localizedDescription)
                finishLoadDebug(
                    phase: .failed,
                    operation: "GGUF load failed",
                    detail: error.localizedDescription,
                    error: error.localizedDescription
                )
                Diagnostics.shared.error(
                    "GGUF assistant load failed · \(error.localizedDescription)",
                    category: "assistant"
                )
                ToastCenter.shared.error("GGUF model failed to load", detail: error.localizedDescription)
            }
            return
        }

        // Latch tracks whether THIS specific load looks like a real
        // network transfer (still slow after 3 s). Starts false (=
        // "Preparing"). Flips once on slow-progress evidence and stays
        // there for the rest of the load so the label doesn't bounce.
        final class VerbLatch: @unchecked Sendable {
            private let lock = NSLock()
            private var _isNetwork: Bool
            init(_ v: Bool) { _isNetwork = v }
            var isNetwork: Bool { lock.withLock { _isNetwork } }
            func markNetwork() { lock.withLock { _isNetwork = true } }
        }
        let latch = VerbLatch(false)

        let loadingModel = activeModel
        let loadID = UUID()
        activeLoadID = loadID

        let isDualRole = DualRoleModelPolicy.isTextAndVision(repoID: loadingModel.repoID)
        let requiredVisionBytes = isDualRole
            ? LensInferenceLoop.requiredVisionLoadBytes(repoID: loadingModel.repoID)
            : 0
        let availableBytes = UInt64(max(0, MemoryAdvisor.availableMemoryForModel))
        let shouldShareVisionRuntime = DualRoleModelPolicy.shouldShareRuntime(
            isDualRole: isDualRole,
            selectionsMatch: isDualRole
                && DualRoleModelPolicy.selectionsMatch(repoID: loadingModel.repoID),
            requiredVisionBytes: requiredVisionBytes,
            availableBytes: availableBytes
        )

        if shouldShareVisionRuntime {
            Self.markMLXLoadInFlight(
                loadingModel,
                sourceDirectory: stagedDirectory,
                weightBytes: measuredWeightBytes,
                estimatedPeakBytes: measuredFootprintBytes,
                phase: "shared-vision-runtime"
            )
            defer { Self.clearMLXLoadInFlightMarker() }
            updateLoadDebug(
                operation: "Preparing shared text + vision runtime",
                detail: "The selected model is being loaded through the Lens runtime."
            )
            await LensInferenceLoop.shared.switchTo(
                repoID: loadingModel.repoID,
                preserveMatchingAssistant: true
            )
            guard activeLoadID == loadID, activeModel.id == loadingModel.id else { return }
            activeLoadID = nil
            if LensInferenceLoop.shared.sharedContainer(for: loadingModel.repoID) != nil {
                state = .ready
                refreshVisionChatCapability()
                let elapsedMs = Date().timeIntervalSince(loadStart) * 1_000
                finishLoadDebug(
                    phase: .ready,
                    operation: "Shared runtime resident",
                    detail: "Loaded in \(String(format: "%.1fs", elapsedMs / 1_000))."
                )
                Diagnostics.shared.breadcrumb(
                    "assistant loaded shared dual-role runtime · \(loadingModel.id) · \(String(format: "%.1fs", elapsedMs / 1_000)) · visionChat=\(isVisionChatCapable)",
                    category: "assistant"
                )
                ModelUsageTracker.shared.recordLoadTime(modelID: loadingModel.id, ms: elapsedMs)
                // The Assistant status row owns model-ready feedback.
            } else if case .failed(let message) = LensInferenceLoop.shared.state {
                state = .failed(message)
                finishLoadDebug(phase: .failed, operation: "Shared runtime failed", detail: message, error: message)
            } else {
                let message = "The shared text + vision runtime could not be prepared."
                state = .failed(message)
                finishLoadDebug(phase: .failed, operation: "Shared runtime failed", detail: message, error: message)
            }
            return
        } else if isDualRole {
            Diagnostics.shared.notice(
                "assistant using bounded text runtime · \(loadingModel.id) · vision required=\(Int64(requiredVisionBytes).formattedBytes) · available=\(Int64(availableBytes).formattedBytes)",
                category: "assistant"
            )
        }

        do {
            try Task.checkCancellation()
            let capabilityProfile = ModelCapabilityProfile.resolve(for: loadingModel)
            if let stagedDirectory {
                try ModelRuntimeContextConfigurator.applyIfNeeded(
                    directory: stagedDirectory,
                    profile: capabilityProfile
                )
            }
            // Snapshot self-derived values up here so the gate's @Sendable
            // closure doesn't have to capture self for them.
            let modelConfig = Self.modelConfig(
                for: loadingModel,
                stagedDirectory: stagedDirectory
            )
            let lowMemoryPolicy = MLXLowMemoryPolicy.resolve(
                enabled: AppSettings.shared.largeModelLowMemoryEnabled,
                physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
                processCeilingBytes: MemoryAdvisor.processMemoryCeiling
            )
            let sourceName = stagedDirectory?.lastPathComponent
                ?? loadingModel.repoID
            updateLoadDebug(
                operation: "Waiting for MLX runtime gate",
                detail: "source=\(sourceName) · memory cap=\(lowMemoryPolicy.memoryLimitBytes.map { Int64($0).formattedBytes } ?? "system")"
            )
            RuntimeLogCenter.emit(
                "Waiting for MLX runtime gate · \(loadingModel.displayName) · kernelHeadroom=\(MemoryAdvisor.processAvailableMemory.formattedBytes) · processCeiling=\(MemoryAdvisor.processMemoryCeiling.formattedBytes) · highMemoryEntitlement=\(MemoryAdvisor.hasIncreasedMemoryLimitEntitlement)",
                subsystem: "model"
            )
            Self.markMLXLoadInFlight(
                loadingModel,
                sourceDirectory: stagedDirectory,
                weightBytes: measuredWeightBytes,
                estimatedPeakBytes: measuredFootprintBytes,
                allocatorLimitBytes: lowMemoryPolicy.memoryLimitBytes,
                cacheLimitBytes: lowMemoryPolicy.cacheLimitBytes,
                phase: "waiting-for-mlx-gate"
            )
            defer { Self.clearMLXLoadInFlightMarker() }
            // Gated so weight uploads can't race a concurrent MLXVision /
            // FastVLM load or another service's in-flight generate.
            let loadedContainer = try await MLXGenerationGate.shared.run { [weak self, latch, loadID] in
                let previousMemoryLimit = MLX.Memory.memoryLimit
                let previousCacheLimit = MLX.Memory.cacheLimit
                if let limit = lowMemoryPolicy.memoryLimitBytes {
                    MLX.Memory.memoryLimit = limit
                }
                if let cache = lowMemoryPolicy.cacheLimitBytes {
                    MLX.Memory.cacheLimit = cache
                }
                Self.updateMLXLoadCheckpoint(
                    phase: "native-loader-entered",
                    allocatorLimitBytes: lowMemoryPolicy.memoryLimitBytes,
                    cacheLimitBytes: lowMemoryPolicy.cacheLimitBytes
                )
                defer {
                    MLX.Memory.memoryLimit = previousMemoryLimit
                    MLX.Memory.cacheLimit = previousCacheLimit
                }
                mlxClearCache()
                // Bind the weak self into a `let` here so the inner Task
                // captures an immutable binding, not the outer closure's
                // implicitly-mutable `[weak self]` slot. Without this
                // shim Swift 6 flags "captured var 'self' in
                // concurrently-executing code" at the Task closure.
                let weakSelf = self
                RuntimeLogCenter.emit(
                    "MLX loader entered · \(loadingModel.displayName) · operation=loadContainer",
                    subsystem: "model"
                )
                Task { @MainActor [weakSelf, loadID] in
                    guard let weakSelf,
                          weakSelf.activeLoadID == loadID else { return }
                    weakSelf.updateLoadDebug(
                        phase: .loading,
                        operation: "MLXModelFactory.loadContainer",
                        detail: "Native MLX loader is active; waiting for progress or completion."
                    )
                }
                if let stagedDirectory {
                    // A completed download/import is already a local MLX
                    // directory. Calling the URL-based Hub downloader here
                    // adds an unnecessary cache/probe layer. The direct
                    // overload keeps the final handoff in one operation.
                    RuntimeLogCenter.emit(
                        "Loading staged MLX directory · \(stagedDirectory.lastPathComponent)",
                        subsystem: "model"
                    )
                    Self.updateMLXLoadCheckpoint(
                        phase: "loading-local-weights",
                        allocatorLimitBytes: lowMemoryPolicy.memoryLimitBytes,
                        cacheLimitBytes: lowMemoryPolicy.cacheLimitBytes
                    )
                    Task { @MainActor [weakSelf, loadID] in
                        guard let weakSelf,
                              weakSelf.activeLoadID == loadID else { return }
                        weakSelf.updateLoadDebug(
                            phase: .loading,
                            operation: "Loading local MLX weights",
                            detail: "Reading the completed model directory directly."
                        )
                    }
                    return try await LLMModelFactory.shared.loadContainer(
                        from: stagedDirectory
                    )
                }
                return try await LLMModelFactory.shared.loadContainer(
                    from: HubApiDownloader(),
                    configuration: modelConfig
                ) { progress in
                    let fraction = progress.fractionCompleted
                    Self.updateMLXLoadCheckpoint(
                        phase: fraction >= Self.nativeWeightFinalizationThreshold
                            ? "native-weight-finalization"
                            : "loader-progress",
                        progress: fraction,
                        allocatorLimitBytes: lowMemoryPolicy.memoryLimitBytes,
                        cacheLimitBytes: lowMemoryPolicy.cacheLimitBytes
                    )
                    let pct = Int(fraction * 100)
                    let elapsed = Date().timeIntervalSince(loadStart)
                    // Network-slow detection: ≥3 seconds elapsed AND
                    // less than 20% complete means this is a real
                    // download (cached loads finish their progress
                    // sweep in well under 3 s on every device class).
                    // Probe hint plus this rule together cover both
                    // the easy case (probe correctly says "not cached")
                    // and the hard case (probe missed an MLX hub
                    // cache variant — we still detect from speed).
                    if !latch.isNetwork,
                       !probedWarm,
                       elapsed > 3.0,
                       progress.fractionCompleted < 0.20 {
                        latch.markNetwork()
                    }
                    let isNetwork = latch.isNetwork
                    Task { @MainActor [weakSelf, isNetwork, pct, fraction, loadID] in
                        // Hops are unordered — drop stale fractions so the
                        // percentage can't render backwards. An unload/model
                        // switch also invalidates this load ID, preventing an
                        // old download from overwriting the new model's state.
                        guard let weakSelf,
                              weakSelf.activeLoadID == loadID,
                              fraction >= weakSelf.loadProgressFraction else { return }
                        weakSelf.loadProgressFraction = fraction
                        // progress.localizedDescription is unreliable across MLX
                        // versions — sometimes "Loading", sometimes empty,
                        // sometimes already contains a %. Pick the verb here
                        // based on the latch (snapshot above into a `let`).
                        let isNativeFinalization = fraction >= CodingAssistantService.nativeWeightFinalizationThreshold
                        if isNativeFinalization {
                            let enteringFinalization = weakSelf.loadDebug.operation != "Loading MLX weights"
                            weakSelf.updateLoadDebug(
                                phase: .loading,
                                operation: "Loading MLX weights",
                                progress: fraction,
                                detail: "MLX is parsing safetensors and uploading weights. Native callbacks pause during this phase.",
                                event: enteringFinalization ? "Native weight finalization started" : nil
                            )
                            weakSelf.state = .loading("Loading MLX weights…")
                        } else {
                            let verb = isNetwork ? "Downloading" : "Preparing"
                            weakSelf.updateLoadDebug(
                                phase: isNetwork ? .downloading : .preparing,
                                operation: isNetwork ? "Downloading model files" : "Preparing MLX container",
                                progress: fraction,
                                detail: "Native loader callback at \(pct)%.",
                                event: "Progress \(pct)%"
                            )
                            weakSelf.state = .loading("\(verb) \(pct)%")
                        }
                    }
                }
            }
            guard activeLoadID == loadID, activeModel.id == loadingModel.id else {
                Diagnostics.shared.breadcrumb(
                    "discarded stale assistant load · \(loadingModel.id)",
                    category: "assistant"
                )
                await MLXGenerationGate.shared.clearCacheWhenIdle()
                return
            }
            activeLoadID = nil
            container = loadedContainer
            state = .ready
            refreshVisionChatCapability()
            let elapsedMs = Date().timeIntervalSince(loadStart) * 1000
            finishLoadDebug(
                phase: .ready,
                operation: "MLX runtime resident",
                detail: "Loaded in \(String(format: "%.1fs", elapsedMs / 1000))."
            )
            Diagnostics.shared.breadcrumb("assistant loaded · \(loadingModel.id) · \(String(format: "%.1fs", elapsedMs / 1000)) · visionChat=\(isVisionChatCapable)", category: "assistant")
            ModelUsageTracker.shared.recordLoadTime(modelID: loadingModel.id, ms: elapsedMs)
            // The Assistant status row owns model-ready feedback.
            // The LLM is now resident and idle. Schedule a background
            // prefetch of the VLM's weight files into the OS page cache
            // after 4 seconds of continued idleness. If the user starts
            // generating before the timer fires, the prefetch is
            // cancelled by `cancelPrefetch()` in `generate()`. Doesn't
            // load the VLM into MLX (would OOM); just primes the
            // kernel's page cache so the eventual MLX load is faster.
            ModelResidency.shared.schedulePrefetch(currentTab: .assistant)
        } catch is MLXGenerationGate.Cancelled {
            guard activeLoadID == loadID else { return }
            activeLoadID = nil
            // Gate drained mid-load — back to unloaded, not a failure.
            finishLoadDebug(
                phase: .cancelled,
                operation: "MLX load cancelled",
                detail: "The runtime gate cancelled this load before it could finish."
            )
            state = .unloaded
        } catch {
            guard activeLoadID == loadID else { return }
            activeLoadID = nil
            // Background cleanup cancels Hub/MLX work cooperatively. That is a
            // lifecycle interruption, not a broken model. Build 75 surfaced
            // the raw "cancelled" error as "Qwen failed to load" after the user
            // briefly left the app, even though memory and the model were fine.
            if Self.isCancellationError(error) {
                finishLoadDebug(
                    phase: .cancelled,
                    operation: "MLX load interrupted",
                    detail: "The load task was cancelled while the native loader was active.",
                    error: error.localizedDescription
                )
                state = .unloaded
                Diagnostics.shared.breadcrumb(
                    "assistant load interrupted · \(activeModel.id) · willResume=\(resumeLoadAfterLifecycleInterruption)",
                    category: "assistant"
                )
                return
            }
            // Translate a raw out-of-storage POSIX error (ENOSPC) into an
            // honest, actionable message. The default localizedDescription is
            // just "The operation couldn't be completed. No space left on
            // device" — which reads as a bug to users who believe they have
            // space, because iOS's "available" figure counts purgeable caches
            // the download path can't actually use. Report the *real* free
            // bytes so the user can tell whether the disk is genuinely full.
            let outOfStorage = Self.isOutOfStorageError(error)
            if outOfStorage {
                // Reclaim the partial snapshot HubApi wrote before running out
                // of space, so it doesn't squat on disk and make the next retry
                // fail even sooner. Guarded to delete only an INCOMPLETE copy of
                // THIS model — a usable cache is never touched.
                Self.removeIncompleteHubSnapshot(for: activeModel)
            }
            let message = outOfStorage
                ? Self.outOfStorageMessage(for: activeModel)
                : error.localizedDescription
            finishLoadDebug(
                phase: .failed,
                operation: outOfStorage ? "MLX load ran out of storage" : "MLX load failed",
                detail: message,
                error: message
            )
            Diagnostics.shared.error("assistant load failed · \(activeModel.id) · \(error.localizedDescription)", category: "assistant")

            // Storage-fallback safety net: when the selected model can't load
            // because the device is out of space, don't strand the user on a
            // dead "Failed to load" screen if another model is already fully on
            // disk. Switch to it for THIS session only — the saved selection is
            // left untouched, so once space is freed a retry still loads the
            // model the user actually picked. allowStorageFallback is false on
            // the re-entry so a second failure can't loop.
            if outOfStorage, allowStorageFallback,
               let fallback = Self.firstReadyOnDiskAssistantModel(excluding: activeModel.id) {
                let failedName = activeModel.displayName
                Diagnostics.shared.breadcrumb(
                    "assistant storage fallback · \(activeModel.id) → \(fallback.id)",
                    category: "assistant")
                activeModel = fallback
                state = .unloaded   // clear .loading so the re-entrant load runs
                ToastCenter.shared.info(
                    "Loaded \(fallback.displayName) instead",
                    detail: "\(failedName) needs more storage. Switched to a model already on your device — free up space, then reselect it in the picker.")
                await load(allowStorageFallback: false, reselectFromSettings: false)
                return
            }

            state = .failed(message)
            ToastCenter.shared.error("\(activeModel.displayName) failed to load",
                                      detail: message)
        }
    }

    /// Removes an INCOMPLETE HubApi snapshot for `model` left behind by a
    /// failed download (e.g. ENOSPC mid-fetch). Checks both default snapshot
    /// layouts and deletes a directory only when it exists AND fails the
    /// usable-model probe, so a complete cache is never destroyed.
    private static func removeIncompleteHubSnapshot(for model: AssistantModel) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dashed = model.repoID.replacingOccurrences(of: "/", with: "--")
        let candidates = [
            docs.appendingPathComponent("huggingface/models/\(model.repoID)", isDirectory: true),
            docs.appendingPathComponent("huggingface/hub/models--\(dashed)", isDirectory: true),
        ]
        for dir in candidates {
            guard FileManager.default.fileExists(atPath: dir.path),
                  !ModelCacheProbe.isUsableModelDirectory(dir) else { continue }
            try? FileManager.default.removeItem(at: dir)
            Diagnostics.shared.breadcrumb(
                "removed incomplete snapshot \(dir.lastPathComponent) after out-of-space",
                category: "assistant")
        }
        MemoryAdvisor.invalidateFootprintCache()
    }

    /// True when `error` (or any error it wraps) is a filesystem
    /// out-of-space condition (POSIX `ENOSPC`). MLX/HubApi surfaces this as
    /// an NSCocoaError wrapping an NSPOSIXError, so we walk the underlying
    /// chain rather than string-matching the localized text.
    private static func isOutOfStorageError(_ error: Error) -> Bool {
        var ns: NSError? = error as NSError
        var depth = 0
        while let current = ns, depth < 6 {
            if current.domain == NSPOSIXErrorDomain && current.code == Int(ENOSPC) {
                return true
            }
            if current.domain == NSCocoaErrorDomain &&
               (current.code == NSFileWriteOutOfSpaceError || current.code == NSFileWriteVolumeReadOnlyError) {
                return true
            }
            ns = current.userInfo[NSUnderlyingErrorKey] as? NSError
            depth += 1
        }
        // Last-resort substring check for paths that flatten the error chain.
        return error.localizedDescription.localizedCaseInsensitiveContains("No space left on device")
    }

    private static func isCancellationError(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled { return true }
        if ns.domain == NSCocoaErrorDomain && ns.code == NSUserCancelledError { return true }
        return ns.localizedDescription
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased() == "cancelled"
    }

    /// Records whether background cleanup interrupted an explicit model load.
    /// Called before lifecycle cancellation changes the published state.
    func noteLifecycleInterruption() {
        if case .loading = state {
            resumeLoadAfterLifecycleInterruption = true
        }
    }

    /// Resumes only a load that was already underway when the app backgrounded.
    /// This preserves the normal lazy-load policy for every other model.
    func resumeInterruptedLoadIfNeeded() async {
        guard resumeLoadAfterLifecycleInterruption else { return }
        resumeLoadAfterLifecycleInterruption = false
        guard case .loading = state else {
            state = .unloaded
            await load()
            return
        }
        // Cleanup may still be retiring the cancelled loader. The generation
        // gate serializes this retry behind it so two model loads never overlap.
        await MLXGenerationGate.shared.clearCacheWhenIdle()
        state = .unloaded
        // Resume the model whose load was interrupted. It may be a
        // conversation-specific choice that intentionally differs from the
        // saved default.
        await load(reselectFromSettings: false)
    }

    /// Keeps the distinction between an explicit runtime target and the
    /// user's saved default testable without loading model weights.
    nonisolated static func loadTarget(
        activeModel: AssistantModel,
        savedDefault: AssistantModel,
        reselectFromSettings: Bool
    ) -> AssistantModel {
        reselectFromSettings ? savedDefault : activeModel
    }

    /// Honest, actionable out-of-storage message reporting the device's real
    /// free space alongside the model's approximate footprint.
    /// Honest, actionable out-of-storage message. `remainingBytes`, when given,
    /// is the still-to-download amount (full size minus what's already cached)
    /// so the figure matches what the user must actually free; otherwise it
    /// falls back to the model's full estimated on-disk size.
    private static func outOfStorageMessage(for model: AssistantModel, remainingBytes: Int64? = nil) -> String {
        let freeStr = HFModelDownloadManager.freeDiskBytes()?.formattedBytes ?? "an unknown amount"
        let needStr = (remainingBytes ?? Int64(Double(model.approxRAMBytes) * 0.6)).formattedBytes
        return "Not enough storage to load \(model.displayName). About \(freeStr) is free, but it needs ~\(needStr) more on disk. Free up space in Settings → General → iPhone Storage, or delete other downloaded models in the Models tab, then retry."
    }

    /// Bytes of `model`'s weights already present in the HubApi cache — the
    /// location an id-based load downloads/resumes into. HubApi skips files
    /// already on disk, so the disk pre-flight credits these against the total.
    private static func hubCacheBytesOnDisk(for model: AssistantModel) -> Int64 {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dashed = model.repoID.replacingOccurrences(of: "/", with: "--")
        let dirs = [
            docs.appendingPathComponent("huggingface/models/\(model.repoID)", isDirectory: true),
            docs.appendingPathComponent("huggingface/hub/models--\(dashed)", isDirectory: true),
        ]
        var total: Int64 = 0
        for dir in dirs where FileManager.default.fileExists(atPath: dir.path) {
            if let sz = try? FileManager.default.allocatedSizeOfDirectory(at: dir) {
                total += sz
            }
        }
        return total
    }

    // MARK: - Generate (streaming, full conversation history)

    nonisolated private static func nativeToolSpecs(
        from definitions: [AssistantNativeToolDefinition]
    ) -> [ToolSpec] {
        definitions.compactMap { definition in
            guard let data = definition.parametersJSON.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data),
                  let parameters = object as? [String: Any] else {
                return nil
            }
            var function: [String: any Sendable] = [
                "name": definition.name,
                "parameters": sendableDictionary(parameters)
            ]
            if let description = definition.description, !description.isEmpty {
                function["description"] = description
            }
            return [
                "type": "function",
                "function": function
            ]
        }
    }

    nonisolated private static func nativeToolCalls(
        from metadata: [ChatMessage.ToolCallMetadata]
    ) -> [MLXLMCommon.ToolCall] {
        metadata.compactMap { call in
            guard let data = call.argumentsJSON.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data),
                  let arguments = object as? [String: Any] else {
                return nil
            }
            return MLXLMCommon.ToolCall(
                function: .init(
                    name: call.name,
                    arguments: arguments.mapValues { MLXLMCommon.JSONValue.from($0) }
                ),
                id: call.id
            )
        }
    }

    nonisolated private static func sendableDictionary(
        _ dictionary: [String: Any]
    ) -> [String: any Sendable] {
        dictionary.mapValues(sendableJSONValue)
    }

    nonisolated private static func sendableJSONValue(_ value: Any) -> any Sendable {
        if value is NSNull { return NSNull() }
        if let value = value as? Bool { return value }
        if let value = value as? Int { return value }
        if let value = value as? Double { return value }
        if let value = value as? String { return value }
        if let value = value as? [String: Any] {
            return sendableDictionary(value)
        }
        if let value = value as? [Any] {
            return value.map(sendableJSONValue)
        }
        if let value = value as? NSNumber {
            let double = value.doubleValue
            return double.rounded() == double ? value.intValue : double
        }
        return String(describing: value)
    }

    /// Convert MLX's parsed call back to a stable neutral wire format. The
    /// local API parser then emits the exact dialect required by OpenAI,
    /// Anthropic, Ollama, Hermes, or OpenCode without exposing model syntax.
    nonisolated private static func encodedNativeToolCall(
        _ call: MLXLMCommon.ToolCall
    ) -> String {
        var payload: [String: Any] = [
            "name": call.function.name,
            "arguments": call.function.arguments.mapValues { $0.anyValue }
        ]
        if let id = call.id, !id.isEmpty {
            payload["id"] = id
        }
        guard JSONSerialization.isValidJSONObject(payload),
              let data = try? JSONSerialization.data(
                withJSONObject: payload,
                options: [.sortedKeys]
              ),
              let json = String(data: data, encoding: .utf8) else {
            return "<tool_call>{\"name\":\"\(call.function.name)\",\"arguments\":{}}</tool_call>"
        }
        return "<tool_call>\(json)</tool_call>"
    }

    /// The `enable_thinking` value `generate` will hand the chat template.
    /// When this is false the template does not pre-fill `<think>`, so the
    /// model emits a plain answer and a streaming reader must not start
    /// inside a reasoning block.
    func resolvedThinkingEnabled(forceNoThinking: Bool) -> Bool {
        let wantsThinking = AssistantModelSettingsStore.shared
            .settings(for: activeModel.repoID)?.thinkingEnabled
            ?? AppSettings.shared.assistantThinking
        return activeModel.supportsThinking && wantsThinking && !forceNoThinking
    }

    /// `maxTokensOverride` bypasses `AppSettings.assistantMaxTokens` for
    /// lightweight background calls (e.g. conversation titling) so they don't
    /// clobber the user's preferred response length.
    func generate(
        messages: [ChatMessage],
        maxTokensOverride: Int? = nil,
        temperatureOverride: Double? = nil,
        topPOverride: Double? = nil,
        samplerConfig: SamplerConfig? = nil,
        jsonMode: Bool = false,
        toolConstraint: MLXToolCallConstraintConfiguration? = nil,
        nativeTools: [AssistantNativeToolDefinition] = [],
        nativeToolFormat: AssistantNativeToolFormat? = nil,
        collectLogprobs: Bool = false,
        forceNoThinking: Bool = false,
        onToken: @escaping @Sendable (String) -> Void,
        // Optional logprob stream sits before onComplete so call sites that
        // pass it (the chat view) match declaration order; callers that omit
        // it (benchmark, compare, quality eval, titler) skip the default.
        onLogprobToken: (@Sendable (TokenLogprob) -> Void)? = nil,
        onGenerationResult: (@Sendable (AssistantGenerationResult) -> Void)? = nil,
        onComplete: @escaping @Sendable (Double) -> Void
    ) {
#if CORE_AI_SERVER_APP
        guard CoreAIInferenceService.shared.isReady else {
            onComplete(0)
            return
        }
        state = .generating
        let started = Date()
        generateTask?.cancel()
        generateTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                try await CoreAIInferenceService.shared.generate(
                    messages: messages,
                    maxTokens: maxTokensOverride ?? AppSettings.shared.assistantMaxTokens,
                    temperature: temperatureOverride ?? AppSettings.shared.assistantTemperature,
                    onToken: onToken
                )
                state = .ready
                onComplete(Date().timeIntervalSince(started))
            } catch is CancellationError {
                state = .ready
                onComplete(0)
            } catch {
                state = .failed(error.localizedDescription)
                onComplete(0)
            }
            generateTask = nil
        }
        return
#endif
        if activeExecutionLocation == .applePrivateCloud {
            generateWithApplePrivateCloud(
                messages: messages,
                maxTokensOverride: maxTokensOverride,
                temperatureOverride: temperatureOverride,
                jsonMode: jsonMode,
                forceNoThinking: forceNoThinking,
                onToken: onToken,
                onComplete: onComplete
            )
            return
        }

        // A native GGUF decode is not re-entrant. Voice Mode turn 2 used to
        // land here while turn 1 was still draining and silently no-op —
        // the mic looked live but the model never answered again. Defer
        // briefly onto the main actor so a just-finished decode can flip
        // back to `.ready`, then retry once.
        if case .generating = state {
            Task { @MainActor [weak self] in
                guard let self else { onComplete(0); return }
                var waited = 0
                while case .generating = self.state, waited < 40 {
                    try? await Task.sleep(nanoseconds: 50_000_000)
                    waited += 1
                }
                guard case .ready = self.state else {
                    onComplete(0)
                    return
                }
                self.generate(
                    messages: messages,
                    maxTokensOverride: maxTokensOverride,
                    temperatureOverride: temperatureOverride,
                    topPOverride: topPOverride,
                samplerConfig: samplerConfig,
                jsonMode: jsonMode,
                toolConstraint: toolConstraint,
                nativeTools: nativeTools,
                nativeToolFormat: nativeToolFormat,
                collectLogprobs: collectLogprobs,
                forceNoThinking: forceNoThinking,
                    onToken: onToken,
                    onLogprobToken: onLogprobToken,
                    onGenerationResult: onGenerationResult,
                    onComplete: onComplete
                )
            }
            return
        }

        // Lazy load. The view layer no longer auto-loads on appear (that
        // path made every launch show a "Downloading X%" banner even when
        // the user wasn't planning to chat). The first send is the load
        // trigger now: we kick off the load, then re-enter generate once
        // the container is ready. onComplete still fires on failure so
        // the UI placeholder unfreezes either way.
        let hasRuntimeModel = activeModel.runtime == .llamaCpp
            ? ggufModel != nil
            : resolvedMLXContainer != nil
        if state != .ready || !hasRuntimeModel {
            Task { [weak self] in
                guard let self else { onComplete(0); return }
                if case .loading = state {
                    // Another caller is already loading — wait it out, with a
                    // deadline so a wedged load (network stall, HubApi hang)
                    // can't spin this poll forever and freeze the send.
                    let deadline = Date().addingTimeInterval(60)
                    while case .loading = self.state {
                        if Date() >= deadline { onComplete(0); return }
                        try? await Task.sleep(nanoseconds: 100_000_000)
                    }
                } else if state != .ready {
                    await self.load()
                }
                let isLoaded = self.activeModel.runtime == .llamaCpp
                    ? self.ggufModel != nil
                    : self.resolvedMLXContainer != nil
                if case .ready = self.state, isLoaded {
                    self.generate(
                        messages: messages,
                        maxTokensOverride: maxTokensOverride,
                        temperatureOverride: temperatureOverride,
                        topPOverride: topPOverride,
                        samplerConfig: samplerConfig,
                        jsonMode: jsonMode,
                        toolConstraint: toolConstraint,
                        nativeTools: nativeTools,
                        nativeToolFormat: nativeToolFormat,
                        collectLogprobs: collectLogprobs,
                        forceNoThinking: forceNoThinking,
                        onToken: onToken,
                        onLogprobToken: onLogprobToken,
                        onGenerationResult: onGenerationResult,
                        onComplete: onComplete
                    )
                } else {
                    onComplete(0)
                }
            }
            return
        }
        guard case .ready = state else { onComplete(0); return }

        // Device-safety gate — refuse to start a fresh generation only when
        // the device is genuinely at .critical thermal state or under
        // sustained memory pressure. Use the reason-specific message so a
        // memory warning isn't mislabelled as "too hot" (and vice versa).
        let safety = DeviceSafetyMonitor.shared
        if let reason = safety.stopReason {
            ToastCenter.shared.error(reason.title, detail: reason.detail)
            onComplete(0)
            return
        }

        generateTask?.cancel()
        state = .generating
        // User started inference — cancel any pending background
        // prefetch so it doesn't compete for disk bandwidth with
        // the active generate.
        ModelResidency.shared.cancelPrefetch()

        let s = AppSettings.shared
        // Clamp max output tokens by the user's setting and the thermal advisor.
        let executionProfile = MLXAssistantExecutionProfile.resolve(
            repoID: activeModel.repoID,
            catalogContextLength: activeModel.contextWindowTokens,
            supportsThinking: activeModel.supportsThinking
        )
        let lowMemoryPolicy = MLXLowMemoryPolicy.resolve(
            enabled: s.largeModelLowMemoryEnabled,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            processCeilingBytes: MemoryAdvisor.processMemoryCeiling
        )
        let modelSettings = AssistantModelSettingsStore.shared.settings(
            for: activeModel.repoID
        )
        let requestedMaxTokens = min(
            maxTokensOverride
                ?? modelSettings?.maxTokens
                ?? s.assistantMaxTokens,
            safety.recommendedMaxTokens
        )
        let safeMaxTokens = executionProfile.maxOutputTokens.map {
            min(requestedMaxTokens, $0)
        } ?? requestedMaxTokens

        // --- Full Sampler Control (Feature #1) ---
        // Resolution order: per-call samplerConfig > per-call overrides >
        // the active model's saved profile > preset from AppSettings >
        // AppSettings individual knobs > defaults.
        let sc = samplerConfig ?? SamplerConfig.default

        // If a preset is active and no per-call config was given, use preset values.
        let effectivePreset: SamplerPreset? = {
            if samplerConfig != nil { return nil }
            guard !s.samplerPresetID.isEmpty,
                  let p = SamplerPreset.preset(for: s.samplerPresetID)
            else { return nil }
            return p
        }()

        let temp = min(2.0, max(0.0,
            sc.temperature
            ?? temperatureOverride
            ?? modelSettings?.temperature
            ?? effectivePreset?.temperature
            ?? s.assistantTemperature
        ))
        let topP = min(1.0, max(0.0,
            sc.topP
            ?? topPOverride
            ?? modelSettings?.topP
            ?? effectivePreset?.topP
            ?? s.assistantTopP
        ))
        let topK = sc.topK
            ?? modelSettings?.topK
            ?? effectivePreset?.topK
            ?? s.assistantTopK
        let minP = sc.minP
            ?? modelSettings?.minP
            ?? effectivePreset?.minP
            ?? s.assistantMinP
        let repPenalty = sc.repetitionPenalty
            ?? modelSettings?.repetitionPenalty
            ?? effectivePreset?.repetitionPenalty
            ?? s.assistantRepetitionPenalty
        let freqPenalty = sc.frequencyPenalty
            ?? effectivePreset?.frequencyPenalty
            ?? s.assistantFrequencyPenalty
        let presPenalty = sc.presencePenalty
            ?? effectivePreset?.presencePenalty
            ?? s.assistantPresencePenalty
        let seed: UInt64? = {
            if let s = sc.seed { return s }
            if let s = effectivePreset?.seed { return s }
            let stored = s.assistantSeed
            return stored != 0 ? stored : nil
        }()

        // `let` (not `var`): every sampler knob this MLX build supports is set
        // in the initializer, so params is never mutated afterward — and an
        // immutable value is safe to capture in the concurrent generate closure
        // below (a captured `var` is a Swift 6 concurrency error).
        // The family profile owns KV precision/window and prefill chunking.
        // Normal models retain the previous 8-bit/unbounded behavior; Bonsai
        // 27B uses a rotating 2K 4-bit cache so text generation remains below
        // the iPhone process watermark.
        let params = GenerateParameters(
            maxTokens: safeMaxTokens,
            maxKVSize: executionProfile.maxKVSize,
            kvBits: executionProfile.kvBits,
            temperature: Float(temp),
            topP: Float(topP),
            topK: topK,
            minP: Float(minP),
            repetitionPenalty: Float(repPenalty),
            prefillStepSize: executionProfile.prefillStepSize
        )
        // The vendored mlx-swift-lm `GenerateParameters` exposes topK /
        // minP (wired above) and the KV-cache quantization knobs; it has
        // no seed parameter, and the frequency / presence penalties are
        // intentionally left at the package defaults for now. The resolved
        // values are still computed above (settings UI, sampler presets),
        // so discard them explicitly to keep them out of dead-code warnings.
        _ = (freqPenalty, presPenalty, seed)

        // --- JSON Mode (Feature #2) ---
        // Inject JSON-output instructions into the last system message.
        var effectiveMessages = messages
        let useJSONMode = jsonMode || s.jsonModeEnabled
        if useJSONMode {
            let jsonHint = s.jsonSchemaHint.isEmpty
                ? "Respond with valid JSON only. No markdown fences, no commentary."
                : "Respond with valid JSON matching this schema. No markdown fences, no commentary:\n\(s.jsonSchemaHint)"
            if let sysIdx = effectiveMessages.lastIndex(where: { $0.role == .system }) {
                var updated = effectiveMessages[sysIdx]
                updated.content += "\n\n[JSON MODE] \(jsonHint)"
                effectiveMessages[sysIdx] = updated
            } else {
                effectiveMessages.insert(
                    ChatMessage(role: .system, content: "[JSON MODE] \(jsonHint)"),
                    at: 0
                )
            }
        }

        // Trim conversation history to fit within the model's practical context
        // window, capped by tier: an unclamped 28K-token prompt's KV cache
        // (~140 KB/token fp16 on a 4B model) adds multiple GB that the load
        // gate never accounted for. 8-bit KV quantization (above) halves the
        // per-token cost; this cap bounds the count.
        let isImportedGGUF = activeModel.runtime == .llamaCpp
        let messagesForRuntime: [ChatMessage]
        if isImportedGGUF {
            // Imported recurrent Gemma models do not accept system turns, but
            // long-conversation memory must not disappear with the rest of the
            // system prompt. Recast only our bounded memory record as ordinary
            // chat context; persona/tool system instructions remain excluded.
            messagesForRuntime = effectiveMessages.compactMap { message in
                guard message.role == .system else { return message }
                guard message.content.hasPrefix("[CONVERSATION MEMORY]") else {
                    return nil
                }
                return ChatMessage(
                    id: message.id,
                    role: .user,
                    content: message.content,
                    timestamp: message.timestamp
                )
            }
        } else {
            messagesForRuntime = effectiveMessages
        }
        let inputBudget = isImportedGGUF
            ? Self.importedGGUFInputBudget(repoID: activeModel.repoID)
            : executionProfile.inputBudget(
                modelContextWindowTokens: executionProfile.maxContextTokens,
                deviceContextCap: executionProfile.maxContextTokens,
                requestedOutputTokens: safeMaxTokens
            )
        let trimmedMessages = Self.trimToInputBudget(messagesForRuntime, maxTokens: inputBudget)

        // Expose estimated input token count so the UI can render context-window bar.
        estimatedInputTokens = trimmedMessages.map {
            $0.contentForModel.count / 4 + 4
        }.reduce(0, +)

        // --- Chat Template Support (Feature #6) ---
        // Use the model's resolved chat template instead of hardcoded Qwen3 template.
        // Non-ChatML models (Llama, Gemma, Phi) now get proper formatting.
        let template = activeModel.chatTemplate
        let enableThinking = resolvedThinkingEnabled(forceNoThinking: forceNoThinking)
        let manualPrompt: String? = template.supportsThinking
            ? trimmedMessages.formattedWithTemplate(template, enableThinking: enableThinking)
            : trimmedMessages.formattedWithTemplate(template)

        // NOTE: the MLXLMCommon `[Chat.Message]` for the fallback path is built
        // INSIDE the generation closure (below), from `trimmedMessages`. That
        // app-owned `[ChatMessage]` is Sendable; the MLX `[Chat.Message]` is
        // not, so constructing it here and capturing it in the @Sendable gate
        // closure is a Swift 6 error. Building it at the use site captures only
        // the Sendable source — same result, no boundary violation.

        // --- Logprobs Collection (Feature #3) ---
        // This MLX build's `Generation` yields only `.chunk` / `.info` — no
        // per-token logprobs to accumulate or rank — so we stream each token
        // through onLogprobToken with neutral values (the documented
        // "unavailable → no-op" path) rather than keeping a write-only array.
        let shouldCollectLogprobs = collectLogprobs || s.logprobsEnabled

        // Capture model ID for the usage tracker.
        let trackerModelID = activeModel.id

        // Assistant image-viewing (when the loaded runtime is the dual-role
        // shared vision container): any user turn carrying imageThumbnails
        // is passed to the model via MLXVLM `.ciImage` inputs below. Text-only
        // runtimes — bounded text container (a dual-role model loaded with no
        // headroom for its vision envelope) or a GGUF assistant loaded without
        // an mmproj projector — were flagged `!isVisionChatCapable` at load
        // time, so the chat stays text-only and any attached thumbnails are
        // display-only in the chat bubble. This mirrors TokenAI's MLXEngine:
        // image thumbnails only become `UserInput.Image`s on user turns when
        // a vision runtime actually consumed them.
        let useVisionChat = isVisionChatCapable
            && trimmedMessages.contains {
                $0.role == .user
                    && (!$0.imageThumbnails.isEmpty || $0.imageThumbnailData != nil)
            }

        if let ggufModel, activeModel.runtime == .llamaCpp {
            let service = self
            let generationID = UUID()
            activeGGUFGenerationID = generationID
            let loggedFirstToken = OSAllocatedUnfairLock(initialState: false)
            let ggufMaxTokens = min(
                safeMaxTokens,
                Self.importedGGUFMaxOutputTokens(repoID: activeModel.repoID)
            )
            let ggufPrompt = trimmedMessages
                .filter { $0.role != .system }
                .map { message in
                    "\(message.role.rawValue): \(message.contentForModel)"
                }
                .joined(separator: "\n\n")
            // mtmd currently consumes one bitmap per Assistant turn. Prefer
            // the newest user image (the one the question refers to) while
            // retaining all attachments for MLX VLMs and transcript display.
            let ggufImageData: Data? = trimmedMessages.reversed().lazy
                .filter { $0.role == .user }
                .compactMap { message in
                    message.imageThumbnails.first?.data ?? message.imageThumbnailData
                }
                .first
            let useGGUFVision = useVisionChat && ggufImageData != nil
            Diagnostics.shared.breadcrumb(
                "GGUF generation start · vision=\(useGGUFVision) · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel)",
                category: "assistant"
            )
            generateTask = Task.detached(priority: .userInitiated) {
                do {
                    let tokenSink: @Sendable (String) -> Void = { token in
                            let isFirst = loggedFirstToken.withLock { logged -> Bool in
                                guard !logged else { return false }
                                logged = true
                                return true
                            }
                            if isFirst {
                                Diagnostics.shared.breadcrumb(
                                    "GGUF first token · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel)",
                                    category: "assistant"
                                )
                            }
                            onToken(token)
                    }
                    let rate: Double
                    let generationResult: AssistantGenerationResult?
                    if useGGUFVision, let ggufImageData {
                        guard let image = UIImage(data: ggufImageData) else {
                            throw LlamaCppError.bitmapInitFailed
                        }
                        let completedRate = OSAllocatedUnfairLock(initialState: 0.0)
                        try ggufModel.describe(
                            image: image,
                            prompt: ggufPrompt,
                            maxTokens: ggufMaxTokens,
                            onToken: tokenSink,
                            onComplete: { value in
                                completedRate.withLock { $0 = value }
                            }
                        )
                        rate = completedRate.withLock { $0 }
                        generationResult = nil
                    } else {
                        let result = try ggufModel.generateText(
                            // Imported GGUFs are standalone community models.
                            // Keep actual chat turns and let the bridge add BOS.
                            prompt: ggufPrompt,
                            maxTokens: ggufMaxTokens,
                            onToken: tokenSink
                        )
                        rate = result.tokensPerSecond
                        generationResult = AssistantGenerationResult(
                            promptTokenCount: result.promptTokenCount,
                            completionTokenCount: result.completionTokenCount,
                            stopReason: result.stopReason == .length ? .length : .stop
                        )
                    }
                    // `generateText` has returned, so its prompt/decode batches
                    // and generation guard are fully released before the UI can
                    // initiate an automatic recovery/tool follow-up.
                    Diagnostics.shared.breadcrumb(
                        "GGUF generation complete · rate=\(String(format: "%.2f", rate)) · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel)",
                        category: "assistant"
                    )
                    await MainActor.run {
                        let ownsState = service.activeGGUFGenerationID == generationID
                        if ownsState {
                            service.activeGGUFGenerationID = nil
                            service.generateTask = nil
                            if case .generating = service.state { service.state = .ready }
                        }
                        if ownsState {
                            service.tokenRate = rate
                            if let generationResult {
                                onGenerationResult?(generationResult)
                            }
                            ModelUsageTracker.shared.recordGeneration(
                                modelID: trackerModelID,
                                tokens: generationResult?.completionTokenCount ?? Int(rate),
                                tokensPerSecond: rate
                            )
                        }
                        onComplete(rate)
                    }
                } catch is CancellationError {
                    await MainActor.run {
                        if service.activeGGUFGenerationID == generationID {
                            service.activeGGUFGenerationID = nil
                            service.generateTask = nil
                            if case .generating = service.state { service.state = .ready }
                        }
                        onComplete(0)
                    }
                } catch LlamaCppError.cancelled {
                    await MainActor.run {
                        // User cancellation is an expected control flow, not a
                        // model failure. Keep the loaded runtime reusable and
                        // avoid exposing the retry path while it is resident.
                        if service.activeGGUFGenerationID == generationID {
                            service.activeGGUFGenerationID = nil
                            service.generateTask = nil
                            if case .generating = service.state { service.state = .ready }
                        }
                        Diagnostics.shared.breadcrumb(
                            "GGUF assistant generation cancelled",
                            category: "assistant"
                        )
                        onComplete(0)
                    }
                } catch {
                    await MainActor.run {
                        if service.activeGGUFGenerationID == generationID {
                            service.activeGGUFGenerationID = nil
                            service.generateTask = nil
                            // A failed decode does not necessarily invalidate
                            // the resident model. Keep it reusable when the
                            // native runtime is still present; otherwise the
                            // next API request is permanently rejected as
                            // "not loaded or busy" until the user reloads it.
                            service.state = service.hasResidentRuntime
                                ? .ready
                                : .failed(error.localizedDescription)
                        }
                        Diagnostics.shared.error(
                            "GGUF assistant generation failed · \(error.localizedDescription)",
                            category: "assistant"
                        )
                        onComplete(0)
                    }
                }
            }
            return
        }

        guard let container = resolvedMLXContainer else { onComplete(0); return }

        let generationID = UUID()
        activeMLXGenerationID = generationID
        generateTask = Task {
            // Watch for iOS memory warnings — bail out gracefully instead of
            // getting Jetsam-killed silently.
            let memoryWarningTask = Task { @MainActor in
                let center = NotificationCenter.default
                for await _ in center.notifications(
                    named: UIApplication.didReceiveMemoryWarningNotification
                ).map({ _ in () }) {
                    print("[CodingAssistantService] Memory warning — queueing GPU cache clear")
                    await MLXGenerationGate.shared.clearCacheWhenIdle()
                    // Keep generating; only stop on repeated rapid warnings.
                    break
                }
            }
            defer { memoryWarningTask.cancel() }

            // Watch for thermal escalation — if the device crosses into
            // .critical mid-stream, stop so we don't push it further into a
            // thermal shutdown. We deliberately do NOT stop at .serious: that
            // state is normal during prompt prefill / sustained inference on
            // modern silicon (iOS throttles clocks there on its own), and
            // stopping on it produced spurious "device too hot" interruptions
            // mid-reply. Matches DeviceSafetyMonitor's throttle-at-serious,
            // stop-at-critical schedule.
            let thermalTask = Task { @MainActor [weak self] in
                let center = NotificationCenter.default
                for await _ in center.notifications(
                    named: ProcessInfo.thermalStateDidChangeNotification
                ).map({ _ in () }) {
                    guard let self else { break }
                    let s = ProcessInfo.processInfo.thermalState
                    if s == .critical {
                        print("[CodingAssistantService] Thermal \(s.rawValue) — stopping")
                        self.stopGeneration()
                        ToastCenter.shared.error(
                            "Stopped — device too hot",
                            detail: "Set the device down to cool, then continue."
                        )
                        break
                    }
                }
            }
            defer { thermalTask.cancel() }

            do {
                let start = Date()
                // Capture the final tokens/sec SYNCHRONOUSLY inside the loop.
                // The @Published `tokenRate` is updated via an unordered
                // `Task { @MainActor … }` hop, so reading it back after the
                // loop raced those hops and recorded a stale (often 0) rate.
                let finalRate = OSAllocatedUnfairLock<Double>(initialState: 0)
                let finalGenerationResult = OSAllocatedUnfairLock<AssistantGenerationResult?>(
                    initialState: nil
                )

                // Funnel through MLXGenerationGate to serialize against
                // FastVLM / MLXVision — concurrent submits to Metal's
                // command queue cause the C++ check_error abort.
                let generationModelKey = activeModel.repoID
                let generationSupportsThinking = activeModel.supportsThinking
                try await MLXGenerationGate.shared.run { [container] in
                    let previousMemoryLimit = MLX.Memory.memoryLimit
                    let previousCacheLimit = MLX.Memory.cacheLimit
                    if let limit = lowMemoryPolicy.memoryLimitBytes {
                        MLX.Memory.memoryLimit = limit
                    }
                    // The load policy's normal 256 MiB cache bound prevents
                    // transient loader buffers from growing without limit.
                    // During generation retain the stricter model-family
                    // profile (usually zero, 64 MiB for generic models).
                    MLX.Memory.cacheLimit = min(
                        lowMemoryPolicy.cacheLimitBytes
                            ?? executionProfile.cacheLimitBytes,
                        executionProfile.cacheLimitBytes
                    )
                    defer {
                        MLX.Memory.memoryLimit = previousMemoryLimit
                        MLX.Memory.cacheLimit = previousCacheLimit
                    }
                    // Inside the gate, this clear is ordered after all prior
                    // MLX operations and before this generation's first submit.
                    mlxClearCache()
                    try await container.perform { context in
                        // Use the manually-built prompt from the model's chat template.
                        // Previously only Qwen3 got a manual prompt; now ALL models do
                        // via ChatTemplate.format(). The fallback to UserInput(chat:)
                        // is kept for models where the template is .generic (ChatML),
                        // and is also forced ON when image inputs are attached —
                        // manualPrompt is text-only, so it can't carry pixels.
                        // Tool-aware requests must pass structured chat plus
                        // `UserInput.tools` through the tokenizer's own chat
                        // template. A manually formatted prompt cannot carry
                        // tool schemas or correlate tool results.
                        let useManualPrompt = nativeTools.isEmpty
                            && !template.format.hasPrefix("generic")
                            && !useVisionChat
                        var userInput: UserInput
                        if useManualPrompt, let manualPrompt {
                            userInput = UserInput(prompt: manualPrompt)
                        } else {
                            // Build the MLX chat array here (see note above) so
                            // the non-Sendable [Chat.Message] never crosses the
                            // @Sendable boundary. When `useVisionChat`, attach
                            // each user turn's `imageThumbnails` as MLXVLM
                            // `.ciImage` inputs so the dual-role vision container
                            // actually sees the pixels. Images ride ONLY user
                            // turns — Qwen-VL's chat template emits the
                            // `<|image_pad|>` placeholder for user content, and
                            // an image embedded in an assistant turn leaves the
                            // processor with a frame and no matching placeholder
                            // ("Number of placeholder tokens does not match
                            // number of frames").
                            let chatMessages: [Chat.Message] = trimmedMessages.compactMap { msg -> Chat.Message? in
                                switch msg.role {
                                case .system:
                                    return .system(msg.contentForModel)
                                case .assistant:
                                    let calls = msg.toolCalls.flatMap { metadata in
                                        let parsed = Self.nativeToolCalls(from: metadata)
                                        return parsed.isEmpty ? nil : parsed
                                    }
                                    return .assistant(
                                        msg.content,
                                        toolCalls: calls
                                    )
                                case .tool:
                                    return .tool(msg.content, id: msg.toolCallID)
                                case .user:
                                    guard useVisionChat else { return .user(msg.contentForModel) }
                                    let imgs: [UserInput.Image] = msg.imageThumbnails.compactMap { att -> UserInput.Image? in
                                        guard let ui = UIImage(data: att.data),
                                              let ci = Self.ciImage(from: ui) else { return nil }
                                        return .ciImage(ci)
                                    }
                                    return .user(msg.contentForModel, images: imgs)
                                }
                            }
                            userInput = UserInput(chat: chatMessages)
                        }
                        if !nativeTools.isEmpty {
                            userInput.tools = Self.nativeToolSpecs(from: nativeTools)
                        }
                        // Qwen3-family processors consume this chat-template
                        // variable directly. Keep it model-aware so unrelated
                        // runtimes never receive an option they do not define.
                        // Manual-template models already receive the same
                        // decision through `formattedWithTemplate` above.
                        if generationSupportsThinking {
                            userInput.additionalContext = [
                                "enable_thinking": enableThinking
                            ]
                        }

                        var generationContext = context
                        if let nativeToolFormat {
                            generationContext.configuration.toolCallFormat = switch nativeToolFormat {
                            case .hermesJSON: .json
                            case .qwenXML: .xmlFunction
                            }
                        }
                        let lmInput = try await generationContext.processor.prepare(input: userInput)
                        let cache = generationContext.model.newCache(parameters: params)

                        let generationStream: AsyncStream<Generation>
                        if nativeTools.isEmpty,
                           let toolConstraint,
                           toolConstraint.isSuitableForNativeConstraint {
                            let iterator = try MLXToolCallConstraintRuntime.makeIterator(
                                input: lmInput,
                                context: generationContext,
                                cache: cache,
                                parameters: params,
                                configuration: toolConstraint,
                                modelKey: generationModelKey
                            )
                            // The app owns the API-facing envelope and parser.
                            // Select a non-JSON package tool format here so
                            // MLXLMCommon does not consume our private envelope
                            // before the local API can normalize it.
                            var generationConfiguration = generationContext.configuration
                            generationConfiguration.toolCallFormat = .xmlFunction
                            let result = MLXLMCommon.generateTask(
                                promptTokenCount: lmInput.text.tokens.size,
                                modelConfiguration: generationConfiguration,
                                tokenizer: generationContext.tokenizer,
                                iterator: iterator
                            )
                            generationStream = result.0
                        } else {
                            generationStream = try MLXLMCommon.generate(
                                input: lmInput,
                                cache: cache,
                                parameters: params,
                                context: generationContext
                            )
                        }

                        var tokenIndex = 0
                        for await generation in generationStream {
                            if Task.isCancelled { break }
                            if let chunk = generation.chunk {
                                onToken(chunk)

                                // Collect logprobs if enabled (Feature #3).
                                // MLX may not return logprobs natively; this is a
                                // best-effort collection from whatever the framework
                                // provides. When logprobs are unavailable, this is a
                                // no-op and the caller gets an empty array.
                                if shouldCollectLogprobs {
                                    let lp = TokenLogprob(
                                        tokenIndex: tokenIndex,
                                        token: chunk,
                                        logprob: 0,
                                        topAlternatives: []
                                    )
                                    onLogprobToken?(lp)
                                    tokenIndex += 1
                                }
                            }
                            if case .toolCall(let call) = generation {
                                onToken(Self.encodedNativeToolCall(call))
                            }
                            if let info = generation.info {
                                let rate = info.tokensPerSecond
                                finalRate.withLock { $0 = rate }
                                let stopReason: AssistantGenerationResult.StopReason = switch info.stopReason {
                                case .stop: .stop
                                case .length: .length
                                case .cancelled: .cancelled
                                }
                                finalGenerationResult.withLock {
                                    $0 = AssistantGenerationResult(
                                        promptTokenCount: info.promptTokenCount,
                                        completionTokenCount: info.generationTokenCount,
                                        stopReason: stopReason
                                    )
                                }
                                Task { @MainActor [weak self] in self?.tokenRate = rate }
                            }
                        }

                        mlxClearCache()
                    }
                }

                let elapsed = Date().timeIntervalSince(start)
                let rate = elapsed > 0 ? finalRate.withLock({ $0 }) : 0
                let generationResult = finalGenerationResult.withLock { $0 }
                let generatedTokens = generationResult?.completionTokenCount
                    ?? Int(rate * elapsed)
                await MainActor.run { [weak self] in
                    guard AssistantGenerationOwnership.isCurrent(
                        activeID: self?.activeMLXGenerationID,
                        completingID: generationID
                    ) else {
                        onComplete(rate)
                        return
                    }
                    self?.activeMLXGenerationID = nil
                    self?.generateTask = nil
                    // Restore .ready only if still generating — unload() may
                    // have torn the container down mid-flight, and flipping
                    // back to .ready would show "ready" with nothing loaded.
                    if case .generating = self?.state { self?.state = .ready }
                    if let generationResult {
                        onGenerationResult?(generationResult)
                    }
                    ModelUsageTracker.shared.recordGeneration(
                        modelID: trackerModelID,
                        tokens: generatedTokens,
                        tokensPerSecond: rate
                    )
                    onComplete(rate)
                    // Re-arm the background prefetch now that the LLM
                    // is idle again. If the user is between messages
                    // we'll prime the VLM's page cache for the next
                    // tab switch.
                    ModelResidency.shared.schedulePrefetch(currentTab: .assistant)
                }

            } catch is CancellationError {
                await MainActor.run { [weak self] in
                    guard AssistantGenerationOwnership.isCurrent(
                        activeID: self?.activeMLXGenerationID,
                        completingID: generationID
                    ) else {
                        onComplete(0)
                        return
                    }
                    self?.activeMLXGenerationID = nil
                    self?.generateTask = nil
                    if case .generating = self?.state { self?.state = .ready }
                    onComplete(0)
                }

            } catch is MLXGenerationGate.Cancelled {
                // Gate drained — same UX as a user-initiated cancel.
                await MainActor.run { [weak self] in
                    guard AssistantGenerationOwnership.isCurrent(
                        activeID: self?.activeMLXGenerationID,
                        completingID: generationID
                    ) else {
                        onComplete(0)
                        return
                    }
                    self?.activeMLXGenerationID = nil
                    self?.generateTask = nil
                    if case .generating = self?.state { self?.state = .ready }
                    onComplete(0)
                }

            } catch {
                await MainActor.run { [weak self] in
                    guard let self else {
                        onComplete(0)
                        return
                    }
                    guard AssistantGenerationOwnership.isCurrent(
                        activeID: self.activeMLXGenerationID,
                        completingID: generationID
                    ) else {
                        onComplete(0)
                        return
                    }
                    self.activeMLXGenerationID = nil
                    self.generateTask = nil
                    // Preserve a resident runtime after a transient
                    // generation error. Marking the service failed here made
                    // every following Hermes request return 503 even though
                    // the model was still loaded and could recover.
                    self.state = self.hasResidentRuntime
                        ? .ready
                        : .failed(error.localizedDescription)
                    onComplete(0)
                }
            }
        }
    }

    // MARK: - Apple Private Cloud generation

    func refreshApplePrivateCloudStatus() async {
        guard ApplePrivateCloud.isSupportedOnCurrentOS else {
            applePrivateCloudStatus = .unsupportedOS
            applePrivateCloudContextSize = nil
            return
        }
        let status = await ApplePrivateCloud.currentStatus()
        applePrivateCloudStatus = status
        applePrivateCloudContextSize = status.canSend
            ? await ApplePrivateCloud.contextSize()
            : nil
    }

    /// Selects PCC without pretending it is a downloaded model. No prompt is
    /// sent here; the picker must collect versioned privacy consent first.
    @discardableResult
    func selectApplePrivateCloud(
        persistAsDefault: Bool = true
    ) async -> Bool {
        guard ApplePrivateCloud.isSupportedOnCurrentOS else {
            ToastCenter.shared.error(
                "Apple Private Cloud unavailable",
                detail: ApplePrivateCloud.unavailableError.localizedDescription
            )
            return false
        }
        if case .loading = state {
            ToastCenter.shared.info(
                "Model still loading",
                detail: "Wait for the current local model load to finish before switching."
            )
            return false
        }
        guard !isTransitioning else { return false }

        isTransitioning = true
        defer { isTransitioning = false }

        if activeExecutionLocation != .applePrivateCloud {
            stopGeneration()
            await unloadAndWaitForCleanup()
            activeExecutionLocation = .applePrivateCloud
        }
        if persistAsDefault {
            AppSettings.shared.hasPickedAssistantModel = true
            AppSettings.shared.assistantModelID = ApplePrivateCloud.modelID
        }
        await refreshApplePrivateCloudStatus()
        return true
    }

    private func generateWithApplePrivateCloud(
        messages: [ChatMessage],
        maxTokensOverride: Int?,
        temperatureOverride: Double?,
        jsonMode: Bool,
        forceNoThinking: Bool,
        onToken: @escaping @Sendable (String) -> Void,
        onComplete: @escaping @Sendable (Double) -> Void
    ) {
        let settings = AppSettings.shared
        guard settings.hasCurrentApplePCCPrivacyConsent else {
            ToastCenter.shared.error(
                "Review privacy before using Apple Private Cloud",
                detail: "Choose Apple Private Cloud again to review and accept the disclosure."
            )
            onComplete(0)
            return
        }
        guard applePrivateCloudGenerationState != .generating else {
            onComplete(0)
            return
        }

        pccGenerateTask?.cancel()
        let requestID = UUID()
        activePCCRequestID = requestID
        applePrivateCloudGenerationState = .generating
        tokenRate = 0

        pccGenerateTask = Task { [weak self] in
            guard let self else {
                onComplete(0)
                return
            }

            await self.refreshApplePrivateCloudStatus()
            guard self.applePrivateCloudStatus.canSend else {
                let error = self.applePrivateCloudError(
                    for: self.applePrivateCloudStatus
                )
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .failure(error),
                    onComplete: onComplete
                )
                return
            }
            guard let contextSize = self.applePrivateCloudContextSize else {
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .failure(.temporary("Apple Private Cloud context information is unavailable.")),
                    onComplete: onComplete
                )
                return
            }

            let maximumResponseTokens = max(
                1,
                min(maxTokensOverride ?? settings.assistantMaxTokens, contextSize - 1)
            )
            let inputBudget = ApplePrivateCloud.inputBudget(
                contextSize: contextSize,
                maximumResponseTokens: maximumResponseTokens
            )
            var effectiveMessages = messages
            if jsonMode || settings.jsonModeEnabled {
                let jsonHint = settings.jsonSchemaHint.isEmpty
                    ? "Respond with valid JSON only. No markdown fences or commentary."
                    : "Respond with valid JSON matching this schema. No markdown fences or commentary:\n\(settings.jsonSchemaHint)"
                effectiveMessages.insert(
                    ChatMessage(role: .system, content: "[JSON MODE] \(jsonHint)"),
                    at: 0
                )
            }
            let trimmedMessages = Self.trimToInputBudget(
                effectiveMessages,
                maxTokens: inputBudget
            )
            self.estimatedInputTokens = trimmedMessages.reduce(0) {
                $0 + $1.contentForModel.count / 4 + 4
            }
            let conversation = ApplePrivateCloudPromptBuilder.build(
                messages: trimmedMessages
            )
            let requestedReasoning: ApplePCCReasoningLevel =
                forceNoThinking ? .light : settings.applePCCReasoningLevel
            let temperature = min(
                2,
                max(0, temperatureOverride ?? settings.assistantTemperature)
            )
            let request = ApplePCCRequest(
                id: requestID,
                prompt: conversation.prompt,
                instructions: conversation.instructions,
                reasoning: requestedReasoning,
                maximumResponseTokens: maximumResponseTokens,
                temperature: temperature
            )

            do {
                let stream = await ApplePrivateCloud.stream(request)
                for try await delta in stream {
                    try Task.checkCancellation()
                    onToken(delta)
                }
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .success(()),
                    onComplete: onComplete
                )
            } catch is CancellationError {
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .failure(.cancelled),
                    onComplete: onComplete
                )
            } catch let error as ApplePCCError {
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .failure(error),
                    onComplete: onComplete
                )
            } catch {
                self.finishApplePrivateCloudRequest(
                    id: requestID,
                    result: .failure(.unknown(error.localizedDescription)),
                    onComplete: onComplete
                )
            }
        }
    }

    private func finishApplePrivateCloudRequest(
        id: UUID,
        result: Result<Void, ApplePCCError>,
        onComplete: @escaping @Sendable (Double) -> Void
    ) {
        guard activePCCRequestID == id else { return }
        activePCCRequestID = nil
        pccGenerateTask = nil
        switch result {
        case .success:
            applePrivateCloudGenerationState = .idle
        case .failure(.cancelled):
            applePrivateCloudGenerationState = .idle
        case .failure(let error):
            applePrivateCloudGenerationState = .failed(error)
            ToastCenter.shared.error(
                "Apple Private Cloud unavailable",
                detail: error.localizedDescription
            )
        }
        onComplete(0)
    }

    private func applePrivateCloudError(
        for status: ApplePCCStatus
    ) -> ApplePCCError {
        switch status {
        case .limitReached:
            return .quotaExceeded
        case .offline:
            return .offline
        case .unsupportedOS:
            return ApplePrivateCloud.unavailableError
        case .unsupportedDevice:
            return .unavailable("This device isn't eligible for Apple Private Cloud.")
        case .appleIntelligenceUnavailable:
            return .unavailable("Apple Intelligence isn't ready. Check Settings and try again.")
        case .temporarilyUnavailable, .entitlementUnavailable, .unknown:
            return .temporary("")
        case .ready, .approachingLimit:
            return .unknown("")
        }
    }

    // MARK: - Context window trimming

    /// Trims a conversation to fit within `maxTokens` of estimated input.
    /// Always preserves system messages and the most-recent turn pair.
    /// Drops the oldest non-system messages first (same strategy used by
    /// ChatGPT, Claude, and Gemini mobile apps to handle long conversations).
    ///
    /// Token estimate: 1 token ≈ 4 characters (conservative for code+English).
    nonisolated static func trimToInputBudget(
        _ messages: [ChatMessage],
        maxTokens: Int
    ) -> [ChatMessage] {
        let tokenEstimate: (ChatMessage) -> Int = { msg in
            // +4 for per-message overhead (role header, separators)
            msg.contentForModel.count / 4 + 4
        }

        let systemMessages   = messages.filter { $0.role == .system }
        let dialogMessages   = messages.filter { $0.role != .system }

        let systemBudgetUsed = systemMessages.map(tokenEstimate).reduce(0, +)
        var remaining        = maxTokens - systemBudgetUsed

        guard let latestIndex = dialogMessages.indices.last else {
            return systemMessages
        }

        // Reserve the newest message first. It is normally the user's current
        // prompt and must survive even if it alone exceeds the estimate.
        var keptByIndex: [Int: ChatMessage] = [
            latestIndex: dialogMessages[latestIndex]
        ]
        remaining = max(0, remaining - tokenEstimate(dialogMessages[latestIndex]))

        // A terse follow-up such as "Have you summarized all?" depends on the
        // complete prior turn: both the user's subject and the assistant's
        // response. Reserve the prior user message first so a long assistant
        // reply cannot consume the entire budget and erase what "all" means.
        let priorAssistantIndex = dialogMessages.indices
            .reversed()
            .first { index in
                index < latestIndex && dialogMessages[index].role == .assistant
            }
        let priorUserIndex = priorAssistantIndex.flatMap { assistantIndex in
            dialogMessages.indices
                .reversed()
                .first { index in
                    index < assistantIndex && dialogMessages[index].role == .user
                }
        }
        if let priorUserIndex, remaining > 4 {
            let priorUser = dialogMessages[priorUserIndex]
            let fullCost = tokenEstimate(priorUser)
            if fullCost <= remaining {
                keptByIndex[priorUserIndex] = priorUser
                remaining -= fullCost
            } else {
                // Grounding payloads can dwarf the visible request. When the
                // whole source no longer fits, preserve the displayed request
                // so pronouns and follow-up intent still have an antecedent.
                var visibleOnly = priorUser
                visibleOnly.modelContent = nil
                let visibleCost = tokenEstimate(visibleOnly)
                if visibleCost <= remaining {
                    keptByIndex[priorUserIndex] = visibleOnly
                    remaining -= visibleCost
                }
            }
        }

        // Preserve the prior assistant response next. If it does not fit
        // whole, retain its tail because models conventionally put conclusions
        // and proposed next actions at the end.
        if let priorAssistantIndex, remaining > 4 {
            let prior = dialogMessages[priorAssistantIndex]
            let priorCost = tokenEstimate(prior)
            if priorCost <= remaining {
                keptByIndex[priorAssistantIndex] = prior
                remaining -= priorCost
            } else {
                let truncationPrefix = "…\n"
                let characterBudget = max(
                    0,
                    (remaining - 4) * 4 - truncationPrefix.count
                )
                if characterBudget > 0 {
                    keptByIndex[priorAssistantIndex] = ChatMessage(
                        id: prior.id,
                        role: prior.role,
                        content: truncationPrefix
                            + String(prior.contentForModel.suffix(characterBudget)),
                        timestamp: prior.timestamp
                    )
                }
                remaining = 0
            }
        }

        // Fill any remaining budget newest → oldest, skipping the protected
        // immediate context above.
        if remaining > 0 {
            for index in dialogMessages.indices.reversed()
                where keptByIndex[index] == nil {
                let message = dialogMessages[index]
                let cost = tokenEstimate(message)
                guard cost <= remaining else { break }
                keptByIndex[index] = message
                remaining -= cost
            }
        }

        let kept = dialogMessages.indices.compactMap { keptByIndex[$0] }
        return systemMessages + kept
    }

    // MARK: - Vision input helper

    /// Safely convert a `UIImage` into a `CIImage` for MLX-VLM image input.
    /// Returns `nil` (rather than force-unwrapping) for images backed by
    /// neither a `CGImage` nor a `CIImage` — e.g. some symbol/vector or
    /// filtered sources that would otherwise crash the vision processor.
    /// Mirrors the helper in TokenAI's `MLXEngine` so a degenerate
    /// thumbnail is skipped instead of poisoning the chat-array build.
    nonisolated static func ciImage(from image: UIImage) -> CIImage? {
        if let ci = image.ciImage { return ci }
        if let cg = image.cgImage { return CIImage(cgImage: cg) }
        return nil
    }

    // MARK: - Stop

    func stopGeneration() {
#if CORE_AI_SERVER_APP
        CoreAIInferenceService.shared.cancel()
#endif
        if let activePCCRequestID {
            ApplePrivateCloud.cancel(activePCCRequestID)
        }
        pccGenerateTask?.cancel()
        ggufModel?.cancelCurrent()
        generateTask?.cancel()
        // Do not expose `.ready` until the native decode has observed the
        // cancellation and fully unwound. Starting another request while the
        // old context is still freeing is unsafe in llama.cpp.
    }

    /// Waits for the native generation task to finish its final cache/runtime
    /// cleanup. API callers use this at the boundary between two sequential
    /// requests; without it, the first response could be visible while the
    /// service still reported `.generating` for the next request.
    func waitForGenerationToFinish(timeout: Duration = .seconds(5)) async -> Bool {
        guard case .generating = state else {
            return state == .ready && hasResidentRuntime
        }
        guard let inflight = generateTask else {
            return false
        }

        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                _ = await inflight.value
            }
            group.addTask {
                try? await Task.sleep(for: timeout)
            }
            _ = await group.next()
            group.cancelAll()
        }

        return state == .ready && hasResidentRuntime
    }

    /// Called during background transition. Cooperatively cancels the active
    /// generation and waits briefly for it to unwind. Does NOT block
    /// indefinitely — the background task expiration handler cancels the
    /// waiting Task.
    func cancelAndDrainInference() async {
        let inflight = generateTask
        let pccInflight = pccGenerateTask
        stopGeneration()
        Diagnostics.shared.breadcrumb(
            "assistant inference cancel requested · slot=assistant · model=\(activeModel.id)",
            category: "lifecycle"
        )
        // Holding the task and awaiting its value is the synchronization
        // boundary between decoding and teardown. Without it, background
        // cleanup could release an MLX/llama.cpp runtime while its final Metal
        // command buffer or native decode frame was still retiring.
        if let inflight {
            _ = await inflight.value
        }
        if let pccInflight {
            _ = await pccInflight.value
        }
        Diagnostics.shared.breadcrumb(
            "assistant inference drained · slot=assistant · model=\(activeModel.id)",
            category: "lifecycle"
        )
    }

    // MARK: - Unload

    func unload() {
        // Same drain-before-clear pattern as MLXVisionService.unload():
        // calling mlxClearCache() while a generate task is mid-Metal
        // command-buffer raced the completion handler into
        // `mlx::core::gpu::check_error` and SIGABRTed. Wait for the
        // in-flight task to actually end, THEN flush the buffer pool.
        cancelLoad()
        activeLoadID = nil
        state = .unloaded
        refreshVisionChatCapability()
        _ = beginUnload(policy: .external)
    }

    /// Async variant for callers that need deterministic memory
    /// reclamation before proceeding with another large model load.
    func unloadAndWaitForCleanup(
        policy: AssistantUnloadDrainPolicy = .external
    ) async {
        let task = beginUnload(policy: policy)
        await task.value
    }

    private func beginUnload(
        policy: AssistantUnloadDrainPolicy
    ) -> Task<Void, Never> {
        if let unloadTask { return unloadTask }

        let taskID = UUID()
        unloadTaskID = taskID
        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.performUnloadCleanup(policy: policy)
            guard self.unloadTaskID == taskID else { return }
            self.unloadTask = nil
            self.unloadTaskID = nil
        }
        unloadTask = task
        return task
    }

    private func performUnloadCleanup(
        policy: AssistantUnloadDrainPolicy
    ) async {
#if CORE_AI_SERVER_APP
        stopGeneration()
        loadTask?.cancel()
        loadTask = nil
        await CoreAIInferenceService.shared.suspend()
        state = .unloaded
        return
#endif
        let footprintBefore = MemoryAdvisor.physFootprint
        let headroomBefore = MemoryAdvisor.availableMemoryForModel
        let hadGGUFAtStart = ggufModel != nil
        Diagnostics.shared.breadcrumb(
            "assistant unload begin · gguf=\(hadGGUFAtStart) · footprint=\(footprintBefore) · headroom=\(headroomBefore)",
            category: "assistant"
        )
        var inflight = generateTask
        var inflightLoad = policy.loadTask ? loadTask : nil
        var inflightTransition = policy.transitionTask ? transitionTask : nil
        activeLoadID = nil
        if policy.loadTask {
            // Invalidate publication before cancelling so a late MLX result
            // cannot install a container after the unload has begun.
            loadTaskID = nil
            loadTask?.cancel()
        }
        generateTask = nil
        activeMLXGenerationID = nil
        inflight?.cancel()
        ggufModel?.cancelCurrent()
        if let inflightLoad {
            Diagnostics.shared.breadcrumb(
                "assistant unload draining load task",
                category: "assistant"
            )
            _ = await inflightLoad.value
        }
        if let inflightTransition {
            Diagnostics.shared.breadcrumb(
                "assistant unload draining transition task",
                category: "assistant"
            )
            _ = await inflightTransition.value
        }
        if let inflight {
            Diagnostics.shared.breadcrumb(
                "assistant unload draining generation task",
                category: "assistant"
            )
            _ = await inflight.value
        }
        // A completed Task may keep its closure context alive until its last
        // Task handle is released. Drop these local handles before handing
        // off the container so a captured ModelContainer cannot become the
        // hidden final owner during teardown.
        inflightLoad = nil
        inflightTransition = nil
        inflight = nil
        if policy.loadTask {
            loadTask = nil
        }
        if policy.transitionTask {
            transitionTask = nil
        }
        // Release runtime ownership only after the cancelled generation has
        // completely unwound. This ordering is required for both MLX Metal
        // command buffers and llama.cpp's native decode context.
        let releasedGGUF = ggufModel != nil
        let ggufRelease = GGUFRuntimeReleaseBox(ggufModel)
        ggufModel = nil
        ggufVisionProjectorPath = nil
        let mlxRelease = MLXContainerReleaseBox(container)
        container = nil
        Diagnostics.shared.breadcrumb(
            "assistant unload releasing MLX runtime off main actor",
            category: "assistant"
        )
        await MLXGenerationGate.shared.cleanupRuntimeWhenIdle {
            mlxRelease.release()
        }
        Diagnostics.shared.breadcrumb(
            "assistant unload MLX runtime released",
            category: "assistant"
        )
        if releasedGGUF {
            Diagnostics.shared.breadcrumb(
                "assistant unload releasing GGUF runtime off main actor",
                category: "assistant"
            )
            await Task.detached(priority: .utility) {
                ggufRelease.release()
                autoreleasepool { }
            }.value
            Diagnostics.shared.breadcrumb(
                "assistant unload GGUF runtime released",
                category: "assistant"
            )
        }
        autoreleasepool { }

        // llama.cpp/Metal can retire residency-set allocations shortly after
        // the Swift owner is released. Give that accounting time to settle
        // before another multi-GB runtime starts allocating.
        if releasedGGUF {
            var previous = MemoryAdvisor.physFootprint
            var stableSamples = 0
            for _ in 0..<25 {
                try? await Task.sleep(nanoseconds: 200_000_000)
                autoreleasepool { }
                let current = MemoryAdvisor.physFootprint
                if abs(current - previous) < 64 * 1_024 * 1_024 {
                    stableSamples += 1
                    if stableSamples >= 3 { break }
                } else {
                    stableSamples = 0
                }
                previous = current
            }
        }
        Diagnostics.shared.breadcrumb(
            "assistant unload complete · footprint=\(MemoryAdvisor.physFootprint) · headroom=\(MemoryAdvisor.availableMemoryForModel)",
            category: "assistant"
        )
        state = .unloaded
        refreshVisionChatCapability()
    }

    // MARK: - Switch model

    /// Hot-swap the active model. Conversation-level switches leave the
    /// user's saved default unchanged; settings/model-management callers can
    /// opt into making the choice the default. No-op when already active.
    func switchTo(
        _ model: AssistantModel,
        persistAsDefault: Bool = true
    ) async {
#if CORE_AI_SERVER_APP
        // The Core AI product has one manifest-selected runtime. The picker is
        // retained for visual parity, but it cannot switch this target back to
        // an MLX or llama.cpp model.
        return
#endif
        // MLX package loading is intentionally non-interruptible: invalidating
        // activeLoadID only prevents a late result from publishing; it does
        // not stop the loader from allocating the model. Starting a switch
        // while that work is active used to let a stale multi-GB load continue
        // behind the newly selected model and made background cleanup wait on
        // it. Keep the current selection stable until the load completes.
        if case .loading = state {
            ToastCenter.shared.info(
                "Model still loading",
                detail: "Wait for the current load to finish before choosing another model."
            )
            return
        }
        guard !isTransitioning else {
            ToastCenter.shared.info("Model already loading",
                                     detail: "Please wait for the current load to finish.")
            return
        }
        if activeExecutionLocation == .applePrivateCloud {
            let pccInflight = pccGenerateTask
            stopGeneration()
            if let pccInflight {
                _ = await pccInflight.value
            }
            activeExecutionLocation = .localDownloaded
            applePrivateCloudGenerationState = .idle
        }
        let hasLoadedModel = model.runtime == .llamaCpp
            ? ggufModel != nil
            : resolvedMLXContainer != nil
        if model.id == activeModel.id, hasLoadedModel {
            if persistAsDefault {
                AppSettings.shared.hasPickedAssistantModel = true
                AppSettings.shared.assistantModelID = model.id
            }
            return
        }
        if let compatibility = model.platformCompatibility,
           !compatibility.supportsCurrentPlatform {
            ToastCenter.shared.error("Can't load \(model.displayName)",
                                     detail: compatibility.detail)
            return
        }

        isTransitioning = true
        defer { isTransitioning = false }

        if persistAsDefault {
            // Persist only explicit default choices. A conversation switch
            // must not silently replace what future chats use.
            AppSettings.shared.hasPickedAssistantModel = true
            AppSettings.shared.assistantModelID = model.id
        }
        activeModel = model

        // Unload any active LoRA adapter when switching models.
        LoRAAdapterStore.shared.activeAdapterID = nil
        LoRAAdapterStore.shared.activeAdapterIDs = []

        // Fully reclaim the previous model's memory BEFORE loading the next
        // one. `unload()` returns immediately and (when a generate is still
        // in flight) defers `mlxClearCache()` to a detached task, so the old
        // weights + Metal buffers are still resident when `load()`'s memory
        // gate reads `os_proc_available_memory()`. That mismatch is the
        // "switch to a bigger model → 'only 2.0 GB available to the app'"
        // failure: the gate counts the model we're about to free. Awaiting the
        // synchronous-cleanup variant drains the in-flight task, frees the
        // GPU cache, and lets the per-process headroom recover first.
        // This method is running inside `transitionTask`. Awaiting the
        // transition handle from the shared cleanup path deadlocks the task
        // against itself and leaves the final breadcrumb at "draining
        // transition task" until iOS watchdog-terminates the app.
        await unloadAndWaitForCleanup(policy: .transitionOwned)
        guard !Task.isCancelled else {
            Diagnostics.shared.breadcrumb(
                "assistant transition cancelled after cleanup · \(model.id)",
                category: "assistant"
            )
            return
        }
        ToastCenter.shared.info("Loading \(model.displayName)…")
        // The caller supplied an explicit model. Conversation-level switches
        // intentionally do not persist that choice as the default, so a
        // settings reselect here would immediately replace `model` with the
        // saved default before loading it.
        await load(reselectFromSettings: false)
    }

    /// Loads a small model for a one-shot background analysis without changing
    /// the user's persisted assistant choice. The caller must finish with
    /// `finishTemporaryAnalysis(restoring:)` so the temporary weights are
    /// reclaimed and the picker returns to the original selection unloaded.
    func prepareTemporaryAnalysisModel(_ model: AssistantModel) async -> AssistantModel? {
        let maximumSafeFootprint: Int64 = 1_600_000_000
        guard model.runtime == .mlx,
              model.approxRAMBytes <= maximumSafeFootprint,
              model.platformCompatibility?.supportsCurrentPlatform ?? true else {
            return nil
        }

        let original = activeModel
        await unloadAndWaitForCleanup()
        activeModel = model
        await load(allowStorageFallback: false, reselectFromSettings: false)
        guard case .ready = state else {
            activeModel = original
            return nil
        }
        return original
    }

    /// Reclaims a temporary analysis model and restores the user's assistant
    /// identity without reloading its weights. The next normal chat send keeps
    /// the existing lazy-load behavior.
    func finishTemporaryAnalysis(restoring model: AssistantModel) async {
        await unloadAndWaitForCleanup()
        activeModel = model
        state = .unloaded
    }

    /// Restore `model` as the active selection WITHOUT loading it.
    /// Voice-mode teardown uses this: ending a conversation shouldn't
    /// eagerly pull a multi-GB chat model back into memory just to put
    /// the picker back — generate()'s lazy-load path brings the model
    /// up on the next send instead.
    func adoptSelectionWithoutLoading(_ model: AssistantModel) async {
        activeModel = model
        await unloadAndWaitForCleanup()
    }

    // MARK: - LoRA Adapter Support (Feature #9)

    /// Whether a LoRA adapter is currently loaded alongside the base model.
    private(set) var loadedLoRAID: String? = nil

    /// Load a LoRA adapter onto the currently active base model.
    /// MLX must support LoRA loading via the adapter path.
    /// NOTE: LoRA support in mlx-swift is experimental. This method
    /// attempts to load adapter weights and gracefully falls back
    /// if the API is unavailable.
    func loadLoRA(adapterID: String) async {
        guard let adapter = LoRAAdapterStore.shared.adapter(for: adapterID) else {
            ToastCenter.shared.error("LoRA not found", detail: "Adapter \(adapterID) is not in the catalog.")
            return
        }
        guard adapter.baseModelID == activeModel.id || adapter.baseModelID.isEmpty else {
            ToastCenter.shared.error("Incompatible LoRA",
                                     detail: "\(adapter.name) targets \(adapter.baseModelID), but \(activeModel.displayName) is loaded.")
            return
        }
        guard case .ready = state, resolvedMLXContainer != nil else {
            ToastCenter.shared.error("No model loaded", detail: "Load a base model before attaching a LoRA.")
            return
        }

        // Determine adapter path: local > HF cache > repo download.
        let adapterPath: URL?
        if let local = adapter.localPath {
            adapterPath = URL(fileURLWithPath: local)
        } else if let repoID = adapter.repoID {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let dashed = repoID.replacingOccurrences(of: "/", with: "--")
            adapterPath = docs.appendingPathComponent("huggingface/hub/models--\(dashed)")
        } else {
            adapterPath = nil
        }

        guard let path = adapterPath, FileManager.default.fileExists(atPath: path.path) else {
            ToastCenter.shared.error("LoRA weights missing",
                                     detail: "Download the adapter before loading it.")
            return
        }

        // LoRA support in mlx-swift is evolving. Attempt the load;
        // if the API isn't available yet, surface a clear message.
        // The container's LoRA methods will be available once
        // mlx-swift exposes them in a future release.
        ToastCenter.shared.info(
            "LoRA support coming soon",
            detail: "\(adapter.name) adapter weights found at \(path.lastPathComponent). Full LoRA loading will be available in a future mlx-swift update."
        )
    }

    /// Unload the currently active LoRA adapter, restoring the base model.
    func unloadLoRA() async {
        guard loadedLoRAID != nil else { return }
        loadedLoRAID = nil
        LoRAAdapterStore.shared.activeAdapterID = nil
        LoRAAdapterStore.shared.activeAdapterIDs = []
        ToastCenter.shared.info("LoRA removed")
    }
}
