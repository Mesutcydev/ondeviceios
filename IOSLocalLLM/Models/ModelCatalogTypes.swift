import SwiftUI

// MARK: - ModelCatalogTypes
//
// Shared metadata enums for catalog entries. Lives at the Model layer
// (not Views/) so non-UI code — download manager, gated-repo probe,
// onboarding picker filters — can reason about capabilities and
// vendor without importing SwiftUI.
//
// `KCapabilityPill` and `KVendorThumb` in Views/Components/ consume
// these and own the visual rendering.

// MARK: - Platform compatibility

/// Curated platform support for models whose runtime requirements cannot be
/// inferred from RAM alone. `MemoryAdvisor` still decides whether a supported
/// model fits the exact device; this handles hard publisher constraints such
/// as a build explicitly marked as laptop-only.
enum ModelPlatformCompatibility: String, Codable, Hashable, Sendable {
    case mobileAndMac
    case highMemoryMobileAndMac
    case macOnly

    var label: String {
        switch self {
        case .mobileAndMac:           return "iPhone · iPad · Mac"
        case .highMemoryMobileAndMac: return "high-memory iOS · Mac"
        case .macOnly:                return "Mac only"
        }
    }

    var symbol: String {
        switch self {
        case .mobileAndMac:           return "iphone.and.arrow.forward"
        case .highMemoryMobileAndMac: return "memorychip"
        case .macOnly:                return "laptopcomputer"
        }
    }

    var detail: String {
        switch self {
        case .mobileAndMac:
            return "Runs on supported iPhone, iPad, and Apple silicon Mac devices."
        case .highMemoryMobileAndMac:
            return "Requires a high-memory iPhone or iPad; Apple silicon Macs are also supported. Exact fit depends on live memory."
        case .macOnly:
            return "This build exceeds the iOS per-app memory budget and is available only on Apple silicon Mac."
        }
    }

    var supportsCurrentPlatform: Bool {
        #if os(macOS) || targetEnvironment(macCatalyst)
        return true
        #else
        return self != .macOnly
        #endif
    }
}

// MARK: - ModelCapability
//
// User-facing capability tags shown as colored pills on catalog rows.
// Multiple can apply per model (e.g. Qwen3-VL gets [.vision, .thinking,
// .best]).
//
// Ordering matches the display order on a row: status tags first
// (recommended/best/new), then capability tags (vision/thinking),
// then a gated lock last.

public enum ModelCapability: String, CaseIterable, Codable, Hashable, Sendable {
    case recommended    // we suggest this for the user's device
    case best           // peak quality at this size class
    case newRelease     // shipped recently — surface to discovery
    case vision         // multimodal (image + text in)
    case thinking       // emits reasoning trace (Qwen3 /think, DeepSeek-R1)
    case tools          // reliably emits/consumes tool-call JSON (function calling)
    case coder          // code-specialized weights (Qwen2.5-Coder, etc.)
    case fast           // sub-2B / latency-tuned — snappiest decode
    case multilingual   // strong non-English coverage (Qwen, Gemma, Kokoro voices)
    case gated          // requires HF token

    /// Title-cased label for the pill ("Vision", "Thinking", …).
    var label: String {
        switch self {
        case .recommended:  return "Recommended"
        case .best:         return "Best"
        case .newRelease:   return "New"
        case .vision:       return "Vision"
        case .thinking:     return "Thinking"
        case .tools:        return "Tools"
        case .coder:        return "Coder"
        case .fast:         return "Fast"
        case .multilingual: return "Multilingual"
        case .gated:        return "Gated"
        }
    }

    /// SF Symbol for the pill leading glyph.
    var symbol: String {
        switch self {
        case .recommended:  return "checkmark.seal.fill"
        case .best:         return "crown.fill"
        case .newRelease:   return "sparkles"
        case .vision:       return "eye.fill"
        case .thinking:     return "lightbulb.fill"
        case .tools:        return "wrench.and.screwdriver.fill"
        case .coder:        return "chevron.left.forwardslash.chevron.right"
        case .fast:         return "bolt.fill"
        case .multilingual: return "globe"
        case .gated:        return "lock.fill"
        }
    }

    /// Foreground hue for the pill — semantic-fixed so users can
    /// learn the color → meaning mapping across the app. The Kodu
    /// theme's rose palette doesn't apply here; semantic markers
    /// need to be distinct from brand color.
    var tint: Color {
        switch self {
        case .recommended: return Color(red: 0.255, green: 0.722, blue: 0.392)   // emerald
        case .best:        return Color(red: 0.196, green: 0.659, blue: 0.396)   // forest green
        case .newRelease:  return Color(red: 0.961, green: 0.486, blue: 0.149)   // pumpkin
        case .vision:      return Color(red: 0.557, green: 0.745, blue: 0.431)   // sage — matches T.accent2
        case .thinking:    return Color(red: 0.612, green: 0.420, blue: 0.871)   // amethyst
        case .tools:       return Color(red: 0.204, green: 0.553, blue: 0.898)   // azure
        case .coder:       return Color(red: 0.133, green: 0.600, blue: 0.553)   // teal
        case .fast:        return Color(red: 0.945, green: 0.769, blue: 0.204)   // amber
        case .multilingual:return Color(red: 0.400, green: 0.420, blue: 0.847)   // indigo
        case .gated:       return Color(red: 0.878, green: 0.443, blue: 0.498)   // rose (matches T.accent)
        }
    }

    // MARK: - Inference
    //
    // Capability markers inferable from a model's identity, surfaced as
    // pills in addition to any curated/explicit capabilities. Centralizes
    // the "does this row deserve a Tools / Coder / Fast / Multilingual
    // pill" decision so every render path (assistant picker rows, catalog
    // family cards) agrees instead of each re-deriving from substrings.
    //
    // `supportsTools` stays the single source of truth for the .tools pill
    // — we surface it here rather than duplicating it into every preset's
    // `capabilities` literal.

    static func inferred(repoID: String,
                         tags: [String] = [],
                         supportsTools: Bool = false) -> Set<ModelCapability> {
        var out: Set<ModelCapability> = []
        let lower = repoID.lowercased()
        let tagSet = Set(tags.map { $0.lowercased() })

        if supportsTools { out.insert(.tools) }
        if lower.contains("coder") || lower.contains("-code") || tagSet.contains("code") {
            out.insert(.coder)
        }
        if tagSet.contains("fast") || tagSet.contains("light") {
            out.insert(.fast)
        }
        // Multilingual: families with strong documented non-English coverage.
        // Deliberately narrow — a bare repo id with none of these markers
        // gets no multilingual pill rather than a guessed one.
        if lower.contains("qwen") || lower.contains("bonsai") || lower.contains("gemma")
            || lower.contains("aya") || lower.contains("bloom") {
            out.insert(.multilingual)
        }
        return out
    }
}

// MARK: - ModelRuntime
//
// Which on-device inference engine actually executes a model. This is a
// real, testable axis on iPhone: the same family can run through Apple's
// MLX (4/6/8-bit safetensors) or through llama.cpp (GGUF), with different
// speed / memory / quality trade-offs. Surfacing it lets the compare and
// benchmark surfaces label a result honestly ("Qwen3-4B · MLX 4-bit" vs
// "… · GGUF Q4_K_M") instead of pretending every number is comparable.

public enum ModelRuntime: String, Codable, Hashable, CaseIterable, Sendable {
    case mlx        // Apple MLX (MLXLLM / MLXVLM) — safetensors
    case llamaCpp   // llama.cpp + GGUF (LlamaCppBridge / mtmd)

    /// Short badge label shown on picker rows and result cards.
    var label: String {
        switch self {
#if CORE_AI_SERVER_APP
        case .mlx:      return "Core AI"
#else
        case .mlx:      return "MLX"
#endif
        case .llamaCpp: return "GGUF"
        }
    }
}

// MARK: - ModelVendor
//
// Issuer/publisher of the model — drives the thumbnail. Inferred from
// the repo ID prefix in most cases (apple/, google/, mlx-community/,
// ggml-org/, mistralai/, Qwen/, ibm-granite/) so existing catalog
// entries get a vendor for free without manual annotation.

enum ModelVendor: String, Codable, Hashable {
    case apple
    case google
    case mistral
    case qwen
    case ibm
    case meta
    case huggingFace
    case ggmlOrg
    case mlxCommunity
    case liquid
    case prism
    case generic

    /// Best-effort inference from a HuggingFace repo id.
    /// "google/gemma-3-4b-it" → .google, "Qwen/Qwen3-VL" → .qwen, etc.
    static func infer(from repoID: String) -> ModelVendor {
        let lower = repoID.lowercased()
        // org-prefix match (before the slash)
        let org = lower.split(separator: "/").first.map(String.init) ?? lower
        switch org {
        case "apple":                       return .apple
        case "google":                      return .google
        case "mistralai":                   return .mistral
        case "qwen":                        return .qwen
        case "ibm-granite", "ibm":          return .ibm
        case "meta-llama", "meta":          return .meta
        case "huggingfacetb", "huggingface":return .huggingFace
        case "ggml-org":                    return .ggmlOrg
        case "mlx-community":               return .mlxCommunity
        case "liquidai":                    return .liquid
        case "prism-ml":                    return .prism
        default: break
        }
        // body-substring fallback (e.g. "someone/qwen3-finetune")
        if lower.contains("qwen")    { return .qwen }
        if lower.contains("gemma")   { return .google }
        if lower.contains("llama")   { return .meta }
        if lower.contains("mistral") || lower.contains("ministral") { return .mistral }
        if lower.contains("granite") { return .ibm }
        if lower.contains("smolvlm") || lower.contains("smollm") { return .huggingFace }
        if lower.contains("bonsai") { return .prism }
        return .generic
    }

    // MARK: - Thumbnail rendering data
    //
    // We render thumbnails procedurally (monogram letter on a brand
    // gradient) rather than ship per-vendor assets. Reasons:
    //   • Zero binary bloat
    //   • Survives any future logo changes
    //   • Renders consistently in light/dark without dual assets

    /// Single character/short string drawn on the tile.
    var monogram: String {
        switch self {
        case .apple:        return ""    // SF Symbol used instead
        case .google:       return "G"
        case .mistral:      return "M"
        case .qwen:         return "Q"
        case .ibm:          return "IBM"
        case .meta:         return "M"
        case .huggingFace:  return "🤗"
        case .ggmlOrg:      return "g"
        case .mlxCommunity: return "x"
        case .liquid:       return "L"
        case .prism:        return "P"
        case .generic:      return "•"
        }
    }

    /// SF Symbol to draw instead of `monogram`, when one fits better
    /// (Apple).
    var systemSymbol: String? {
        switch self {
        case .apple: return "apple.logo"
        default:     return nil
        }
    }

    /// Gradient pair for the tile background. The two stops are the
    /// top-left → bottom-right corners.
    var gradient: (Color, Color) {
        switch self {
        case .apple:        return (Color(white: 0.18), Color(white: 0.04))
        case .google:       return (Color(red: 0.302, green: 0.561, blue: 0.992),    // #4d8fff
                                    Color(red: 0.169, green: 0.396, blue: 0.890))    // #2b65e3
        case .mistral:      return (Color(red: 0.992, green: 0.529, blue: 0.180),    // #FD872E
                                    Color(red: 0.875, green: 0.247, blue: 0.067))    // #DF3F11
        case .qwen:         return (Color(red: 0.475, green: 0.451, blue: 0.984),    // #7973FB
                                    Color(red: 0.341, green: 0.282, blue: 0.847))    // #5748D8
        case .ibm:          return (Color(red: 0.063, green: 0.388, blue: 0.984),    // #1063FB
                                    Color(red: 0.024, green: 0.180, blue: 0.580))    // #062E94
        case .meta:         return (Color(red: 0.314, green: 0.514, blue: 0.969),    // #5083F7
                                    Color(red: 0.106, green: 0.282, blue: 0.733))    // #1B48BB
        case .huggingFace:  return (Color(red: 1.000, green: 0.835, blue: 0.314),    // #FFD550
                                    Color(red: 0.969, green: 0.620, blue: 0.122))    // #F79E1F
        case .ggmlOrg:      return (Color(red: 0.231, green: 0.231, blue: 0.247),    // slate
                                    Color(red: 0.082, green: 0.082, blue: 0.106))
        case .mlxCommunity: return (Color(red: 0.196, green: 0.659, blue: 0.604),    // teal
                                    Color(red: 0.063, green: 0.392, blue: 0.392))
        case .liquid:       return (Color(red: 0.451, green: 0.694, blue: 0.969),    // sky
                                    Color(red: 0.235, green: 0.435, blue: 0.808))
        case .prism:        return (Color(red: 0.553, green: 0.353, blue: 0.969),    // violet
                                    Color(red: 0.255, green: 0.137, blue: 0.663))
        case .generic:      return (Color(red: 0.482, green: 0.475, blue: 0.529),
                                    Color(red: 0.282, green: 0.275, blue: 0.337))
        }
    }

    /// Display name for the vendor — surfaces in the "by Google",
    /// "by Mistral AI" caption under a model row when we want extra
    /// context.
    var displayName: String {
        switch self {
        case .apple:        return "Apple"
        case .google:       return "Google"
        case .mistral:      return "Mistral AI"
        case .qwen:         return "Qwen"
        case .ibm:          return "IBM"
        case .meta:         return "Meta"
        case .huggingFace:  return "Hugging Face"
        case .ggmlOrg:      return "ggml-org"
        case .mlxCommunity: return "MLX Community"
        case .liquid:       return "Liquid AI"
        case .prism:        return "Prism ML"
        case .generic:      return "Community"
        }
    }
}

enum LocalModelRole: String, CaseIterable, Codable, Hashable {
    case assistant
    case vision
    case voice
    case image      // text-to-image diffusion (SD / SDXL); not runnable through the LLM/VLM/voice runtimes
}

/// Routing contract for a single model package that can serve Assistant and
/// Lens. A fitting VLM runtime may be shared, while oversized vision models
/// remain usable through an independently bounded Assistant text runtime.
enum DualRoleModelPolicy {
    static func isTextAndVision(repoID: String) -> Bool {
        AssistantModelCatalog.presets.contains {
            $0.repoID.caseInsensitiveCompare(repoID) == .orderedSame
                && $0.runtime == .mlx
                && $0.capabilities.contains(.vision)
        }
    }

    @MainActor
    static func selectionsMatch(repoID: String) -> Bool {
        let assistantRepo = AssistantModelCatalog.currentSelection().repoID
        let visionSelection = LocalModelRegistry.storedVisionSelectionID(
            AppSettings.shared.cameraVisualModelID
        )
        return isTextAndVision(repoID: repoID)
            && assistantRepo.caseInsensitiveCompare(repoID) == .orderedSame
            && visionSelection.caseInsensitiveCompare(repoID) == .orderedSame
    }

    /// A unified package should own both roles only when the user selected the
    /// same package for each role *and* its vision runtime fits the live
    /// process budget. Text-only execution can be substantially smaller than
    /// image execution, so a dual-role catalog flag alone is not sufficient.
    /// Kept injectable for regression tests and for non-iOS hosts where live
    /// process-memory APIs may not be available.
    static nonisolated func shouldShareRuntime(
        isDualRole: Bool,
        selectionsMatch: Bool,
        requiredVisionBytes: UInt64,
        availableBytes: UInt64
    ) -> Bool {
        isDualRole
            && selectionsMatch
            && requiredVisionBytes > 0
            && availableBytes >= requiredVisionBytes
    }
}

enum LocalModelOrigin: String, CaseIterable, Codable, Hashable {
    case system
    case preset
    case bundled
    case downloaded
    case imported
    case custom
}

struct LocalModelDescriptor: Identifiable, Hashable {
    let id: String
    let repoID: String
    let displayName: String
    let subtitle: String
    let role: LocalModelRole
    let origin: LocalModelOrigin
    let vendor: ModelVendor
    let capabilities: Set<ModelCapability>
    let runtime: ModelRuntime?
    let approxRAMBytes: Int64
    let contextWindowTokens: Int
    let supportsTools: Bool
    let voiceEngine: VoiceEngineKind?

    var assistantModel: AssistantModel? {
        guard role == .assistant else { return nil }
        return AssistantModel(
            id: id,
            repoID: repoID,
            displayName: displayName,
            subtitle: subtitle,
            approxRAMBytes: approxRAMBytes,
            tags: assistantTags,
            contextWindowTokens: contextWindowTokens,
            capabilities: capabilities,
            supportsTools: supportsTools,
            runtime: runtime ?? .mlx
        )
    }

    private var assistantTags: [String] {
        var tags: [String] = []
        switch origin {
        case .preset:
            break
        case .imported:
            tags.append("local")
        case .downloaded:
            tags.append("downloaded")
        case .custom:
            tags.append("custom")
        case .system:
            tags.append("system")
        case .bundled:
            tags.append("bundled")
        }
        if capabilities.contains(.thinking) { tags.append("thinking") }
        if supportsTools { tags.append("tools") }
        return tags
    }
}

enum LocalModelRegistry {
    static let assistantSelectionPrefixes = ["downloaded:", "imported:", "custom:"]
    static let defaultVisionSelectionID = FastVLMService.modelID

    /// Repository IDs that shipped as selectable visual models in older
    /// builds but now have a safer, API-compatible runtime representation.
    /// Keep this at the selection boundary so every caller (Lens, narrator,
    /// residency advisor, picker, and LocalAIController) sees one canonical
    /// model/backend without each feature maintaining its own migration.
    private static let legacyVisionSelectionAliases: [String: String] = [
        "mlx-community/gemma-3-4b-it-4bit": "ggml-org/gemma-3-4b-it-GGUF",
    ]

    private static let voiceKeywords = [
        "kitten", "kokoro", "kittentts", "kitten-tts", "kitten_tts",
        "whisper", "tts", "voicedesign", "voice-design", "voice_design",
        "speecht5", "speech-t5", "bark", "vits", "styletts", "style-tts",
        "parakeet", "moonshine", "piper",
    ]

    private static let visionKeywords = [
        "-vl", "_vl", "vision", "llava", "paligemma", "moondream",
        "smolvlm", "internvl", "minicpmv", "kosmos", "phi-3-v", "phi3v",
        "idefics",
    ]

    static func unwrapAssistantSelectionID(_ stored: String) -> String {
        for prefix in assistantSelectionPrefixes where stored.hasPrefix(prefix) {
            return String(stored.dropFirst(prefix.count))
        }
        return stored
    }

    static func assistantSelectionOrigin(for stored: String) -> LocalModelOrigin {
        if stored.hasPrefix("downloaded:") { return .downloaded }
        if stored.hasPrefix("imported:") { return .imported }
        if stored.hasPrefix("custom:") { return .custom }
        return .preset
    }

    static func assistantSelectionID(for repoID: String, origin: LocalModelOrigin) -> String {
        switch origin {
        case .preset:
            if let preset = AssistantModelCatalog.presets.first(where: { $0.repoID == repoID }) {
                return preset.id
            }
            return repoID
        case .imported:
            return "imported:\(repoID)"
        case .custom:
            return "custom:\(repoID)"
        case .downloaded:
            return "downloaded:\(repoID)"
        case .system, .bundled:
            return repoID
        }
    }

    static func assistantSelectionID(for model: DownloadableModel) -> String {
        if let preset = AssistantModelCatalog.presets.first(where: {
            $0.id == model.id || $0.repoID == model.sourceRepoID
        }) {
            return preset.id
        }
        return assistantSelectionID(for: model.sourceRepoID, origin: origin(for: model))
    }

    static func storedVisionSelectionID(_ storedRepoID: String) -> String {
        guard !storedRepoID.isEmpty else { return defaultVisionSelectionID }
        return legacyVisionSelectionAliases[storedRepoID.lowercased()] ?? storedRepoID
    }

    static func isDefaultVisionSelection(_ selectionID: String) -> Bool {
        selectionID.isEmpty || selectionID == defaultVisionSelectionID
    }

    static func persistedVisionRepoID(for selectionID: String) -> String {
        isDefaultVisionSelection(selectionID) ? "" : selectionID
    }

    static func setVisionSelection(_ selectionID: String, settings: AppSettings = .shared) {
        settings.cameraVisualModelID = persistedVisionRepoID(for: selectionID)
    }

    static func visionRuntime(forStoredSelectionID storedRepoID: String,
                              catalog: [DownloadableModel]) -> ModelRuntime {
        visualDescriptor(forStoredSelectionID: storedRepoID, catalog: catalog).runtime ?? .mlx
    }

    static func visionMemoryAdvisorKey(for selectionID: String) -> String {
        let normalized = storedVisionSelectionID(selectionID)
        if isDefaultVisionSelection(normalized) { return defaultVisionSelectionID }
        return "downloaded:\(persistedVisionRepoID(for: normalized))"
    }

    static func visualSelectionID(for model: DownloadableModel) -> String {
        model.id == defaultVisionSelectionID ? defaultVisionSelectionID : model.sourceRepoID
    }

    static func visualDescriptor(forStoredSelectionID storedRepoID: String,
                                 catalog: [DownloadableModel]) -> LocalModelDescriptor {
        let selectionID = storedVisionSelectionID(storedRepoID)
        if selectionID == defaultVisionSelectionID {
            if let fastVLM = catalog.first(where: { $0.id == defaultVisionSelectionID }) {
                return descriptor(for: fastVLM, forcedRole: .vision, forcedOrigin: .bundled)
            }
            return LocalModelDescriptor(
                id: defaultVisionSelectionID,
                repoID: "apple/FastVLM-0.5B-MLX",
                displayName: "FastVLM (built-in)",
                subtitle: "Apple's encoder + MLX decoder — runs everywhere",
                role: .vision,
                origin: .bundled,
                vendor: .apple,
                capabilities: [.vision],
                runtime: .mlx,
                approxRAMBytes: 1_400_000_000,
                contextWindowTokens: 0,
                supportsTools: false,
                voiceEngine: nil
            )
        }
        if let entry = catalog.first(where: {
            $0.id == selectionID || $0.sourceRepoID == selectionID
        }) {
            return descriptor(for: entry, forcedRole: .vision)
        }
        return LocalModelDescriptor(
            id: selectionID,
            repoID: selectionID,
            displayName: selectionID.split(separator: "/").last.map(String.init) ?? selectionID,
            subtitle: selectionID,
            role: .vision,
            origin: .downloaded,
            vendor: ModelVendor.infer(from: selectionID),
            capabilities: [.vision],
            runtime: selectionID.lowercased().contains("gguf") ? .llamaCpp : .mlx,
            approxRAMBytes: 0,
            contextWindowTokens: 0,
            supportsTools: false,
            voiceEngine: nil
        )
    }

    static func origin(for model: DownloadableModel) -> LocalModelOrigin {
        if model.id == FastVLMService.modelID { return .bundled }
        if model.id.hasPrefix("local/") { return .imported }
        if model.id.contains("/") { return .downloaded }
        return .preset
    }

    static func role(for category: DownloadableModel.Category) -> LocalModelRole {
        switch category {
        case .assistant: return .assistant
        case .vlm:       return .vision
        case .voice:     return .voice
        case .imageGen:  return .image
        }
    }

    static func category(for summary: HFModelSummary) -> DownloadableModel.Category {
        category(repoID: summary.id, pipelineTag: summary.pipelineTag, tags: summary.tags)
    }

    static func category(repoID: String, pipelineTag: String?, tags: [String] = []) -> DownloadableModel.Category {
        switch pipelineTag {
        case "text-to-speech",
             "automatic-speech-recognition":
            return .voice
        case "image-text-to-text",
             "image-to-text",
             "visual-question-answering":
            return .vlm
        case "text-to-image",
             "image-to-image",
             "unconditional-image-generation":
            return .imageGen
        default:
            break
        }

        let haystack = ([repoID] + tags).joined(separator: " ").lowercased()
        if voiceKeywords.contains(where: { haystack.contains($0) }) { return .voice }
        // Image-generation BEFORE vision: diffusion repos don't collide with
        // the VLM keyword list, and we want a text-to-image model recognized
        // as such rather than falling through to .assistant. `diffusers` is a
        // common tag on every SD/SDXL/FLUX repo.
        if tags.contains(where: { $0.lowercased() == "diffusers" })
            || OnDeviceCompatibility.looksLikeImageGenRepo(haystack) {
            return .imageGen
        }
        if visionKeywords.contains(where: { haystack.contains($0) }) { return .vlm }
        return .assistant
    }

    static func category(in directory: URL) -> DownloadableModel.Category {
        let repoShape = category(repoID: directory.lastPathComponent, pipelineTag: nil)
        if repoShape == .voice { return .voice }
        if repoShape == .imageGen { return .imageGen }

        // Diffusers layout on disk: a `model_index.json` at the root, or the
        // `unet/` + `vae/` subfolders that every SD/SDXL checkpoint ships.
        // Recognized so a previously-downloaded image model lists under its
        // own header instead of masquerading as an assistant.
        let fm = FileManager.default
        if fm.fileExists(atPath: directory.appendingPathComponent("model_index.json").path) {
            return .imageGen
        }
        var isDir: ObjCBool = false
        let hasUNet = fm.fileExists(atPath: directory.appendingPathComponent("unet").path, isDirectory: &isDir) && isDir.boolValue
        let hasVAE = fm.fileExists(atPath: directory.appendingPathComponent("vae").path, isDirectory: &isDir) && isDir.boolValue
        if hasUNet && hasVAE { return .imageGen }

        let configPath = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            if let names = try? FileManager.default.contentsOfDirectory(atPath: directory.path) {
                if names.contains(where: {
                    let lower = $0.lowercased()
                    return (lower.hasPrefix("mmproj") || lower.contains("projector")) &&
                        lower.hasSuffix(".gguf")
                }) {
                    return .vlm
                }
                let hasVoiceArtifact = names.contains { name in
                    let lower = name.lowercased()
                    return lower == "voices.npz" || lower == "voices"
                        || lower.contains("g2p") || lower.hasSuffix(".mlmodelc")
                        || lower.hasSuffix(".mlpackage")
                }
                if hasVoiceArtifact { return .voice }
            }
            return .assistant
        }

        if json["vision_config"] != nil || json["image_token_id"] != nil || json["image_token_index"] != nil {
            return .vlm
        }
        if let modelType = json["model_type"] as? String {
            let inferred = category(repoID: modelType, pipelineTag: nil)
            if inferred != .assistant { return inferred }
        }
        if let architectures = json["architectures"] as? [String] {
            let inferred = category(repoID: architectures.joined(separator: " "), pipelineTag: nil)
            if inferred != .assistant { return inferred }
            if architectures.contains(where: { $0.lowercased().contains("forconditionalgeneration") }) {
                return .vlm
            }
        }
        return .assistant
    }

    static func voiceEngine(for model: DownloadableModel) -> VoiceEngineKind? {
        if let supported = model.supportedVoiceEngine { return supported }
        return voiceEngine(repoID: model.id, displayName: model.displayName)
    }

    static func voiceEngine(repoID: String, displayName: String = "") -> VoiceEngineKind? {
        let haystack = "\(repoID) \(displayName)".lowercased()
        if haystack.contains("kitten") { return .kittenTTS }
        if haystack.contains("kokoro") { return .kokoro }
        return nil
    }

    static func descriptor(for preset: AssistantModel) -> LocalModelDescriptor {
        LocalModelDescriptor(
            id: preset.id,
            repoID: preset.repoID,
            displayName: preset.displayName,
            subtitle: preset.subtitle,
            role: .assistant,
            origin: .preset,
            vendor: ModelVendor.infer(from: preset.repoID),
            capabilities: preset.capabilities,
            runtime: preset.runtime,
            approxRAMBytes: preset.approxRAMBytes,
            contextWindowTokens: preset.contextWindowTokens,
            supportsTools: preset.supportsTools,
            voiceEngine: nil
        )
    }

    static func descriptor(forStoredAssistantID stored: String,
                           catalog: [DownloadableModel]) -> LocalModelDescriptor? {
        if let preset = AssistantModelCatalog.presets.first(where: { $0.id == stored }) {
            return descriptor(for: preset)
        }

        let origin = assistantSelectionOrigin(for: stored)
        let repoID = unwrapAssistantSelectionID(stored)
        guard !repoID.isEmpty else { return nil }
        if let entry = catalog.first(where: {
            $0.id == repoID || $0.sourceRepoID == repoID
        }) {
            let base = descriptor(for: entry, forcedRole: .assistant, forcedOrigin: origin)
            return LocalModelDescriptor(
                id: stored,
                repoID: base.repoID,
                displayName: base.displayName,
                subtitle: base.subtitle,
                role: base.role,
                origin: origin,
                vendor: base.vendor,
                capabilities: base.capabilities,
                runtime: base.runtime,
                approxRAMBytes: base.approxRAMBytes,
                contextWindowTokens: base.contextWindowTokens,
                supportsTools: base.supportsTools,
                voiceEngine: nil
            )
        }
        return customAssistantDescriptor(repoID: repoID, origin: origin, storedID: stored)
    }

    static func descriptor(for model: DownloadableModel,
                           forcedRole: LocalModelRole? = nil,
                           forcedOrigin: LocalModelOrigin? = nil) -> LocalModelDescriptor {
        let role = forcedRole ?? self.role(for: model.category)
        let origin = forcedOrigin ?? origin(for: model)
        let repoID = model.sourceRepoID
        let runtime: ModelRuntime? = {
            if let explicit = model.runtime { return explicit }
            switch role {
            case .assistant, .vision:
                return repoID.lowercased().contains("gguf") ? .llamaCpp : .mlx
            case .voice, .image:
                return nil
            }
        }()
        var caps = model.capabilities
        if role == .vision { caps.insert(.vision) }
        if role == .assistant && supportsThinking(repoID: repoID) { caps.insert(.thinking) }
        if KnownGatedRepos.isGated(repoID: repoID) { caps.insert(.gated) }
        // Identity-inferred markers (tools/coder/fast/multilingual) so the
        // catalog family cards surface the same pills as the assistant picker.
        if role == .assistant {
            caps.formUnion(ModelCapability.inferred(
                repoID: repoID,
                supportsTools: inferredAssistantSupportsTools(for: repoID)
            ))
        }

        let approxRAMBytes: Int64 = {
            if let explicit = model.approxRAMBytes { return explicit }
            if role == .assistant { return inferredAssistantRAMBytes(for: repoID) }
            if let engine = voiceEngine(for: model) {
                switch engine {
                case .appleSystem: return 0
                case .kittenTTS:
                    return repoID.lowercased().contains("mini") ? 700_000_000 : 250_000_000
                case .kokoro:
                    return 400_000_000
                }
            }
            return 0
        }()

        let contextWindowTokens: Int = {
            if let explicit = model.contextWindowTokens { return explicit }
            if role == .assistant { return inferredAssistantContextWindow(for: repoID) }
            return 0
        }()

        return LocalModelDescriptor(
            id: role == .assistant ? assistantSelectionID(for: repoID, origin: origin) : repoID,
            repoID: repoID,
            displayName: model.displayName,
            subtitle: model.subtitle.isEmpty ? fallbackSubtitle(for: role, repoID: repoID, origin: origin) : model.subtitle,
            role: role,
            origin: origin,
            vendor: ModelVendor.infer(from: repoID),
            capabilities: caps,
            runtime: runtime,
            approxRAMBytes: approxRAMBytes,
            contextWindowTokens: contextWindowTokens,
            supportsTools: role == .assistant ? inferredAssistantSupportsTools(for: repoID) : false,
            voiceEngine: voiceEngine(for: model)
        )
    }

    static func customAssistantDescriptor(repoID: String,
                                          origin: LocalModelOrigin = .custom,
                                          storedID: String? = nil) -> LocalModelDescriptor {
        let basename = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let selectionID = storedID ?? assistantSelectionID(for: repoID, origin: origin)
        return LocalModelDescriptor(
            id: selectionID,
            repoID: repoID,
            displayName: basename,
            subtitle: fallbackSubtitle(for: .assistant, repoID: repoID, origin: origin),
            role: .assistant,
            origin: origin,
            vendor: ModelVendor.infer(from: repoID),
            capabilities: supportsThinking(repoID: repoID) ? [.thinking] : [],
            runtime: repoID.lowercased().contains("gguf") ? .llamaCpp : .mlx,
            approxRAMBytes: inferredAssistantRAMBytes(for: repoID),
            contextWindowTokens: inferredAssistantContextWindow(for: repoID),
            supportsTools: inferredAssistantSupportsTools(for: repoID),
            voiceEngine: nil
        )
    }

    private static func fallbackSubtitle(for role: LocalModelRole,
                                         repoID: String,
                                         origin: LocalModelOrigin) -> String {
        switch role {
        case .assistant:
            switch origin {
            case .imported:  return "local · imported"
            case .downloaded:return "ready · on device"
            case .custom:    return "custom · \(repoID)"
            case .preset:    return repoID
            case .system:    return "built in"
            case .bundled:   return "bundled"
            }
        case .vision, .voice, .image:
            return repoID
        }
    }

    private static func inferredAssistantRAMBytes(for repoID: String) -> Int64 {
        if let params = OnDeviceCompatibility.paramCount(repoID: repoID) {
            return OnDeviceCompatibility.estimatedFootprint(
                params: params,
                tags: [],
                repoID: repoID
            )
        }
        return 0
    }

    private static func inferredAssistantContextWindow(for repoID: String) -> Int {
        let lower = repoID.lowercased()
        if lower.contains("qwen3") { return 32768 }
        if lower.contains("qwen2.5") || lower.contains("llama-3") || lower.contains("phi-3.5") {
            return 8192
        }
        if lower.contains("gemma-2") { return 4096 }
        return 4096
    }

    private static func inferredAssistantSupportsTools(for repoID: String) -> Bool {
        let lower = repoID.lowercased()
        return [
            "qwen", "llama", "gemma", "mistral", "ministral",
            "smollm", "deepseek", "phi"
        ].contains(where: { lower.contains($0) })
    }

    private static func supportsThinking(repoID: String) -> Bool {
        let lower = repoID.lowercased()
        return lower.contains("qwen3")
            || lower.contains("thinking")
            || lower.contains("deepseek-r1")
            || lower.contains("phi-4")
    }
}

// MARK: - KnownGatedRepos
//
// Hardcoded list of HF org/repo patterns known to require auth.
// Surface these with the .gated capability up-front so the user
// knows they need a token BEFORE tapping download. Falls back to
// a 401/403 probe at download time for anything not on this list
// (see HFModelDownloadManager).

// MARK: - ModelFamily
//
// Family grouping — multiple catalog entries that share a base
// model (different quantizations / sizes / variants) collapse to
// one row in the catalog UI ("Qwen 3 VL · 2 models"). The family
// id is derived from the repo basename, stripping size/quant
// suffixes so siblings hash to the same key.

enum ModelFamily {

    /// Returns a stable family identifier for `repoID`, used for
    /// grouping in catalog UI. Multiple repos collapse to the same
    /// family when they're variants (Qwen3-VL-2B-Instruct-4bit and
    /// Qwen3-VL-4B-Instruct-4bit both → "qwen3-vl").
    ///
    /// Conservative: when the basename doesn't match any known
    /// pattern, falls back to the basename itself, so unknown repos
    /// get a one-model family by default. The Catalog UI degrades
    /// gracefully to showing them as standalone rows.
    static func inferID(from repoID: String) -> String {
        let basename = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let lower = basename.lowercased()

        // Order matters — longer prefixes first so "qwen3-vl" doesn't
        // get absorbed into "qwen3".
        let patterns: [(needle: String, family: String)] = [
            ("bonsai",         "bonsai"),
            ("qwen3-vl",        "qwen3-vl"),
            ("qwen2.5-vl",      "qwen2.5-vl"),
            ("qwen2-vl",        "qwen2-vl"),
            ("qwen3",           "qwen3"),
            ("qwen2.5-coder",   "qwen2.5-coder"),
            ("qwen2.5",         "qwen2.5"),
            ("qwen2",           "qwen2"),
            ("smolvlm2",        "smolvlm2"),
            ("smolvlm",         "smolvlm"),
            ("smollm3",         "smollm3"),
            ("smollm2",         "smollm2"),
            ("smollm",          "smollm"),
            ("gemma-3n",        "gemma-3n"),
            ("gemma-3",         "gemma-3"),
            ("gemma-2",         "gemma-2"),
            ("gemma",           "gemma"),
            ("ministral",       "ministral"),
            ("mistral",         "mistral"),
            ("llama-3.2",       "llama-3.2"),
            ("llama-3.1",       "llama-3.1"),
            ("llama-3",         "llama-3"),
            ("llama",           "llama"),
            ("phi-4",           "phi-4"),
            ("phi-3.5",         "phi-3.5"),
            ("phi-3",           "phi-3"),
            ("granite-vision",  "granite-vision"),
            ("granite",         "granite"),
            ("fastvlm",         "fastvlm"),
            ("paligemma",       "paligemma"),
        ]
        for (needle, family) in patterns where lower.contains(needle) {
            return family
        }
        return lower
    }

    /// Human-readable display name for a family id. Title-cased and
    /// space-separated for the catalog row header ("Qwen 3 VL").
    static func displayName(forID id: String) -> String {
        let map: [String: String] = [
            "bonsai":          "Bonsai",
            "qwen3-vl":        "Qwen 3 VL",
            "qwen2.5-vl":      "Qwen 2.5 VL",
            "qwen2-vl":        "Qwen 2 VL",
            "qwen3":           "Qwen 3",
            "qwen2.5-coder":   "Qwen 2.5 Coder",
            "qwen2.5":         "Qwen 2.5",
            "qwen2":           "Qwen 2",
            "smolvlm2":        "SmolVLM 2",
            "smolvlm":         "SmolVLM",
            "smollm3":         "SmolLM 3",
            "smollm2":         "SmolLM 2",
            "smollm":          "SmolLM",
            "gemma-3n":        "Gemma 3n",
            "gemma-3":         "Gemma 3",
            "gemma-2":         "Gemma 2",
            "gemma":           "Gemma",
            "ministral":       "Ministral",
            "mistral":         "Mistral",
            "llama-3.2":       "Llama 3.2",
            "llama-3.1":       "Llama 3.1",
            "llama-3":         "Llama 3",
            "llama":           "Llama",
            "phi-4":           "Phi 4",
            "phi-3.5":         "Phi 3.5",
            "phi-3":           "Phi 3",
            "granite-vision":  "Granite Vision",
            "granite":         "Granite",
            "fastvlm":         "FastVLM",
            "paligemma":       "PaliGemma",
        ]
        if let known = map[id] { return known }
        // Fallback: title-case the id by splitting on '-' and '_'
        return id.split(whereSeparator: { $0 == "-" || $0 == "_" })
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}

enum KnownGatedRepos {
    /// Prefixes (case-insensitive) that mark a repo as gated. Use
    /// org-level prefixes where every model in the org is gated
    /// (e.g. "meta-llama/", "mistralai/"), or full repo IDs for
    /// per-model gating (Google's pattern).
    static let prefixes: [String] = [
        // Meta — all Llama 3.x weights require accepting a license
        "meta-llama/",
        // Mistral — Ministral series and 7B/Nemo lines all gated
        "mistralai/",
        // Google — Gemma releases on the official org are gated;
        // mlx-community/gemma-* mirrors are not.
        "google/gemma-",
        "google/paligemma-",
        // IBM Granite Vision — Apache 2 but still gated by org policy
        "ibm-granite/granite-vision-",
    ]

    static func isGated(repoID: String) -> Bool {
        let lower = repoID.lowercased()
        for p in prefixes where lower.hasPrefix(p.lowercased()) {
            return true
        }
        return false
    }
}

// MARK: - Installed model record

/// A locally-installed model discovered on disk — community downloads,
/// HF Search results, local imports, and catalog-preset downloads all
/// converge to this type. Unlike `DownloadableModel` (which represents
/// "what CAN be downloaded"), `InstalledModelRecord` represents "what
/// EXISTS locally and can be selected."
///
/// The registry (`InstalledModelRegistry`) is the single observable
/// source of truth consumed by every model picker.
struct InstalledModelRecord: Codable, Identifiable, Hashable, Sendable {
    let id: UUID
    let repoID: String                // canonical "author/name"
    let displayName: String
    let localURL: URL
    let engine: ModelRuntime           // .mlx or .llamaCpp
    let capabilities: Set<ModelCapability>
    let architecture: String?          // from config.json "architectures"
    let quantization: String?          // e.g. "ternary", "4bit", "1bit"
    let parameterCount: Double?        // billions
    let installedAt: Date
    let validationState: ValidationState
    let downloadBytes: Int64

    enum ValidationState: Codable, Hashable, Sendable {
        case valid
        case missingConfig
        case missingTokenizer
        case missingWeights
        case incomplete
        case unsupportedArchitecture(String)
        case unknown(String)

        var isActivatable: Bool {
            if case .valid = self { return true }
            return false
        }

        var description: String {
            switch self {
            case .valid:                  return "Ready"
            case .missingConfig:          return "Missing config.json"
            case .missingTokenizer:       return "Missing tokenizer files"
            case .missingWeights:         return "Missing model weights"
            case .incomplete:             return "Download incomplete"
            case .unsupportedArchitecture(let a): return "Unsupported architecture: \(a)"
            case .unknown(let reason):    return reason
            }
        }
    }
}

// MARK: - Installed model registry

/// Single shared observable registry of every model installed on disk.
/// Every model picker observes this registry. Freshly-downloaded models
/// appear immediately — no relaunch, no screen-leave, no manual rescan.
@MainActor
final class InstalledModelRegistry: ObservableObject {
    static let shared = InstalledModelRegistry()

    @Published private(set) var records: [InstalledModelRecord] = []

    private let storageURL: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Registry", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("installed-models.json")
    }()

    private init() {
        loadFromDisk()
    }

    // MARK: - CRUD

    func register(_ record: InstalledModelRecord) {
        // Upsert: replace any existing record with the same repoID.
        records.removeAll { $0.repoID.caseInsensitiveCompare(record.repoID) == .orderedSame }
        records.append(record)
        saveToDisk()
        Diagnostics.shared.breadcrumb(
            "installed record created · repoID=\(record.repoID) · engine=\(record.engine) · validation=\(record.validationState)",
            category: "registry"
        )
    }

    func remove(repoID: String) {
        records.removeAll { $0.repoID.caseInsensitiveCompare(repoID) == .orderedSame }
        saveToDisk()
    }

    func record(forRepoID repoID: String) -> InstalledModelRecord? {
        records.first { $0.repoID.caseInsensitiveCompare(repoID) == .orderedSame }
    }

    /// Re-scans known disk locations and rebuilds the registry.
    /// Does NOT delete existing records that still validate — it's a
    /// reconciliation, not a wipe-and-rebuild.
    func reconcileWithDisk() {
        let existingByRepo = Dictionary(grouping: records, by: { $0.repoID.lowercased() })
        var updated = records

        // 1. Rescan LLMModels directory (catalog presets)
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let llmRoot = docs.appendingPathComponent("LLMModels")
        reconcileDirectory(llmRoot, existing: existingByRepo, into: &updated, fm: fm)

        // 2. Rescan HFModels directory (custom downloads)
        let hfRoot = docs.appendingPathComponent("HFModels")
        reconcileDirectory(hfRoot, existing: existingByRepo, into: &updated, fm: fm)

        records = updated
        saveToDisk()
        Diagnostics.shared.breadcrumb(
            "registry reconciled · count=\(records.count)",
            category: "registry"
        )
    }

    private func reconcileDirectory(
        _ root: URL, existing: [String: [InstalledModelRecord]],
        into updated: inout [InstalledModelRecord], fm: FileManager
    ) {
        guard let entries = try? fm.contentsOfDirectory(atPath: root.path) else { return }
        for entry in entries {
            let dir = root.appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }

            // Try to recover the repoID from the directory
            let sidecar = dir.appendingPathComponent(".repoID")
            guard let repoID = (try? String(contentsOf: sidecar, encoding: .utf8))?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                  !repoID.isEmpty else { continue }

            // Skip if already registered and directory still exists
            if let existingRecords = existing[repoID.lowercased()],
               existingRecords.contains(where: { $0.validationState == .valid }) {
                continue
            }

            // Validate and create a new record
            if let record = Self.validateDirectory(dir, repoID: repoID) {
                updated.removeAll { $0.repoID.caseInsensitiveCompare(repoID) == .orderedSame }
                updated.append(record)
            }
        }
    }

    /// Validates a model directory and returns an InstalledModelRecord.
    static func validateDirectory(_ dir: URL, repoID: String) -> InstalledModelRecord? {
        let fm = FileManager.default
        let configPath = dir.appendingPathComponent("config.json")

        guard fm.fileExists(atPath: configPath.path) else {
            return nil  // Not a valid MLX model directory
        }

        // Read config for architecture info
        var arch: String? = nil
        var quant: String? = nil
        if let data = try? Data(contentsOf: configPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            arch = (json["architectures"] as? [String])?.first
            quant = json["quantization"] as? String
                ?? json["quant_method"] as? String
                ?? inferQuantization(fromRepoID: repoID)
        }

        // Detect engine
        let engine: ModelRuntime = LocalModelFileValidator.hasValidGGUFTextModel(in: dir)
            ? .llamaCpp
            : .mlx

        // Detect capabilities
        var capabilities = Set<ModelCapability>()
        capabilities.insert(.recommended) // Mark as user-selected since they downloaded it
        if repoID.lowercased().contains("coder") || (arch?.lowercased().contains("coder") ?? false) {
            capabilities.insert(.coder)
        }

        // Validate required files
        let validation: InstalledModelRecord.ValidationState
        let hasTokenizer = fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("tokenizer_config.json").path)
        let hasWeights = Self.hasModelWeights(in: dir, engine: engine)

        switch (hasTokenizer, hasWeights) {
        case (false, _):     validation = .missingTokenizer
        case (_, false):     validation = .missingWeights
        case (true, true):   validation = .valid
        }

        let displayName = repoID.split(separator: "/").last
            .map(String.init) ?? repoID

        return InstalledModelRecord(
            id: UUID(),
            repoID: repoID,
            displayName: displayName,
            localURL: dir,
            engine: engine,
            capabilities: capabilities,
            architecture: arch,
            quantization: quant,
            parameterCount: nil,
            installedAt: Date(),
            validationState: validation,
            downloadBytes: (try? fm.allocatedSizeOfDirectory(at: dir)) ?? 0
        )
    }

    private static func hasModelWeights(in dir: URL, engine: ModelRuntime) -> Bool {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return false }
        switch engine {
        case .mlx:
            let hasSafetensors = names.contains { $0.hasSuffix(".safetensors") }
            let hasWeightIndex = names.contains("model.safetensors.index.json")
            return hasSafetensors || hasWeightIndex
        case .llamaCpp:
            return names.contains { $0.hasSuffix(".gguf") }
        }
    }

    private static func inferQuantization(fromRepoID repoID: String) -> String? {
        let lower = repoID.lowercased()
        if lower.contains("4bit") || lower.contains("-4b") == false && lower.hasSuffix("4b") { return "4bit" }
        if lower.contains("8bit") { return "8bit" }
        if lower.contains("ternary") { return "ternary" }
        if lower.contains("1bit") { return "1bit" }
        if lower.contains("int8") { return "int8" }
        return nil
    }

    // MARK: - Persistence

    private func loadFromDisk() {
        guard let data = try? Data(contentsOf: storageURL),
              let decoded = try? JSONDecoder().decode([InstalledModelRecord].self, from: data) else {
            return
        }
        records = decoded
    }

    private func saveToDisk() {
        guard let data = try? JSONEncoder().encode(records) else { return }
        try? data.write(to: storageURL, options: [.atomic, .completeFileProtectionUnlessOpen])
    }
}
