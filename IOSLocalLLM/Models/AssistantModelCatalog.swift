import Foundation

// MARK: - AssistantModel
// One picker entry for a swappable on-device LLM. Each model carries its
// HuggingFace repo ID, a friendly display name, and rough RAM footprint.
// The active selection is stored in AppSettings.assistantModelID.

struct AssistantModel: Identifiable, Hashable, Codable {
    let id: String            // stable key, used as picker tag
    let repoID: String        // HuggingFace repo ID
    let displayName: String   // e.g. "Qwen3-4B Instruct"
    let subtitle: String      // e.g. "4-bit · 2.3 GB"
    let approxRAMBytes: Int64 // working-set estimate for MemoryAdvisor
    /// Free-form tag list shown in the picker (`["code", "fast"]`).
    let tags: [String]
    /// Practical context window in tokens. The KV cache for these on-device
    /// models limits how many input tokens are usable before OOM. Trim the
    /// conversation history to stay under this budget.
    let contextWindowTokens: Int

    /// Approximate bytes fetched for this preset. Low-bit families such as
    /// Bonsai do not follow the catalog's usual `RAM × 0.6` 4-bit heuristic,
    /// so they provide their published package size explicitly.
    var downloadSizeBytes: Int64? = nil

    /// Explicit platform support when the publisher documents a constraint
    /// that RAM-fit inference alone cannot represent. Nil means the normal
    /// live `MemoryAdvisor` result is authoritative.
    var platformCompatibility: ModelPlatformCompatibility? = nil

    // MARK: - Structured capability metadata
    //
    // Typed replacements for the old free-form `tags` heuristics. These
    // power capability pills in the picker, device-aware filtering, and
    // the agent's tool-gating — one source of truth instead of substring
    // matching on display names scattered across the app. All have
    // defaults so the memberwise init keeps every existing call site
    // (downloaded / imported / custom / voice models) compiling unchanged;
    // those synthesize an entry on the fly and don't know capabilities.

    /// Semantic capability markers rendered as colored pills (vision,
    /// thinking, best, recommended, new). Status markers (recommended/
    /// best/new) are curated per preset; capability markers (vision/
    /// thinking) reflect what the model can actually do.
    var capabilities: Set<ModelCapability> = []

    /// Whether the model reliably emits/consumes tool-call JSON. Drives
    /// which models the MacBridge agent will offer Mac tools to. Not a
    /// `ModelCapability` case because it has no catalog pill — surfaced
    /// as a plain "tools" tag where relevant.
    var supportsTools: Bool = false

    /// Inference engine that executes this model. Every assistant preset
    /// is MLX today (the chat tab loads through LLMModelFactory); the
    /// field exists so result cards can label runtime honestly and a
    /// future GGUF text path slots in without a model-shape change.
    var runtime: ModelRuntime = .mlx

    // MARK: - Derived conveniences

    /// True when the model accepts image input (multimodal). Assistant-tab
    /// presets are text-only today; VLMs live in the VLM catalog. Kept here
    /// so capability checks read off one field regardless of model origin.
    var supportsVision: Bool { capabilities.contains(.vision) }

    /// True when the model emits a reasoning trace (Qwen3 /think,
    /// DeepSeek-R1, Qwen3-Thinking, SmolLM3 think mode, Phi-4).
    var supportsThinking: Bool { capabilities.contains(.thinking) }

    /// Capabilities in canonical display order for the picker pill row.
    var displayCapabilities: [ModelCapability] {
        let order: [ModelCapability] = [.recommended, .best, .newRelease,
                                        .vision, .thinking, .tools, .coder,
                                        .fast, .multilingual, .gated]
        // Union the curated capabilities with the ones inferable from this
        // model's identity (tools/coder/fast/multilingual) so the picker
        // surfaces them without each preset re-listing what `supportsTools`
        // and `tags` already imply.
        let effective = capabilities.union(
            ModelCapability.inferred(repoID: repoID, tags: tags, supportsTools: supportsTools)
        )
        return order.filter { effective.contains($0) }
    }

    // MARK: - Chat Template (Feature #6)
    /// Resolves the chat template for this model based on repo ID.
    /// Centralizes the "which template does this model use" decision
    /// so callers don't need to substring-match on repo IDs.
    var chatTemplate: ChatTemplate {
        ChatTemplate.detect(for: repoID)
    }

    /// True when this model uses a ChatML-compatible template (Qwen family).
    var usesChatML: Bool {
        chatTemplate.format == "chatml" || chatTemplate.format == "qwen35"
            || chatTemplate.format == "generic"
    }
}

// MARK: - Catalog

enum AssistantModelCatalog {

    /// Built-in models known to work with MLX `LLMModelFactory` on iOS.
    /// Order = priority (first entry is the default).
    ///
    /// Curation rule: every preset must be a **text** model whose
    /// architecture is loadable by `LLMModelFactory` (the chat tab's
    /// runtime). Multimodal models (Qwen3-VL, Gemma 3 vision, SmolVLM2)
    /// run through a different factory and live in the VLM catalog
    /// (`ModelDownloadCenter.buildCatalog`), not here. New entries below
    /// share an architecture already proven by an existing preset
    /// (Qwen3, Qwen2.5, Llama 3.x) so a refresh can't silently 404 or
    /// hit an unsupported-arch load failure.
    static let presets: [AssistantModel] = [
        // ── Qwen3 4B 2507 refresh ─────────────────────────────────────────
        // The "2507" Instruct/Thinking refresh is a large quality jump over
        // the original Qwen3-4B and is now published by mlx-community (the
        // old comment about 2507 404-ing is stale — verified HTTP 200). Same
        // Qwen3 architecture as the model the app already loads, so no new
        // runtime risk. Instruct is the default chat brain.
        AssistantModel(
            id: "qwen3-4b-2507",
            repoID: "mlx-community/Qwen3-4B-Instruct-2507-4bit",
            displayName: "Qwen3-4B 2507",
            subtitle: "4-bit · 2.3 GB · refreshed flagship",
            approxRAMBytes: 3_800_000_000,
            tags: ["code", "chat", "default"],
            contextWindowTokens: 32768,
            capabilities: [.recommended, .best, .newRelease],
            supportsTools: true
        ),
        // Thinking-2507 — emits a reasoning trace; best for hard reasoning
        // / math / multi-step code on capable devices.
        AssistantModel(
            id: "qwen3-4b-thinking-2507",
            repoID: "mlx-community/Qwen3-4B-Thinking-2507-4bit",
            displayName: "Qwen3-4B Thinking 2507",
            subtitle: "4-bit · 2.3 GB · reasoning trace",
            approxRAMBytes: 3_800_000_000,
            tags: ["reason", "thinking"],
            contextWindowTokens: 32768,
            capabilities: [.newRelease, .thinking],
            supportsTools: true
        ),
        // 8-bit sibling of the 2507 flagship. Same weights, higher fidelity,
        // ~2× the RAM — surfaces only on .max devices (RAM-budget filtered
        // in the picker). Its real job: give the A/B compare screen a
        // same-model quantization contrast (4-bit vs 8-bit) so users can
        // measure the quality↔speed↔memory trade-off on their own hardware.
        AssistantModel(
            id: "qwen3-4b-2507-8bit",
            repoID: "mlx-community/Qwen3-4B-Instruct-2507-8bit",
            displayName: "Qwen3-4B 2507 (8-bit)",
            subtitle: "8-bit · 4.3 GB · higher fidelity",
            approxRAMBytes: 6_200_000_000,
            tags: ["quality"],
            contextWindowTokens: 32768,
            capabilities: [.best],
            supportsTools: true
        ),
        // Original Qwen3-4B — kept for back-compat with users who already
        // downloaded it and for benchmark-history continuity.
        AssistantModel(
            id: "qwen3-4b",
            repoID: "mlx-community/Qwen3-4B-4bit",
            displayName: "Qwen3-4B",
            subtitle: "4-bit · 2.3 GB · thinking mode",
            approxRAMBytes: 3_800_000_000,
            tags: ["code", "chat", "thinking"],
            contextWindowTokens: 32768,
            capabilities: [.thinking],
            supportsTools: true
        ),
        // Qwen3-1.7B — thinking mode, fits 4 GB devices, fastest Qwen3.
        AssistantModel(
            id: "qwen3-1.7b",
            repoID: "mlx-community/Qwen3-1.7B-4bit",
            displayName: "Qwen3-1.7B",
            subtitle: "4-bit · 1.0 GB · thinking · fast",
            approxRAMBytes: 1_500_000_000,
            tags: ["fast", "thinking"],
            contextWindowTokens: 32768,
            capabilities: [.thinking],
            supportsTools: true
        ),
        // Qwen3-8B — highest quality Qwen3, requires ≥ 8 GB device.
        AssistantModel(
            id: "qwen3-8b",
            repoID: "mlx-community/Qwen3-8B-4bit",
            displayName: "Qwen3-8B",
            subtitle: "4-bit · 4.8 GB · thinking · best quality",
            approxRAMBytes: 6_500_000_000,
            tags: ["quality", "thinking"],
            contextWindowTokens: 32768,
            capabilities: [.best, .thinking],
            supportsTools: true
        ),
        // Ornith 1.0 9B — Qwen3.5 text architecture, tuned for agentic coding.
        // This used to be reachable only as an installed community model. That
        // made it disappear whenever registry discovery had not finished yet
        // (and removed it entirely from the downloadable catalog). Keep it as
        // a first-class preset on the high-memory tier. Its published long
        // context is recorded here; the capability profile configures a real
        // 64K rotating 4-bit KV cache for the on-device runtime.
        AssistantModel(
            id: "ornith-1.0-9b-4bit",
            repoID: "mlx-community/Ornith-1.0-9B-4bit",
            displayName: "Ornith 1.0 9B",
            subtitle: "4-bit · ~6.0 GB · agentic coding · high-memory",
            approxRAMBytes: 7_250_000_000,
            tags: ["code", "agent", "quality"],
            contextWindowTokens: 262_144,
            downloadSizeBytes: 5_980_000_000,
            platformCompatibility: .highMemoryMobileAndMac,
            capabilities: [.best, .newRelease],
            supportsTools: true
        ),
        // ── PrismML Bonsai low-bit family ─────────────────────────────────
        // Bonsai keeps Qwen's architecture and chat vocabulary while replacing
        // the dense weights with binary (1-bit) or ternary (2-bit packed)
        // weights. The 1-bit variants require PrismML's MLX kernels; the app's
        // mlx-swift dependency points at that compatible fork. Bonsai 27B is
        // based on Qwen3.5's hybrid-attention architecture, which is available
        // in mlx-swift-lm 3.31.4. Smaller variants remain Qwen3 text models.
        AssistantModel(
            id: "bonsai-27b-1bit",
            repoID: "prism-ml/Bonsai-27B-mlx-1bit",
            displayName: "Bonsai 27B",
            subtitle: "1-bit · 5.2 GB download · high-memory iOS + Mac",
            // Text-only peak with the bounded 2K/4-bit KV profile. Lens uses
            // its separate vision admission estimate (~8.7 GB including load
            // and camera headroom), so this value must not include image
            // activations or Assistant becomes unusable on 12 GB iPhones.
            approxRAMBytes: 5_500_000_000,
            tags: ["reason", "thinking", "code"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 5_160_000_000,
            platformCompatibility: .highMemoryMobileAndMac,
            capabilities: [.best, .newRelease, .vision, .thinking, .multilingual],
            supportsTools: true
        ),
        AssistantModel(
            id: "bonsai-27b-ternary",
            repoID: "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
            displayName: "Ternary Bonsai 27B",
            subtitle: "2-bit · 8.5 GB download · Mac only",
            approxRAMBytes: 9_200_000_000,
            tags: ["reason", "thinking", "code", "quality"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 8_525_000_000,
            platformCompatibility: .macOnly,
            capabilities: [.best, .newRelease, .vision, .thinking, .multilingual],
            supportsTools: true
        ),
        AssistantModel(
            id: "bonsai-8b-ternary",
            repoID: "prism-ml/Ternary-Bonsai-8B-mlx-2bit",
            displayName: "Ternary Bonsai 8B",
            subtitle: "2-bit · 2.3 GB · quality low-bit",
            approxRAMBytes: 3_300_000_000,
            tags: ["reason", "thinking", "code"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 2_320_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual],
            supportsTools: true
        ),
        AssistantModel(
            id: "bonsai-8b-1bit",
            repoID: "prism-ml/Bonsai-8B-mlx-1bit",
            displayName: "Bonsai 8B",
            subtitle: "1-bit · 1.3 GB · compact reasoning",
            approxRAMBytes: 2_200_000_000,
            tags: ["reason", "thinking", "code", "fast"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 1_300_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual],
            supportsTools: true
        ),
        AssistantModel(
            id: "bonsai-4b-ternary",
            repoID: "prism-ml/Ternary-Bonsai-4B-mlx-2bit",
            displayName: "Ternary Bonsai 4B",
            subtitle: "2-bit · 1.1 GB · balanced",
            approxRAMBytes: 1_850_000_000,
            tags: ["reason", "thinking", "fast"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 1_150_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual]
        ),
        AssistantModel(
            id: "bonsai-4b-1bit",
            repoID: "prism-ml/Bonsai-4B-mlx-1bit",
            displayName: "Bonsai 4B",
            subtitle: "1-bit · 650 MB · fast reasoning",
            approxRAMBytes: 1_350_000_000,
            tags: ["reason", "thinking", "fast", "light"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 650_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual]
        ),
        AssistantModel(
            id: "bonsai-1.7b-ternary",
            repoID: "prism-ml/Ternary-Bonsai-1.7B-mlx-2bit",
            displayName: "Ternary Bonsai 1.7B",
            subtitle: "2-bit · 510 MB · ultralight",
            approxRAMBytes: 1_000_000_000,
            tags: ["thinking", "fast", "light"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 510_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual]
        ),
        AssistantModel(
            id: "bonsai-1.7b-1bit",
            repoID: "prism-ml/Bonsai-1.7B-mlx-1bit",
            displayName: "Bonsai 1.7B",
            subtitle: "1-bit · 290 MB · smallest Bonsai",
            approxRAMBytes: 800_000_000,
            tags: ["thinking", "fast", "light"],
            contextWindowTokens: 32768,
            downloadSizeBytes: 290_000_000,
            platformCompatibility: .mobileAndMac,
            capabilities: [.newRelease, .thinking, .multilingual]
        ),
        // Qwen2.5-Coder 1.5B — code-tuned, stable, works on all devices.
        AssistantModel(
            id: "qwen2.5-coder-1.5b",
            repoID: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
            displayName: "Qwen2.5-Coder 1.5B",
            subtitle: "4-bit · 900 MB · code-tuned · stable",
            approxRAMBytes: 1_400_000_000,
            tags: ["code", "fast"],
            contextWindowTokens: 8192,
            capabilities: [],
            supportsTools: true
        ),
        // ── Lightweight / fast tier ──────────────────────────────────────
        // Sub-2B models for the snappiest decode and the lowest-RAM devices.
        // All share an architecture already proven by a larger preset
        // (Qwen2.5 / Qwen3), so no new load risk. Repo IDs verified HTTP 200.
        //
        // Qwen3-0.6B — the smallest model that still emits a thinking trace;
        // fastest first token in the roster.
        AssistantModel(
            id: "qwen3-0.6b",
            repoID: "mlx-community/Qwen3-0.6B-4bit",
            displayName: "Qwen3-0.6B",
            subtitle: "4-bit · ~400 MB · thinking · fastest",
            approxRAMBytes: 1_000_000_000,
            tags: ["fast", "light", "thinking"],
            contextWindowTokens: 32768,
            capabilities: [.thinking],
            supportsTools: true
        ),
        // Qwen2.5-0.5B — the lightest text model in the catalog; the floor
        // option for the oldest / most memory-constrained iPhones.
        AssistantModel(
            id: "qwen2.5-0.5b",
            repoID: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            displayName: "Qwen2.5 0.5B",
            subtitle: "4-bit · ~300 MB · ultralight · fastest",
            approxRAMBytes: 800_000_000,
            tags: ["fast", "light"],
            contextWindowTokens: 8192,
            capabilities: [],
            supportsTools: true
        ),
        // Qwen2.5-1.5B (general) — a fast, general-chat counterpart to the
        // code-tuned 1.5B above for non-coding prompts.
        AssistantModel(
            id: "qwen2.5-1.5b",
            repoID: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            displayName: "Qwen2.5 1.5B",
            subtitle: "4-bit · 900 MB · general · fast",
            approxRAMBytes: 1_400_000_000,
            tags: ["fast", "general"],
            contextWindowTokens: 8192,
            capabilities: [],
            supportsTools: true
        ),
        AssistantModel(
            id: "qwen2.5-7b",
            repoID: "mlx-community/Qwen2.5-7B-Instruct-4bit",
            displayName: "Qwen2.5 7B",
            subtitle: "4-bit · 4.2 GB · general chat",
            approxRAMBytes: 5_500_000_000,
            tags: ["chat", "general"],
            contextWindowTokens: 8192,
            capabilities: [],
            supportsTools: true
        ),
        AssistantModel(
            id: "llama-3.2-3b",
            repoID: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            displayName: "Llama 3.2 3B",
            subtitle: "4-bit · 1.8 GB",
            approxRAMBytes: 3_000_000_000,
            tags: ["chat"],
            contextWindowTokens: 8192,
            capabilities: [],
            supportsTools: true
        ),
        // Llama 3.2 1B — smallest well-supported text model; the floor
        // option for 3-4 GB devices and the fastest decode in the roster.
        // Same Llama architecture as the 3B above, so no new load risk.
        AssistantModel(
            id: "llama-3.2-1b",
            repoID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
            displayName: "Llama 3.2 1B",
            subtitle: "4-bit · 700 MB · fastest · low-tier",
            approxRAMBytes: 1_200_000_000,
            tags: ["fast", "light"],
            contextWindowTokens: 8192,
            capabilities: []
        ),
        AssistantModel(
            id: "phi-3.5-mini",
            repoID: "mlx-community/Phi-3.5-mini-instruct-4bit",
            displayName: "Phi-3.5 Mini",
            subtitle: "4-bit · 2.2 GB · reasoning",
            approxRAMBytes: 3_500_000_000,
            tags: ["reason"],
            contextWindowTokens: 8192,
            capabilities: []
        ),
        AssistantModel(
            id: "gemma-2-2b",
            repoID: "mlx-community/gemma-2-2b-it-4bit",
            displayName: "Gemma 2 2B",
            subtitle: "4-bit · 1.5 GB · light",
            approxRAMBytes: 2_300_000_000,
            tags: ["chat", "fast"],
            contextWindowTokens: 4096,
            capabilities: []
        ),
    ]

    /// Returns the AssistantModel matching the stored settings, or the first
    /// preset if the stored ID is unknown.
    ///
    /// Stored IDs come in four flavors:
    ///   • preset key                  → matches a `presets` entry
    ///   • "downloaded:<repoID>"       → HF Search download registered in ModelDownloadCenter
    ///   • "imported:<repoID>"         → local Files import (repoID starts with `local/`)
    ///   • "custom:<repoID>"           → "load this repo" typed by the user
    ///
    /// Without this resolution, picking a non-preset model used to silently
    /// fall back to `presets[0]` inside `load()`, which is exactly the bug
    /// where selecting DeepSeek activated Qwen2.5.
    @MainActor
    static func currentSelection() -> AssistantModel {
#if CORE_AI_SERVER_APP
        return AssistantModel(
            id: CoreAIModelStore.defaultModelID,
            repoID: "coreai/qwen3-0.6b",
            displayName: "Core AI Qwen 0.6B",
            subtitle: "Core AI · downloadable .aimodel",
            approxRAMBytes: 1_000_000_000,
            tags: ["chat", "vision", "core-ai"],
            contextWindowTokens: 32_768,
            capabilities: [.recommended, .vision, .multilingual],
            supportsTools: true
        )
#else
        let stored = AppSettings.shared.assistantModelID
        return selection(forStoredID: stored) ?? presets[0]
#endif
    }

    /// Resolves a persisted preset/download/import/custom selection without
    /// falling back. Conversation restoration uses this to reopen a thread
    /// with its own model while leaving the user's default untouched.
    @MainActor
    static func selection(forStoredID stored: String) -> AssistantModel? {
        if let preset = presets.first(where: { $0.id == stored }) { return preset }
        return resolveNonPreset(id: stored)
    }

    static func model(forID id: String) -> AssistantModel? {
        presets.first { $0.id == id }
    }

    // MARK: - Non-preset resolution

    /// Rebuilds an AssistantModel for a `downloaded:` / `imported:` / `custom:`
    /// id by either looking up the matching entry in ModelDownloadCenter (for
    /// downloaded/imported) or synthesizing one from the bare repoID (custom).
    @MainActor
    private static func resolveNonPreset(id stored: String) -> AssistantModel? {
        LocalModelRegistry
            .descriptor(forStoredAssistantID: stored, catalog: ModelDownloadCenter.shared.models)?
            .assistantModel
    }
}
