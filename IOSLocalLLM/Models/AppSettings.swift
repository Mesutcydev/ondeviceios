import SwiftUI
import Combine

// MARK: - AppSettings
// Single source of truth for all user-configurable values.
// @AppStorage keys are stable — don't rename without a migration.

final class AppSettings: ObservableObject {
    static let shared = AppSettings()

    // Camera / YOLO
    @AppStorage("yoloConfidenceThreshold") var yoloConfidenceThreshold: Double = 0.75
    @AppStorage("autoCaptureEnabled")      var autoCaptureEnabled: Bool = true
    @AppStorage("maxDetections")           var maxDetections: Int = 3

    // Models
    @AppStorage("fastVLMEnabled")          var fastVLMEnabled: Bool = true

    // MARK: - Smart / adjustable model handling
    /// When ON (default), model loads predicted to exceed this device's
    /// memory are refused up-front to prevent OOM crashes. When OFF, the
    /// conservative physical-RAM heuristic is relaxed (the app still blocks
    /// loads that exceed the hard per-process ceiling, since those are a
    /// near-certain crash). Lets power users try borderline models.
    @AppStorage("strictMemoryGate")        var strictMemoryGate: Bool = true
    /// Edge / developer mode for model suggestions and loading. OFF by default:
    /// a normal user only ever sees models that comfortably fit this device
    /// (full load-spike reserve kept), so they can't hit an OOM crash or an
    /// incompatible pick. When ON, "tight" models that sit right at the memory
    /// ceiling are also surfaced. GGUF can use its paging-aware edge path;
    /// MLX keeps the fixed load reserve because its weights materialize eagerly.
    @AppStorage("showEdgeModels")          var showEdgeModels: Bool = false
    /// Experimental low-memory path for large local models. Imported GGUF
    /// weights stay mmap-backed with CPU execution; MLX uses a hard allocator
    /// ceiling, no reusable load cache, quantized/bounded KV, and small prefill
    /// chunks. MLX still materializes its full weights — only GGUF truly pages
    /// weights from storage. The legacy key preserves existing user choices.
    @AppStorage("largeGGUFPagingEnabled")  var largeModelLowMemoryEnabled: Bool = false
    /// Max total seconds a model load may run before the watchdog auto-cancels
    /// it (reusing the same safe path as the Stop button). 0 disables the
    /// watchdog. Default 10 minutes — long enough for a multi-GB cold download
    /// on a slow connection, short enough to recover from a truly stuck load.
    @AppStorage("modelLoadTimeoutSeconds") var modelLoadTimeoutSeconds: Int = 600

    // Assistant
    @AppStorage("assistantThinking")       var assistantThinking: Bool = false  // Qwen3 /think mode
    @AppStorage("assistantMaxTokens")      var assistantMaxTokens: Int = 2048
    @AppStorage("assistantTemperature")    var assistantTemperature: Double = 0.6

    // MARK: - Full Sampler Control (Feature #1)
    @AppStorage("assistantTopP")           var assistantTopP: Double = 0.95
    @AppStorage("assistantTopK")           var assistantTopK: Int = 50
    @AppStorage("assistantMinP")           var assistantMinP: Double = 0.0
    @AppStorage("assistantRepetitionPenalty") var assistantRepetitionPenalty: Double = 1.05
    @AppStorage("assistantFrequencyPenalty")  var assistantFrequencyPenalty: Double = 0.0
    @AppStorage("assistantPresencePenalty")   var assistantPresencePenalty: Double = 0.0
    /// Fixed seed for deterministic generation (0 = random).
    /// Backed by an `Int` because `@AppStorage` has no `UInt64` overload;
    /// the public `assistantSeed` keeps its `UInt64` type so call sites
    /// (CodingAssistantService) are unaffected.
    @AppStorage("assistantSeed")           private var assistantSeedRaw: Int = 0
    var assistantSeed: UInt64 {
        get { UInt64(max(0, assistantSeedRaw)) }
        set { assistantSeedRaw = Int(min(newValue, UInt64(Int.max))) }
    }
    /// Active sampler preset ID (empty = use individual knobs).
    @AppStorage("samplerPresetID")         var samplerPresetID: String = ""

    // MARK: - JSON Mode (Feature #2)
    /// When ON, injects JSON-output instructions into the system prompt
    /// and validates the output for well-formed JSON.
    @AppStorage("jsonModeEnabled")         var jsonModeEnabled: Bool = false
    /// Optional JSON schema to constrain output (informal — MLX doesn't
    /// support grammar-constrained generation yet, so this is prompt-based).
    @AppStorage("jsonSchemaHint")          var jsonSchemaHint: String = ""

    // MARK: - Logprobs (Feature #3)
    /// When ON, the generate() pipeline collects per-token log probabilities
    /// and attaches them to the assistant message.
    @AppStorage("logprobsEnabled")         var logprobsEnabled: Bool = false
    /// Number of top alternative tokens to return per position.
    @AppStorage("logprobsTopK")            var logprobsTopK: Int = 3

    // MARK: - Context Window (Feature #5)
    /// Show the context-window usage bar above the chat.
    @AppStorage("showContextWindowBar")    var showContextWindowBar: Bool = true
    /// Warn when input exceeds this fraction of the context window.
    @AppStorage("contextWindowWarnFraction") var contextWindowWarnFraction: Double = 0.75
    /// User-selected default assistant model ID. New conversations and cold
    /// launches start here; an individual conversation may temporarily use a
    /// different model without overwriting this preference.
    @AppStorage("assistantModelID")        var assistantModelID: String = "qwen2.5-coder-1.5b"
    /// Versioned consent for sending assistant conversation content to Apple
    /// Private Cloud Compute. A disclosure wording change increments the
    /// required version and asks again before the next cloud request.
    @AppStorage("applePCCPrivacyConsentVersion")
    var applePCCPrivacyConsentVersion: Int = 0
    /// Stored as a raw string so future reasoning levels can fall back safely.
    @AppStorage("applePCCReasoningLevel")
    private var applePCCReasoningLevelRaw: String = ApplePCCReasoningLevel.automatic.rawValue
    var applePCCReasoningLevel: ApplePCCReasoningLevel {
        get { ApplePCCReasoningLevel(rawValue: applePCCReasoningLevelRaw) ?? .automatic }
        set { applePCCReasoningLevelRaw = newValue.rawValue }
    }
    var hasCurrentApplePCCPrivacyConsent: Bool {
        ApplePrivateCloud.hasCurrentPrivacyConsent(
            version: applePCCPrivacyConsentVersion
        )
    }
    /// True once the user explicitly chose a default model. Until then the
    /// device-tier advisor may re-tune the default to fit the device class.
    @AppStorage("hasPickedAssistantModel") var hasPickedAssistantModel: Bool = false
    /// True once the user explicitly changed the response length (max tokens).
    /// Until then DeviceTierAdvisor sets a tier-appropriate default on first launch.
    @AppStorage("hasPickedResponseLength") var hasPickedResponseLength: Bool = false
    /// Optional override used by voice conversation mode. Empty string means
    /// "use whatever the chat tab is using". Set to a model id (preset or
    /// `downloaded:…` / `imported:…` / `custom:…`) to pick a different one.
    @AppStorage("voiceConversationModelID") var voiceConversationModelID: String = ""
    /// Enable tool / function calling (calculator, datetime, unit convert).
    @AppStorage("toolsEnabled")            var toolsEnabled: Bool = true
    /// Sync conversation history to user's private CloudKit database.
    @AppStorage("iCloudSyncEnabled")       var iCloudSyncEnabled: Bool = false

    /// Send the user's HF token (stored in Keychain via HFTokenStore)
    /// with downloads + search. Token presence is necessary but not
    /// sufficient — this toggle lets users temporarily disable auth
    /// to debug 401s without clearing the token. Default ON because
    /// once a user has bothered to enter a token, they want it used.
    @AppStorage("useHFToken")              var useHFToken: Bool = true

    /// Restrict model downloads to Wi-Fi (non-expensive) connections.
    /// Default OFF to preserve the existing allow-cellular behavior.
    /// Read by BackgroundDownloadCoordinator at session creation AND per
    /// task at enqueue time (the background session config is immutable
    /// once created).
    @AppStorage("wifiOnlyDownloads")       var wifiOnlyDownloads: Bool = false

    /// Allow pairing with a Mac that did NOT advertise a TLS cert fingerprint,
    /// which means the pairing handshake (nonce + bearer token) travels over
    /// plaintext http on the local network. Default OFF — a LAN attacker could
    /// otherwise sniff the bearer. Only legacy Mac builds need this; the
    /// pairing flow refuses an insecure Mac unless the user opts in here.
    @AppStorage("allowInsecureBridgePairing") var allowInsecureBridgePairing: Bool = false

    // Local OpenAI / Ollama compatibility server. The access key is stored
    // separately in Keychain; only these non-secret preferences use defaults.
    @AppStorage("localAPIEnabled") var localAPIEnabled: Bool = true
    @AppStorage("localAPIPort") var localAPIPort: Int = 11434
    @AppStorage("localAPIKeepScreenAwake") var localAPIKeepScreenAwake: Bool = true
    @AppStorage("localAPIAutoLoadModel") var localAPIAutoLoadModel: Bool = false
    /// Expose OpenAI/Anthropic/Ollama tool-call envelopes. When disabled,
    /// incoming tool definitions are ignored and the model behaves as a text
    /// completion endpoint.
    @AppStorage("localAPIToolCallingEnabled") var localAPIToolCallingEnabled: Bool = true
    /// Allow Qwen-style hidden reasoning during API generation. Reasoning
    /// blocks are still removed from the protocol response for compatibility.
    @AppStorage("localAPIReasoningEnabled") var localAPIReasoningEnabled: Bool = false
    /// Permit one response to contain multiple independent tool calls.
    @AppStorage("localAPIParallelToolCallsEnabled") var localAPIParallelToolCallsEnabled: Bool = true
    /// User-selected ceiling for parallel tool calls. The active model
    /// capability profile may apply a lower safety limit.
    @AppStorage("localAPIParallelToolCallsLimit") var localAPIParallelToolCallsLimit: Int = 2
    /// Validate model-produced arguments against the request's JSON schemas
    /// before returning a call to Hermes or another client.
    @AppStorage("localAPIStrictToolSchemasEnabled") var localAPIStrictToolSchemasEnabled: Bool = true

    // FastVLM pipeline
    // OCR fallback defaults OFF: the lens starts in visual (describe) mode, so
    // falling back to raw text extraction is off by default. Users who want it
    // can re-enable it in Settings.
    @AppStorage("useOCRFallback")          var useOCRFallback: Bool = false
    @AppStorage("fastvlmMaxTokens")        var fastvlmMaxTokens: Int = 512
    @AppStorage("fastvlmTemperature")      var fastvlmTemperature: Double = 0.2
    @AppStorage("showDebugModelShapes")    var showDebugModelShapes: Bool = false
    /// HuggingFace repo ID for FastVLM MLX weights. User-editable so they
    /// can switch to a working mirror if the default 404s/401s.
    @AppStorage("fastVLMRepoID")           var fastVLMRepoID: String = "apple/FastVLM-0.5B-MLX"

    /// Active visual model the camera tab uses. Empty string = the built-in
    /// FastVLM path (Core ML encoder + MLX decoder). Any other value is a
    /// HuggingFace repo id of an MLX-format VLM the user has downloaded
    /// (Qwen2-VL, SmolVLM, Paligemma, Gemma 3 vision, etc.) and is routed
    /// through MLXVisionService instead.
    @AppStorage("cameraVisualModelID")     var cameraVisualModelID: String = ""

    /// True once the user explicitly picked a camera VLM from the picker.
    /// Until then `DeviceTierAdvisor.applyDefaultVisualModelIfNeeded()` is
    /// allowed to set a tier-appropriate SmolVLM on each cold launch.
    @AppStorage("hasPickedCameraVisualModel") var hasPickedCameraVisualModel: Bool = false

    /// Auto-caption follow-up interval (ms) while the live caption overlay is
    /// up in visual mode. Clamped to [6000, 30000] when read by the loop.
    @AppStorage("smolVLMIntervalMS")       var smolVLMIntervalMS: Int = 6000

    /// Active LensPromptPreset (raw value). Shapes the prompt sent to the
    /// VLM in visual mode. Defaults to `.describe`.
    @AppStorage("lensPromptPresetID")      var lensPromptPresetID: String = "describe"

    /// Optional free-form question for the live Lens. When non-empty it takes
    /// precedence over `lensPromptPresetID` for prompt-capable visual models.
    /// Keeping it in AppStorage means a user can briefly leave Lens to manage
    /// a model and return without losing the question they were composing.
    @AppStorage("lensCustomPrompt")        var lensCustomPrompt: String = ""

    // Thermal protection
    /// When false, the assistant UI hides the "device hot" pill and stops
    /// auto-clamping max-tokens / refusing generation based on thermal state.
    /// Memory-warning protection stays on regardless.
    @AppStorage("thermalWarningsEnabled")  var thermalWarningsEnabled: Bool = true

    // Analysis mode — lens opens in visual (describe) mode by default.
    @AppStorage("analysisMode")            var analysisMode: String = AnalysisMode.visual.rawValue

    // Onboarding
    @AppStorage("hasSeenOnboarding")       var hasSeenOnboarding: Bool = false

    // UI
    @AppStorage("showFPSCounter")          var showFPSCounter: Bool = true
    @AppStorage("hapticsEnabled")          var hapticsEnabled: Bool = true
    /// "system", "light", "dark", or "oled" — the palette switches accordingly.
    /// "oled" is the Plum Dusk glass direction (see KoduTheme.oled).
    @AppStorage("appearance")              var appearance: String = "system"
    /// Selected brand-accent palette; the rawValue of `KoduTheme.KoduAccent`.
    @AppStorage("themeAccent")             var themeAccent: String = KoduTheme.appAccent.rawValue

    /// SwiftUI color scheme for the chosen appearance. Both "dark" and "oled"
    /// resolve to `.dark`; only "light" is light. "system" resolves to nil so
    /// the window follows the device setting. Use this instead of an inline
    /// `appearance == "dark" ? .dark : .light` so OLED doesn't fall through to
    /// light.
    var resolvedColorScheme: ColorScheme? {
        switch appearance {
        case "system": return nil
        case "light":  return .light
        default:       return .dark
        }
    }
    /// UI language override: "system" (follow device), "en", or "tr".
    /// Live-switchable via LocalizationService — see Settings → INTERFACE
    /// → language picker.
    @AppStorage("uiLanguage")              var uiLanguage: String = "system"
    /// Lens debug overlay: shows a thumbnail of the EXACT image MLX
    /// receives (post-orientation, post-resize) so we can verify the
    /// preprocessing pipeline visually instead of guessing why captions
    /// are off. Off by default; toggle in Settings → INTERFACE.
    @AppStorage("showModelInputDebug")     var showModelInputDebug: Bool = false

    // Voice / TTS preferences are intentionally not part of this target.
    /// Playback speed multiplier (0.5–2.0).
    @AppStorage("voiceSpeed")              var voiceSpeed: Double = 1.0
    /// Pitch multiplier (0.5–2.0).
    @AppStorage("voicePitch")              var voicePitch: Double = 1.0
    /// Output volume (0.0–1.0).
    @AppStorage("voiceVolume")             var voiceVolume: Double = 1.0
    /// Automatically read analysis results when they appear.
    @AppStorage("voiceAutoRead")           var voiceAutoRead: Bool = false
    /// Read code blocks aloud (vs. skipping them).
    @AppStorage("voiceReadCodeBlocks")     var voiceReadCodeBlocks: Bool = false
    /// Voice-conversation mode: whether the assistant speaks its reply
    /// aloud. When OFF the conversation still works hands-free on the
    /// input side (mic → STT → LLM → text reply on screen) but TTS is
    /// skipped — useful in quiet rooms or when the user just wants
    /// dictation-driven chat without audio feedback. Toggle exposed
    /// directly in the voice-conversation controls bar.
    @AppStorage("voiceAnswerEnabled")      var voiceAnswerEnabled: Bool = true
    /// Speech-to-text provider for the chat composer's single-shot
    /// dictation. "system" routes through Apple's SFSpeechRecognizer
    /// (cloud-fallback on some locales). "whisper" routes through the
    /// on-device whisper.cpp model — fully offline, multilingual, but
    /// requires the user to download a model first. The continuous
    /// voice-conversation mode is unaffected and always uses the
    /// system recognizer (Whisper is not a streaming model).
    @AppStorage("sttProvider")             var sttProvider: String = "system"
    /// Optional locale identifier to pin SFSpeechRecognizer to a specific
    /// language (e.g. "tr-TR", "es-ES"). Empty string means "auto" — use
    /// the device locale with a fallback chain. Honoured by both
    /// single-shot dictation and the continuous voice-conversation
    /// recogniser. Users on multilingual devices set this when their
    /// spoken language differs from the system language.
    @AppStorage("sttLocaleOverride")       var sttLocaleOverride: String = ""
    /// Lens streaming TTS: when ON, narrates the live caption clause-by-
    /// clause as the VLM emits tokens (rather than the historical
    /// "speak full caption once on completion" behavior). Off by
    /// default — opt-in for low-vision users. The frame-rate guard in
    /// `LensVoiceNarrator` suppresses re-narration when the new
    /// frame's caption substantially overlaps the previous one, so
    /// turning this on doesn't cause the ~35fps re-inference cycle to
    /// stutter the audio.
    @AppStorage("voiceSpeakInLens")        var voiceSpeakInLens: Bool = false

    /// Voice-conversation mode: enable AVAudioInputNode voice processing
    /// (AEC + noise suppression + AGC). Makes barge-in possible —
    /// without it, the mic picks up the assistant's own TTS through
    /// the loudspeaker. On a non-trivial set of devices/routes the
    /// `vpio` AudioUnit returns continuous `render err: -1` errors
    /// against `.spokenAudio` mode, which produces unmistakable
    /// zombie-voice output. Default OFF for clean out-of-box audio;
    /// users who want barge-in can flip it ON in Voice settings and
    /// will hit the failing path themselves rather than the reverse.
    @AppStorage("voiceProcessingEnabled")  var voiceProcessingEnabled: Bool = false

    // MARK: - Voice Improvements (Feature #10)
    /// Whether to allow the user to interrupt the assistant mid-speech (barge-in).
    @AppStorage("bargeInEnabled")         var bargeInEnabled: Bool = true
    /// Auto-save raw transcript so user can tap-to-edit before sending.
    @AppStorage("transcriptEditingEnabled") var transcriptEditingEnabled: Bool = true
    /// Voice profile per persona: map of personaID → voiceID for persona-specific voices.
    @AppStorage("personaVoiceProfileJSON")  var personaVoiceProfileJSON: String = ""

    // MARK: - Benchmark Expansion (Feature #12)
    /// Whether to record throughput-over-time data during benchmarks.
    @AppStorage("benchmarkRecordThroughput") var benchmarkRecordThroughput: Bool = true
    /// Whether to export benchmark results as shareable CSV.
    @AppStorage("benchmarkExportCSV")       var benchmarkExportCSV: Bool = false
}
