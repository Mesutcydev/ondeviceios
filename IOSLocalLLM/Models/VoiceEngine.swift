import AVFoundation

// MARK: - VoiceEngineKind
// Identifies which synthesis backend a VoiceOption or VoiceService operation uses.

enum VoiceEngineKind: String, CaseIterable, Identifiable, Codable {
    case appleSystem = "apple_system"   // AVSpeechSynthesizer — always available
    case kittenTTS   = "kitten_tts"     // KittenTTS CoreML — requires download
    case kokoro      = "kokoro"         // Kokoro-82M CoreML — experimental

    var id: String { rawValue }

    /// Engines the user can pick from the settings UI. Kokoro lands
    /// here once its CoreML bundle is on disk — the catalog now ships
    /// a real download for `aufklarer/Kokoro-82M-CoreML`, so the old
    /// "hide it until there's a working in-app download" caveat no
    /// longer applies. Until the user actually downloads it, Kokoro
    /// stays off the picker so they don't get a dead-end "model not
    /// found" failure when they tap a row that can't be loaded.
    static var userSelectableCases: [VoiceEngineKind] {
        allCases.filter { kind in
            switch kind {
            case .appleSystem, .kittenTTS, .kokoro: return true
            }
        }
    }

    var displayName: String {
        switch self {
        case .appleSystem: return "Apple System Voice"
        case .kittenTTS:   return "KittenTTS — Tiny Neural"
        case .kokoro:      return "Kokoro — Neural"
        }
    }

    var shortName: String {
        switch self {
        case .appleSystem: return "System"
        case .kittenTTS:   return "KittenTTS"
        case .kokoro:      return "Kokoro"
        }
    }

    /// Whether this engine requires a model download before first use.
    var requiresDownload: Bool {
        self == .kittenTTS || self == .kokoro
    }

    /// Whether this engine is labelled experimental in the UI.
    ///
    /// KittenTTS now ships with a CMUdict-backed phonemizer and the
    /// correct upstream vocab + framing tokens (see
    /// PhonemizerEN.swift / KittenTTSService.swift). It still depends
    /// on a downloaded model and uses an LTS fallback for OOV words,
    /// so it's not bit-for-bit espeak quality — but it's well past
    /// the "experimental, broken" stage that the previous BasicG2P
    /// implementation produced. Kokoro is in the same boat; both
    /// share the StyleTTS2 vocabulary. Apple System Voice remains
    /// the recommended default because Premium voices on iOS 18+
    /// are state-of-the-art and need no extra download.
    var isExperimental: Bool {
        // Kept as a flag in case future UI wants to gate a feature,
        // but no engine is "broken-experimental" anymore.
        false
    }

    /// Approximate sample rate of generated audio.
    var sampleRate: Double {
        switch self {
        case .appleSystem: return 22050
        case .kittenTTS:   return 24000
        case .kokoro:      return 24000
        }
    }
}

// MARK: - VoiceOption
// A single selectable voice within an engine.

struct VoiceOption: Identifiable, Hashable, Codable {
    /// Stable unique ID (used as @AppStorage key).
    let id: String
    /// Display name shown in pickers.
    let name: String
    /// The engine this voice belongs to.
    let engineKind: VoiceEngineKind
    /// IETF locale tag, e.g. "en-US".
    let locale: String
    /// Short description or style label.
    let description: String?

    // MARK: Equatable / Hashable
    static func == (lhs: VoiceOption, rhs: VoiceOption) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

// MARK: - VoiceSynthesisSettings
// Per-request audio parameters passed into synthesize().

struct VoiceSynthesisSettings {
    /// Playback speed multiplier. Range 0.5–2.0. Default 1.0.
    var speed: Float  = 1.0
    /// Pitch multiplier. Range 0.5–2.0. Default 1.0.
    var pitch: Float  = 1.0
    /// Output volume. Range 0.0–1.0. Default 1.0.
    var volume: Float = 1.0
}

// MARK: - LocalVoiceEngine
// Protocol every voice backend must implement.
// All methods are MainActor-isolated so state mutations are safe.

@MainActor
protocol LocalVoiceEngine: AnyObject {
    var kind: VoiceEngineKind { get }
    var availableVoices: [VoiceOption] { get }
    var isReady: Bool { get }
    var modelState: VoiceModelState { get }

    /// Load Core ML models or any other heavy setup (no-op for system engine).
    func load() async

    /// Release model memory.
    func unload()

    /// Synthesize `text` with the given voice and settings.
    ///
    /// - Returns: A `VoiceSynthesisResult`. When `pcmBuffer` is nil the engine
    ///   handled playback internally (e.g. AVSpeechSynthesizer).
    /// - Throws: `VoiceError` if synthesis cannot proceed.
    func synthesize(
        text: String,
        voice: VoiceOption,
        settings: VoiceSynthesisSettings
    ) async throws -> VoiceSynthesisResult
}

// MARK: - VoiceError

enum VoiceError: LocalizedError {
    case engineNotReady(String)
    case modelNotFound(String)
    case synthesisFailedWithError(String)
    case noVoiceSelected
    case audioBufferCreationFailed

    var errorDescription: String? {
        switch self {
        case .engineNotReady(let msg):           return "Voice engine not ready: \(msg)"
        case .modelNotFound(let msg):            return "Voice model not found: \(msg)"
        case .synthesisFailedWithError(let msg): return "Synthesis failed: \(msg)"
        case .noVoiceSelected:                   return "No voice selected."
        case .audioBufferCreationFailed:         return "Could not create audio buffer."
        }
    }
}
