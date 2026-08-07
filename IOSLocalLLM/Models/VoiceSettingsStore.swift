import Combine
import Foundation

enum VoiceEngineAvailabilityState: Equatable {
    case loading
    case ready
    case failed
}

/// The authoritative, persistent source of truth for Voice Mode choices.
/// Availability is intentionally separate: a temporary load failure may route
/// audio through Apple TTS, but must not erase the user's preferred engine.
@MainActor
final class VoiceSettingsStore: ObservableObject {
    static let shared = VoiceSettingsStore()

    static let selectedEngineKey = "voice.selectedTTSEngine"
    static let selectedVoiceKey = "voice.selectedVoiceID"
    static let selectedKokoroVoiceKey = "voice.selectedKokoroVoiceID"
    static let selectedKittenVoiceKey = "voice.selectedKittenVoiceID"
    static let selectedKittenVariantKey = "voice.selectedKittenVariantID"
    static let selectedAppleVoiceKey = "voice.selectedAppleVoiceID"
    static let renderingModeKey = "voice.renderingMode"
    static let legacyEngineKey = "voiceEngine"
    static let legacyVoiceKey = "voiceID"

    @Published private(set) var selectedEngine: VoiceEngineKind
    @Published private(set) var selectedVoiceID: String
    @Published private(set) var selectedKittenVariant: KittenVariant
    @Published private(set) var availabilityState: VoiceEngineAvailabilityState = .loading
    @Published private(set) var renderingMode: VoiceRenderingMode = .automatic

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults

        let engineRaw = defaults.string(forKey: Self.selectedEngineKey)
            ?? defaults.string(forKey: Self.legacyEngineKey)
        let restoredEngine = VoiceEngineKind(rawValue: engineRaw ?? "") ?? .appleSystem
        selectedEngine = restoredEngine
        selectedVoiceID = defaults.string(forKey: Self.voiceKey(for: restoredEngine))
            ?? defaults.string(forKey: Self.selectedVoiceKey)
            ?? defaults.string(forKey: Self.legacyVoiceKey)
            ?? ""
        selectedKittenVariant = KittenVariant(
            rawValue: defaults.string(forKey: Self.selectedKittenVariantKey) ?? ""
        ) ?? .nano08Int8
        renderingMode = VoiceRenderingMode(
            rawValue: defaults.string(forKey: Self.renderingModeKey) ?? ""
        ) ?? .automatic
        // Write stable namespaced keys once so legacy users migrate without
        // losing either the engine or its model-specific voice selection.
        defaults.set(selectedEngine.rawValue, forKey: Self.selectedEngineKey)
        defaults.set(selectedVoiceID, forKey: Self.selectedVoiceKey)
        defaults.set(renderingMode.rawValue, forKey: Self.renderingModeKey)
        print("[VoiceSettings] restored engine=\(selectedEngine.rawValue)")
    }

    func setRenderingMode(_ mode: VoiceRenderingMode) {
        renderingMode = mode
        defaults.set(mode.rawValue, forKey: Self.renderingModeKey)
    }

    func selectEngine(_ engine: VoiceEngineKind) {
        selectedEngine = engine
        if let engineVoice = defaults.string(forKey: Self.voiceKey(for: engine)) {
            selectedVoiceID = engineVoice
            defaults.set(engineVoice, forKey: Self.selectedVoiceKey)
            defaults.set(engineVoice, forKey: Self.legacyVoiceKey)
        }
        defaults.set(engine.rawValue, forKey: Self.selectedEngineKey)
        defaults.set(engine.rawValue, forKey: Self.legacyEngineKey)
        print("[VoiceSettings] user selected engine=\(engine.rawValue)")
    }

    func selectVoice(_ voiceID: String) {
        selectedVoiceID = voiceID
        defaults.set(voiceID, forKey: Self.selectedVoiceKey)
        defaults.set(voiceID, forKey: Self.voiceKey(for: selectedEngine))
        defaults.set(voiceID, forKey: Self.legacyVoiceKey)
    }

    func selectKittenVariant(_ variant: KittenVariant) {
        selectedKittenVariant = variant
        defaults.set(variant.rawValue, forKey: Self.selectedKittenVariantKey)
        selectEngine(.kittenTTS)
    }

    func beginAvailabilityCheck() {
        availabilityState = .loading
    }

    /// Reconciles the restored preference only after disk discovery finishes.
    /// A failed engine initialization is not conclusive unavailability: callers
    /// may temporarily synthesize with Apple while preserving the preference.
    func completeAvailabilityCheck(availableEngines: Set<VoiceEngineKind>) {
        availabilityState = .ready
        print("[VoiceSettings] availability complete engines=\(availableEngines.map(\.rawValue).sorted())")
        guard !availableEngines.contains(selectedEngine) else {
            print("[VoiceSettings] active engine resolved=\(selectedEngine.rawValue)")
            return
        }

        let unavailable = selectedEngine
        selectEngine(.appleSystem)
        print("[VoiceSettings] fallback engine=apple_system reason=\(unavailable.rawValue)_unavailable")
    }

    func markAvailabilityCheckFailed() {
        availabilityState = .failed
        print("[VoiceSettings] availability failed; preserving engine=\(selectedEngine.rawValue)")
    }

    private static func voiceKey(for engine: VoiceEngineKind) -> String {
        switch engine {
        case .appleSystem: return selectedAppleVoiceKey
        case .kittenTTS:   return selectedKittenVoiceKey
        case .kokoro:      return selectedKokoroVoiceKey
        }
    }
}
