import XCTest
@testable import IOSLocalLLM

final class KittenRuntimeTests: XCTestCase {
    func testOfficialManifestsArePinnedAndComplete() {
        XCTAssertEqual(Set(KittenManifest.configurations.map(\.id)), Set(KittenVariant.allCases))
        for configuration in KittenManifest.configurations {
            XCTAssertEqual(configuration.revision.count, 40)
            XCTAssertEqual(configuration.sampleRate, 24_000)
            XCTAssertEqual(configuration.artifacts.count, 2)
            XCTAssertTrue(configuration.artifacts.allSatisfy { $0.required })
            XCTAssertTrue(configuration.artifacts.allSatisfy { $0.expectedBytes > 0 })
            XCTAssertTrue(configuration.artifacts.allSatisfy { $0.sha256.count == 64 })
        }
    }

    @MainActor
    func testSelectedKittenVariantSurvivesStoreRestart() {
        let suite = "KittenRuntimeTests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let first = VoiceSettingsStore(defaults: defaults)
        first.selectKittenVariant(.micro08)
        let restored = VoiceSettingsStore(defaults: defaults)
        XCTAssertEqual(restored.selectedKittenVariant, .micro08)
        XCTAssertEqual(restored.selectedEngine, .kittenTTS)
    }

    /// Non-mocked integration gate. This test intentionally skips unless the
    /// exact pinned artifacts have been staged in the app sandbox. Release
    /// acceptance on a physical phone must run it without any skips.
    @MainActor
    func testStagedOfficialVariantsProduceRealAudio() async throws {
        for configuration in KittenManifest.configurations {
            guard FileManager.default.fileExists(atPath: configuration.modelURL.path),
                  FileManager.default.fileExists(atPath: configuration.voiceDataURL.path) else {
                throw XCTSkip("Stage pinned Kitten artifacts before running the inference gate.")
            }
            try await VoiceService.shared.activateAndTestKittenVariant(configuration.id)
            XCTAssertEqual(VoiceService.shared.currentEngineKind, .kittenTTS)
        }
    }
}
