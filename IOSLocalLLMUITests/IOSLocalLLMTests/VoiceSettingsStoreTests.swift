import XCTest
@testable import IOSLocalLLM

@MainActor
final class VoiceSettingsStoreTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suiteName: String!

    override func setUp() {
        super.setUp()
        suiteName = "VoiceSettingsStoreTests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        suiteName = nil
        super.tearDown()
    }

    func testFirstLaunchDefaultsToApple() {
        XCTAssertEqual(VoiceSettingsStore(defaults: defaults).selectedEngine, .appleSystem)
    }

    func testSelectingKokoroPersistsStableIdentifier() {
        let store = VoiceSettingsStore(defaults: defaults)
        store.selectEngine(.kokoro)
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedEngineKey), "kokoro")
    }

    func testRelaunchWithInstalledKokoroRestoresKokoro() {
        VoiceSettingsStore(defaults: defaults).selectEngine(.kokoro)
        let relaunched = VoiceSettingsStore(defaults: defaults)
        relaunched.completeAvailabilityCheck(availableEngines: [.appleSystem, .kokoro])
        XCTAssertEqual(relaunched.selectedEngine, .kokoro)
    }

    func testLoadingDoesNotResetRestoredKokoro() {
        defaults.set("kokoro", forKey: VoiceSettingsStore.selectedEngineKey)
        let store = VoiceSettingsStore(defaults: defaults)
        store.beginAvailabilityCheck()
        XCTAssertEqual(store.availabilityState, .loading)
        XCTAssertEqual(store.selectedEngine, .kokoro)
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedEngineKey), "kokoro")
    }

    func testRemovingKokoroCausesControlledPersistedFallback() {
        defaults.set("kokoro", forKey: VoiceSettingsStore.selectedEngineKey)
        let store = VoiceSettingsStore(defaults: defaults)
        store.completeAvailabilityCheck(availableEngines: [.appleSystem])
        XCTAssertEqual(store.selectedEngine, .appleSystem)
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedEngineKey), "apple_system")
    }

    func testCorruptedPersistedEngineFallsBackSafely() {
        defaults.set("localized Kokoro label", forKey: VoiceSettingsStore.selectedEngineKey)
        XCTAssertEqual(VoiceSettingsStore(defaults: defaults).selectedEngine, .appleSystem)
    }

    func testSwitchingBackToApplePersists() {
        let store = VoiceSettingsStore(defaults: defaults)
        store.selectEngine(.kokoro)
        store.selectEngine(.appleSystem)
        XCTAssertEqual(VoiceSettingsStore(defaults: defaults).selectedEngine, .appleSystem)
    }

    func testRecreatingViewModelBackingStoreDoesNotResetEngine() {
        VoiceSettingsStore(defaults: defaults).selectEngine(.kokoro)
        XCTAssertEqual(VoiceSettingsStore(defaults: defaults).selectedEngine, .kokoro)
    }

    func testAvailabilityRefreshDoesNotOverwriteAvailableSelection() {
        let store = VoiceSettingsStore(defaults: defaults)
        store.selectEngine(.kokoro)
        store.completeAvailabilityCheck(availableEngines: [.appleSystem, .kittenTTS, .kokoro])
        XCTAssertEqual(store.selectedEngine, .kokoro)
    }

    func testSharedStoreIsOneAuthoritativeInstance() {
        XCTAssertTrue(VoiceSettingsStore.shared === VoiceSettingsStore.shared)
    }

    func testKokoroVoiceVariantIsRestoredAfterEngineRoundTrip() {
        let store = VoiceSettingsStore(defaults: defaults)
        store.selectEngine(.kokoro)
        store.selectVoice("kokoro:af_heart")
        store.selectEngine(.appleSystem)
        store.selectVoice("apple:premium")
        store.selectEngine(.kokoro)
        XCTAssertEqual(store.selectedVoiceID, "kokoro:af_heart")
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedKokoroVoiceKey), "kokoro:af_heart")
    }

    func testFailedInitializationPreservesPreference() {
        defaults.set("kokoro", forKey: VoiceSettingsStore.selectedEngineKey)
        let store = VoiceSettingsStore(defaults: defaults)
        store.markAvailabilityCheckFailed()
        XCTAssertEqual(store.availabilityState, .failed)
        XCTAssertEqual(store.selectedEngine, .kokoro)
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedEngineKey), "kokoro")
    }

    func testLegacyKeysMigrateAndFullLifecycleKeepsKokoroActive() {
        defaults.set("kokoro", forKey: VoiceSettingsStore.legacyEngineKey)
        defaults.set("kokoro:af_heart", forKey: VoiceSettingsStore.legacyVoiceKey)

        let appState = VoiceSettingsStore(defaults: defaults)
        appState.beginAvailabilityCheck()
        appState.completeAvailabilityCheck(availableEngines: [.appleSystem, .kokoro])

        XCTAssertEqual(appState.selectedEngine, .kokoro)
        XCTAssertEqual(appState.selectedVoiceID, "kokoro:af_heart")
        XCTAssertEqual(defaults.string(forKey: VoiceSettingsStore.selectedEngineKey), "kokoro")
    }

    func testDownloadCompletionEquivalentRefreshDoesNotSelectAnEngine() {
        let store = VoiceSettingsStore(defaults: defaults)
        store.selectEngine(.appleSystem)
        store.completeAvailabilityCheck(availableEngines: [.appleSystem, .kokoro])
        XCTAssertEqual(store.selectedEngine, .appleSystem)
    }
}
