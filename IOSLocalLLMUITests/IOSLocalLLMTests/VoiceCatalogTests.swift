import XCTest
@testable import IOSLocalLLM

@MainActor
final class VoiceCatalogTests: XCTestCase {
    private var store: VoiceCatalogStore!

    override func setUp() {
        super.setUp()
        store = VoiceCatalogStore(bundle: .main)
    }

    func testBundledProductionCatalogContainsEveryRequiredID() throws {
        let required: Set<String> = [
            "tts.apple.system", "tts.kokoro.82m.official",
            "tts.kitten.micro.0_8", "tts.kitten.nano.0_8.int8",
            "tts.kitten.nano.0_8.fp32", "tts.kitten.mini.0_8",
            "tts.piper.family", "tts.melotts.family",
            "tts.parler.mini.v1_1", "tts.parler.mini.multilingual.v1_1",
            "tts.outetts.family", "tts.qwen.family", "tts.f5.family",
            "tts.chatterbox.family", "tts.xtts.v2", "tts.styletts2.family",
            "stt.whisper.tiny", "stt.whisper.tiny.en", "stt.whisper.base",
            "stt.whisper.base.en", "stt.whisper.small", "stt.whisper.small.en",
            "stt.moonshine.base", "stt.moonshine.streaming-tiny",
            "stt.moonshine.streaming-medium"
        ]
        XCTAssertEqual(store.loadState, .loaded)
        XCTAssertTrue(required.isSubset(of: Set(store.allEntries.map(\.id))))
        XCTAssertEqual(store.allEntries.count, 25)
    }

    func testExactJSONIsBundledAndUsesSchemaTwo() throws {
        let url = try XCTUnwrap(Bundle.main.url(forResource: "VoiceModelCatalog", withExtension: "json"))
        let document = try JSONDecoder().decode(VoiceCatalogDocument.self, from: Data(contentsOf: url))
        XCTAssertEqual(document.schemaVersion, 2)
        XCTAssertEqual(document.entries.count, 25)
    }

    func testDefaultPresentationRetainsCatalogOnlyAndUnavailableEntries() {
        let presented = store.entries()
        XCTAssertTrue(presented.contains { $0.supportLevel == .catalogOnly })
        XCTAssertTrue(presented.contains { !$0.runtimeAvailable })
        XCTAssertEqual(presented.count, store.allEntries.count)
    }

    func testMacOnlyEntriesRemainVisibleOnIPhone() {
        let ids = Set(store.entries().filter { $0.supportLevel == .macOnly }.map(\.id))
        XCTAssertEqual(ids, ["tts.f5.family", "tts.styletts2.family"])
    }

    func testOnlyRuntimeBackedEntriesEnableDownload() {
        let enabled = store.entries().filter(\.isDownloadEnabled)
        XCTAssertEqual(Set(enabled.map(\.id)), [
            "tts.kokoro.82m.official", "tts.kitten.micro.0_8",
            "tts.kitten.nano.0_8.int8",
            "tts.kitten.mini.0_8", "stt.whisper.base", "stt.whisper.base.en"
        ])
        XCTAssertFalse(store.entries().contains { !$0.runtimeAvailable && $0.isDownloadEnabled })
    }

    func testCompleteFallbackCannotRegressToLegacyFourCards() {
        XCTAssertEqual(VoiceCatalogStore.completeFallback.count, 25)
        XCTAssertTrue(VoiceCatalogStore.completeFallback.contains { $0.id == "tts.piper.family" })
        XCTAssertTrue(VoiceCatalogStore.completeFallback.contains { $0.id == "stt.moonshine.base" })
    }

    func testSearchFindsEveryAcceptanceFamily() {
        for query in ["Piper", "Parler", "Moonshine", "MeloTTS", "F5-TTS"] {
            XCTAssertFalse(store.entries(query: query).isEmpty, "Missing search result for \(query)")
        }
    }

    func testDisabledCardsStillHaveDetailAndSourceData() {
        let disabled = store.entries().filter { !$0.isDownloadEnabled && $0.id != "tts.apple.system" }
        XCTAssertFalse(disabled.isEmpty)
        XCTAssertTrue(disabled.allSatisfy { !$0.summary.isEmpty && $0.sourceURL != nil })
    }

    func testCatalogTaskFiltersDoNotDiscardUnsupportedModels() {
        let tts = store.entries(task: .textToSpeech)
        let stt = store.entries(task: .speechRecognition)
        XCTAssertEqual(tts.count, 16)
        XCTAssertEqual(stt.count, 9)
        XCTAssertTrue(tts.contains { $0.id == "tts.melotts.family" })
        XCTAssertTrue(stt.contains { $0.id == "stt.moonshine.streaming-medium" })
    }

    func testSchemaVersionInvalidatesLegacyV1() {
        XCTAssertEqual(VoiceCatalogStore.schemaVersion, 2)
        XCTAssertNotEqual(VoiceCatalogStore.schemaVersion, 1)
    }
}
