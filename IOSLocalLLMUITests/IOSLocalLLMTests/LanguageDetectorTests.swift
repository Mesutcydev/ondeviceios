import XCTest
@testable import IOSLocalLLM

// MARK: - LanguageDetectorTests
//
// Pins the per-chunk language detection used by the streaming TTS
// branch to pick a voice that matches each sentence. Assertions are
// kept lenient on the region suffix (`en-US` vs `en-GB`) — the
// detector returns a sensible default region from a hard table, but
// the only thing the TTS router actually cares about is the language
// prefix (`en` vs `tr`).

final class LanguageDetectorTests: XCTestCase {

    private let detector = LanguageDetector.shared

    private func languagePrefix(_ tag: String?) -> String? {
        tag?.split(separator: "-").first.map(String.init)
    }

    // MARK: - Single language

    func test_englishParagraph_taggedEn() {
        let text = "The quick brown fox jumps over the lazy dog near the river bank."
        let tag = detector.detect(text)
        XCTAssertEqual(languagePrefix(tag), "en")
    }

    func test_turkishParagraph_taggedTr() {
        // "Today the weather is very nice, would you like to take a walk?"
        let text = "Bugün hava çok güzel, biraz yürüyüş yapmak ister misin?"
        let tag = detector.detect(text)
        XCTAssertEqual(languagePrefix(tag), "tr")
    }

    // MARK: - Short ambiguous chunks rely on hints

    func test_shortAmbiguousChunk_followsHints() {
        // "Tamam" is Turkish ("OK"). Without a hint the recogniser
        // might pick anything; with a Turkish-only hint it sticks.
        let tag = detector.detect("Tamam.", hints: ["tr"])
        XCTAssertEqual(languagePrefix(tag), "tr")
    }

    func test_shortAmbiguousChunk_englishHint() {
        let tag = detector.detect("OK.", hints: ["en"])
        XCTAssertEqual(languagePrefix(tag), "en")
    }

    // MARK: - Empty input / whitespace

    func test_emptyInput_returnsNil() {
        XCTAssertNil(detector.detect(""))
        XCTAssertNil(detector.detect("   "))
        XCTAssertNil(detector.detect("\n\t  "))
    }

    // MARK: - Mixed paragraph, per-sentence

    func test_mixedParagraphDetectedPerSentence() {
        // Caller is expected to split into sentences first, then run
        // detect on each one — same pattern the streamReply pipeline
        // uses.
        let englishSentence = "Let me explain how this works step by step."
        let turkishSentence = "Şimdi bunu Türkçe olarak da açıklayayım."

        let enTag = detector.detect(englishSentence)
        let trTag = detector.detect(turkishSentence)

        XCTAssertEqual(languagePrefix(enTag), "en")
        XCTAssertEqual(languagePrefix(trTag), "tr")
    }

    // MARK: - BCP-47 region folding

    func test_regionFoldingForCommonLanguages() {
        // The detector folds in a default region for AVSpeech's
        // benefit. Lenient on the region itself — we just assert
        // that the tag contains a `-` so it's BCP-47 shaped.
        if let tag = detector.detect("The quick brown fox jumps high.") {
            XCTAssertTrue(tag.contains("-"),
                          "Expected BCP-47 form like en-US, got \(tag)")
        } else {
            XCTFail("Expected detection on a clearly English sentence")
        }
    }
}
