import XCTest
@testable import IOSLocalLLM

// MARK: - TextChunkerSanitizerTests
//
// Covers the sanitizer rules in TextChunker.preprocess that the voice
// pipeline depends on. The chunker has existing rules (markdown
// strip, code-block summary, paragraph collapse) — these tests pin the
// new "insert period before paragraph break when previous line lacks
// terminal punctuation" rule, and a few of the pre-existing rules that
// are load-bearing for the streaming pipeline.

final class TextChunkerSanitizerTests: XCTestCase {

    // MARK: - Period-before-paragraph rule (new)

    func test_paragraphBreak_withMissingPeriod_insertsOne() {
        // Under-punctuating model output. Both sentences fit under
        // the 250-char packing cap so the chunker packs them into
        // one chunk — that's correct existing behavior. What matters
        // is the INSERTED period after "First sentence" so the
        // downstream NLTokenizer sees two sentences rather than one
        // run-on.
        let input = "First sentence\n\nSecond sentence."
        let chunks = TextChunker.chunks(from: input)
        let joined = chunks.joined(separator: " ")
        XCTAssertTrue(joined.contains("First sentence."),
            "Expected an inserted period after 'First sentence', got \(joined)")
        XCTAssertTrue(joined.contains("Second sentence."),
            "Second sentence's existing period must survive, got \(joined)")
    }

    func test_paragraphBreak_withExistingPunctuation_leavesAlone() {
        // Already terminated — the rule must NOT double-up.
        let input = "First sentence.\n\nSecond sentence."
        let chunks = TextChunker.chunks(from: input)
        let joined = chunks.joined(separator: " ")
        XCTAssertFalse(joined.contains(".."),
            "Rule must be idempotent — found doubled period in \(joined)")
        XCTAssertTrue(joined.contains("First sentence."))
        XCTAssertTrue(joined.contains("Second sentence."))
    }

    func test_paragraphBreak_withQuestionMark_leavesAlone() {
        let input = "Did it work?\n\nLet me check."
        let chunks = TextChunker.chunks(from: input)
        let joined = chunks.joined(separator: " ")
        XCTAssertFalse(joined.contains("?."))
    }

    // MARK: - Existing rules that the pipeline depends on

    func test_strippsBoldMarkdown() {
        let chunks = TextChunker.chunks(from: "This is **bold** text.")
        let joined = chunks.joined(separator: " ")
        XCTAssertFalse(joined.contains("**"))
        XCTAssertTrue(joined.contains("bold"))
    }

    func test_strippsLinks_keepsLabel() {
        let chunks = TextChunker.chunks(from: "See [the docs](https://example.com) for more.")
        let joined = chunks.joined(separator: " ")
        XCTAssertTrue(joined.contains("the docs"))
        XCTAssertFalse(joined.contains("https"))
        XCTAssertFalse(joined.contains("]("))
    }

    func test_preservesEmoji() {
        // Spec is explicit — AVSpeech handles emoji sensibly and they
        // carry tone. Sanitizer must NOT strip them.
        let chunks = TextChunker.chunks(from: "Great work! 🎉 Keep it up.")
        let joined = chunks.joined(separator: " ")
        XCTAssertTrue(joined.contains("🎉"))
    }
}
