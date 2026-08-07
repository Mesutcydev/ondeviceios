import XCTest
@testable import IOSLocalLLM

// MARK: - ReasoningStripperTests
//
// These tests pin the streaming state machine that keeps `<think>…</think>`
// blocks out of the TTS branch. The user-visible UI text bypasses this
// filter entirely; the filter only sits in front of `speakStream` and
// the one-shot `speak()` paths.
//
// A note on `feed() + flush()`: a streaming stripper buffers chars that
// COULD become a tag, so the same input split into different chunks
// must produce identical concatenated output. Several tests assert
// this directly by replaying a fixture in different chunk granularities.

final class ReasoningStripperTests: XCTestCase {

    // MARK: - Same chunk

    func test_thinkBlockEntirelyInOneChunk_stripped() {
        let s = ReasoningStripper()
        let out = s.feed("<think>secret</think>visible") + s.flush()
        XCTAssertEqual(out, "visible")
    }

    func test_thinkBlockSurroundedByText_stripped() {
        let s = ReasoningStripper()
        let out = s.feed("before <think>x</think>after") + s.flush()
        XCTAssertEqual(out, "before after")
    }

    // MARK: - Split across chunks

    func test_openTagSplitAcrossChunks_stripped() {
        let s = ReasoningStripper()
        var out = s.feed("<thi")
        out += s.feed("nk>hidden</think>shown")
        out += s.flush()
        XCTAssertEqual(out, "shown")
    }

    func test_closeTagSplitAcrossChunks_stripped() {
        let s = ReasoningStripper()
        var out = s.feed("<think>hidden</thi")
        out += s.feed("nk>shown")
        out += s.flush()
        XCTAssertEqual(out, "shown")
    }

    func test_charByCharFeed_matchesBatch() {
        let input = "alpha <think>internal monologue</think> beta gamma"
        let batch = ReasoningStripper.strip(input)

        let s = ReasoningStripper()
        var streamed = ""
        for ch in input { streamed += s.feed(String(ch)) }
        streamed += s.flush()

        XCTAssertEqual(streamed, batch)
        XCTAssertEqual(streamed, "alpha  beta gamma")
    }

    // MARK: - Multiple blocks

    func test_multipleThinkBlocks_allStripped() {
        let s = ReasoningStripper()
        let out = s.feed("a<think>1</think>b<think>2</think>c") + s.flush()
        XCTAssertEqual(out, "abc")
    }

    // MARK: - Unclosed block

    func test_unclosedBlockAtStreamEnd_dropped() {
        let s = ReasoningStripper()
        var out = s.feed("hello <think>partial reasoning")
        out += s.flush()
        XCTAssertEqual(out, "hello ")
    }

    // MARK: - Passthrough

    func test_textWithNoThinkBlocks_unchanged() {
        let s = ReasoningStripper()
        let input = "Plain text with no tags, just a period."
        let out = s.feed(input) + s.flush()
        XCTAssertEqual(out, input)
    }

    func test_textWithLeftAngleButNoThink_unchanged() {
        let s = ReasoningStripper()
        let input = "x < y and 3 < 4"
        let out = s.feed(input) + s.flush()
        XCTAssertEqual(out, input)
    }

    // MARK: - Recovery from broken open prefix

    func test_partialOpenPrefixFollowedByRealOpen_correctlyStrips() {
        // "<th" looks like the start of "<think>" but the next char
        // breaks the match. The buffered "<th" must emit as visible
        // text and the FRESH "<think>" must enter the block. KMP-
        // style failure-function path covers this.
        let s = ReasoningStripper()
        let out = s.feed("<th<think>hidden</think>tail") + s.flush()
        XCTAssertEqual(out, "<thtail")
    }

    // MARK: - Configurable tags

    func test_customTagsReasoning_stripped() {
        let s = ReasoningStripper(openTag: "<reasoning>", closeTag: "</reasoning>")
        let out = s.feed("ok <reasoning>private</reasoning> done") + s.flush()
        XCTAssertEqual(out, "ok  done")
    }

    func test_customTagsThinking_bracketStyle_stripped() {
        let s = ReasoningStripper(openTag: "[THINKING]", closeTag: "[/THINKING]")
        let out = s.feed("a[THINKING]b[/THINKING]c") + s.flush()
        XCTAssertEqual(out, "ac")
    }

    func test_oneShotStrip_matchesFeedFlush() {
        let input = "head <think>middle</think> tail"
        let oneShot = ReasoningStripper.strip(input)
        let streamed = {
            let s = ReasoningStripper()
            return s.feed(input) + s.flush()
        }()
        XCTAssertEqual(oneShot, streamed)
    }
}
