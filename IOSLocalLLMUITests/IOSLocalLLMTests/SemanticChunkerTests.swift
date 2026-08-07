import XCTest
@testable import IOSLocalLLM

// MARK: - SemanticChunkerTests
//
// Pins the boundary detector that was lifted from
// `VoiceConversationService.sentenceBoundaryIndex(in:after:)` into
// `SemanticChunker`. The `.sentence` strategy MUST match the
// historical behavior bit-for-bit; the `.clause` and `.minWords`
// strategies are new for session 3 (Lens descriptive narration) and
// pinned here so future edits don't drift.

final class SemanticChunkerTests: XCTestCase {

    // MARK: - Sentence strategy

    func test_sentence_belowMinChars_noFlush() {
        // 11-char sentence → no flush (min is 12 by default).
        let c = SemanticChunker(strategy: .sentence)
        let chunks = c.append("Short one. ")
        XCTAssertTrue(chunks.isEmpty,
                      "11-char slice should not flush at default min")
    }

    func test_sentence_atMinChars_flushes() {
        let c = SemanticChunker(strategy: .sentence)
        let chunks = c.append("Hello there. ")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "Hello there. ")
        XCTAssertEqual(chunks.first?.isFinal, false)
    }

    func test_sentence_tailTerminatorWithoutWhitespace_defersToFinish() {
        // `.` at end of buffer with nothing following — the historical
        // cutter explicitly defers this case so the in-flight token
        // doesn't pre-ship before generation completes.
        let c = SemanticChunker(strategy: .sentence)
        let mid = c.append("This is a complete sentence.")
        XCTAssertTrue(mid.isEmpty)
        let tail = c.finish()
        XCTAssertEqual(tail?.text, "This is a complete sentence.")
        XCTAssertEqual(tail?.isFinal, true)
    }

    func test_sentence_paragraphBreakFlushesEvenShort() {
        // `\n\n` IS a boundary, but a preceding sentence terminator
        // followed by the first `\n` ALSO satisfies the "terminator +
        // whitespace" rule, so the cut lands one char after the `!`
        // (not after the full `\n\n` pair). Historical behavior.
        let c = SemanticChunker(strategy: .sentence)
        let chunks = c.append("First line here!\n\nSecond")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "First line here!\n")
    }

    func test_sentence_consecutiveTerminatorsCoalesce() {
        // `...!?` followed by space is one boundary, not three. Pad
        // with enough leading words to clear the 12-char minimum
        // (the coalesce works regardless of length — the min just
        // gates emission).
        let c = SemanticChunker(strategy: .sentence)
        let chunks = c.append("Hold on wait...!? what now")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "Hold on wait...!? ")
    }

    func test_sentence_streamingTokenByToken_concatenatesIdentically() {
        // Same text fed token-by-token produces the same OR MORE
        // chunks than the batch path (streaming emits each boundary
        // as it appears; batch sees all boundaries at once and emits
        // up to the LAST one). The slices concatenate to the same
        // text either way, MODULO the trailing sentence that may
        // stay below the 12-char min in streaming mode — that's a
        // historical behavior we preserved verbatim.
        let input = "Hello world. This is sentence two. And three more words here. "

        let batch = SemanticChunker(strategy: .sentence)
        let batchChunks = batch.append(input)
        let batchText = batchChunks.map(\.text).joined()

        let stream = SemanticChunker(strategy: .sentence)
        var streamed: [SemanticChunker.Chunk] = []
        for ch in input {
            streamed.append(contentsOf: stream.append(String(ch)))
        }
        let streamText = streamed.map(\.text).joined()

        XCTAssertEqual(streamText, batchText,
            "All three sentences clear the 12-char minimum here, so both modes emit them all")
    }

    func test_sentence_reset_clearsBuffer() {
        // "New start emerges." is 18 chars — clears the 12-char min
        // so the post-reset emission produces a chunk.
        let c = SemanticChunker(strategy: .sentence)
        _ = c.append("First sentence here. ")
        c.reset()
        let chunks = c.append("New start emerges. ")
        XCTAssertEqual(chunks.first?.text, "New start emerges. ")
    }

    func test_sentence_finishOnEmpty_returnsNil() {
        let c = SemanticChunker(strategy: .sentence)
        XCTAssertNil(c.finish())
    }

    // MARK: - Multibyte / Turkish

    func test_sentence_turkishCharacters_dontSplitGraphemes() {
        // Turkish has letters that occupy multiple UTF-8 bytes — `ş`
        // (2 bytes), `ğ` (2 bytes). The chunker uses Character (i.e.
        // grapheme) offsets, so the emitted slice must include the
        // full glyph, not a leading byte.
        let c = SemanticChunker(strategy: .sentence)
        let chunks = c.append("Bugün hava güzel. ")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "Bugün hava güzel. ")
    }

    // MARK: - Clause strategy

    func test_clause_commaBreak() {
        let c = SemanticChunker(strategy: .clause)
        let chunks = c.append("First clause here, second clause follows")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "First clause here, ")
    }

    func test_clause_semicolonBreak() {
        let c = SemanticChunker(strategy: .clause)
        let chunks = c.append("Note the difference; consider this also")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "Note the difference; ")
    }

    func test_clause_alsoFlushesOnSentenceTerminator() {
        // .clause should still fire on `.` so .clause is a SUPERSET
        // of .sentence behavior.
        let c = SemanticChunker(strategy: .clause)
        let chunks = c.append("Sentence ends here. Next one")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "Sentence ends here. ")
    }

    // MARK: - minWords strategy

    func test_minWords_belowThreshold_noFlush() {
        let c = SemanticChunker(strategy: .minWords(5))
        let chunks = c.append("One two three four ")
        // 4 words — not enough.
        XCTAssertTrue(chunks.isEmpty)
    }

    func test_minWords_atThreshold_flushesOnWhitespace() {
        let c = SemanticChunker(strategy: .minWords(5))
        let chunks = c.append("One two three four five six")
        // Boundary fires AFTER the whitespace following the 5th word.
        // "One two three four five " is 24 chars, well above 12-char min.
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.text, "One two three four five ")
    }

    func test_minWords_midWordDoesNotFlush() {
        // Reading a partial 6th word without trailing space → no
        // mid-stream flush. The trailing "extra" stays buffered; the
        // finish() flush yields it as isFinal=true.
        let c = SemanticChunker(strategy: .minWords(5))
        let chunks = c.append("One two three four five extra")
        XCTAssertEqual(chunks.first?.text, "One two three four five ")
        let tail = c.finish()
        XCTAssertEqual(tail?.text, "extra")
        XCTAssertEqual(tail?.isFinal, true)
    }

    func test_minWords_bufferEndingMidWord_noMidStreamFlush() {
        // No internal whitespace after the 5th word — nothing to
        // pivot on, so the mid-stream chunker emits nothing. The
        // tail can still be flushed via finish().
        let c = SemanticChunker(strategy: .minWords(5))
        XCTAssertTrue(c.append("One two three four five-no-space").isEmpty)
        let tail = c.finish()
        XCTAssertEqual(tail?.text, "One two three four five-no-space")
        XCTAssertEqual(tail?.isFinal, true)
    }
}
