import XCTest
@testable import IOSLocalLLM

// MARK: - LensVoiceNarratorTests
//
// The narrator's state machine is tangled with `LensInferenceLoop`'s
// hook callbacks and the real audio session, neither of which mocks
// cleanly without re-architecting both for DI — out of scope for
// this session. What WE can pin in isolation is the token-overlap
// math that drives the frame-rate guard, and the threshold/prefix
// constants the audit pushed back on the original spec for. Those
// are the bits that, if wrong, would resurrect zombie-voice on the
// Lens tab.

final class LensVoiceNarratorTests: XCTestCase {

    // MARK: - Identity / extremes

    func test_overlap_identicalStrings_isOne() {
        let s = "a person sitting at a wooden desk with a laptop"
        XCTAssertEqual(LensVoiceNarrator.overlap(s, s), 1.0, accuracy: 0.0001)
    }

    func test_overlap_completelyDifferentStrings_isZero() {
        let a = "kitchen utensils on the counter"
        let b = "dog running through park grass"
        XCTAssertEqual(LensVoiceNarrator.overlap(a, b), 0.0, accuracy: 0.0001)
    }

    func test_overlap_empty_isZero() {
        XCTAssertEqual(LensVoiceNarrator.overlap("", ""), 0.0)
        XCTAssertEqual(LensVoiceNarrator.overlap("hello world", ""), 0.0)
        XCTAssertEqual(LensVoiceNarrator.overlap("", "hello world"), 0.0)
    }

    // MARK: - Real-world cases — scene stays the same

    func test_overlap_sameSceneSlightChange_exceedsSkipThreshold() {
        // Captions of the same scene from consecutive frames typically
        // share most content words. Both should be > 0.30 → skip.
        let a = "a person at a desk"
        let b = "the person at the desk with a laptop"
        let score = LensVoiceNarrator.overlap(a, b)
        XCTAssertGreaterThanOrEqual(
            score,
            LensVoiceNarrator.sceneChangeOverlapThreshold,
            "Same scene with mild rewording must skip, got overlap \(score)"
        )
    }

    func test_overlap_sameSceneTokenAddition_exceedsSkipThreshold() {
        // Real Lens behavior: each frame the VLM phrases the caption
        // slightly differently. Identical core nouns + most adjectives
        // means the overlap should remain above threshold.
        let a = "a wooden desk with a silver laptop and a coffee cup"
        let b = "a wooden desk with a silver laptop and a black coffee mug"
        let score = LensVoiceNarrator.overlap(a, b)
        XCTAssertGreaterThanOrEqual(
            score,
            LensVoiceNarrator.sceneChangeOverlapThreshold
        )
    }

    // MARK: - Real-world cases — scene actually changed

    func test_overlap_realSceneChange_belowSkipThreshold() {
        let a = "a kitchen with utensils on the counter"
        let b = "a dog running in a park"
        let score = LensVoiceNarrator.overlap(a, b)
        XCTAssertLessThan(
            score,
            LensVoiceNarrator.sceneChangeOverlapThreshold,
            "Distinct scenes must drop below threshold, got overlap \(score)"
        )
    }

    func test_overlap_caseAndPunctuationIgnored() {
        // The tokenizer lowercases and splits on non-letters.
        // "Hello, world!" and "hello world" must overlap fully.
        let score = LensVoiceNarrator.overlap("Hello, World!", "hello world")
        XCTAssertEqual(score, 1.0, accuracy: 0.0001)
    }

    func test_overlap_digitsAndSymbols_areNotTokens() {
        // Numbers and punctuation aren't counted — so two captions
        // differing only in numeric content overlap fully on words.
        let score = LensVoiceNarrator.overlap(
            "There are 3 people in the room",
            "There are 5 people in the room"
        )
        XCTAssertEqual(score, 1.0, accuracy: 0.0001)
    }

    // MARK: - Threshold/prefix constants are within sane ranges
    //
    // The audit pushed back on a fixed -40dB threshold in favor of
    // adaptive RMS, and on an unguarded per-frame stop() in favor of
    // overlap-gated narration. These constants are the contract;
    // changing them silently would re-introduce the zombie-voice
    // failure mode. Pin them.

    func test_constants_sceneChangeThresholdInExpectedRange() {
        // Threshold must be low enough that "person at desk" vs.
        // "person at the desk" stays above it (overlap > 0.50), but
        // high enough that a genuinely different scene falls under
        // (overlap < 0.20). The spec calls for 0.30 — pin it.
        XCTAssertEqual(LensVoiceNarrator.sceneChangeOverlapThreshold, 0.30,
                       accuracy: 0.0001)
    }

    // MARK: - Lifecycle smoke tests
    //
    // No real audio session — these just verify the singleton can be
    // activated/deactivated without crashing and that the state flags
    // update.

    @MainActor
    func test_lifecycle_activateDeactivateIdempotent() {
        // Start from a clean state.
        LensVoiceNarrator.shared.deactivate()
        XCTAssertFalse(LensVoiceNarrator.shared.isActive)

        LensVoiceNarrator.shared.activate()
        XCTAssertTrue(LensVoiceNarrator.shared.isActive)

        // Calling activate again should be a no-op.
        LensVoiceNarrator.shared.activate()
        XCTAssertTrue(LensVoiceNarrator.shared.isActive)

        LensVoiceNarrator.shared.deactivate()
        XCTAssertFalse(LensVoiceNarrator.shared.isActive)

        // Calling deactivate again should also be a no-op.
        LensVoiceNarrator.shared.deactivate()
        XCTAssertFalse(LensVoiceNarrator.shared.isActive)
    }
}
