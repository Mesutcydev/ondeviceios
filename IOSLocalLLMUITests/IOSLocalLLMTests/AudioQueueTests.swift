import AVFoundation
import XCTest
@testable import IOSLocalLLM

// MARK: - AudioQueueTests
//
// Pins the producer-consumer queue that drives streaming TTS playback.
// Lifted from `StreamingSpeechSession` and renamed; the suspension
// semantics and the cancellation contract (synchronous `stop`, no
// fadeout) must remain bit-for-bit identical to the previous behavior.

@MainActor
final class AudioQueueTests: XCTestCase {

    // MARK: - enqueue / next basics

    func test_enqueueOnce_nextReturnsImmediately() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "hello"))
        let item = await q.next()
        XCTAssertEqual(item?.text, "hello")
    }

    func test_enqueueTrimsWhitespace() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "  trimmed  "))
        let item = await q.next()
        XCTAssertEqual(item?.text, "trimmed")
    }

    func test_enqueueEmptyDropped() async {
        let q = AudioQueue()
        q.enqueue(.init(text: ""))
        q.enqueue(.init(text: "   "))
        q.enqueue(.init(text: "real"))
        let item = await q.next()
        XCTAssertEqual(item?.text, "real")
    }

    func test_enqueueDoesNotInterrupt_queueDepthReflectsPending() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "first"))
        q.enqueue(.init(text: "second"))
        q.enqueue(.init(text: "third"))
        // Consumer hasn't pulled yet — 3 pending, but `next()` will
        // dequeue immediately when called. Verify depth post-dequeue.
        let item = await q.next()
        XCTAssertEqual(item?.text, "first")
        XCTAssertEqual(q.queueDepth, 2,
                       "After dequeueing first, two remain pending")
    }

    func test_languageTagRoundtrips() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "Merhaba.", language: "tr-TR"))
        let item = await q.next()
        XCTAssertEqual(item?.language, "tr-TR")
    }

    func test_engineHintRoundtrips() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "code-only", language: nil, engineHint: .appleSystem))
        let item = await q.next()
        XCTAssertEqual(item?.engineHint, .appleSystem)
    }

    // MARK: - Observable state

    func test_isSpeakingTogglesAroundConsumer() async {
        let q = AudioQueue()
        XCTAssertFalse(q.isSpeaking)
        q.enqueue(.init(text: "first", language: "en-US"))
        _ = await q.next()
        XCTAssertTrue(q.isSpeaking, "Dequeue marks the queue as speaking")
        XCTAssertEqual(q.currentLanguage, "en-US")

        // Finish + drain → next() returns nil → speaking flips off.
        q.finish()
        let tail = await q.next()
        XCTAssertNil(tail)
        XCTAssertFalse(q.isSpeaking)
        XCTAssertNil(q.currentLanguage)
    }

    func test_currentLanguageReflectsActiveUtterance() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "Hello.", language: "en-US"))
        q.enqueue(.init(text: "Merhaba.", language: "tr-TR"))
        _ = await q.next()
        XCTAssertEqual(q.currentLanguage, "en-US")
        _ = await q.next()
        XCTAssertEqual(q.currentLanguage, "tr-TR")
    }

    // MARK: - finish() — drain all to completion

    func test_finishLetsAllPendingDrain() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "one"))
        q.enqueue(.init(text: "two"))
        q.finish()
        let first  = await q.next()
        let second = await q.next()
        let third  = await q.next()
        XCTAssertEqual(first?.text,  "one")
        XCTAssertEqual(second?.text, "two")
        XCTAssertNil(third, "After draining, next() returns nil")
    }

    // MARK: - drainPending() — drop queued, let current finish

    func test_drainPendingDropsQueuedItems() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "first"))
        q.enqueue(.init(text: "second"))
        q.enqueue(.init(text: "third"))
        // Consumer pulled "first" — it's "currently playing". Then a
        // drainPending should drop second and third.
        let first = await q.next()
        XCTAssertEqual(first?.text, "first")

        q.drainPending()
        XCTAssertEqual(q.queueDepth, 0)

        let next = await q.next()
        XCTAssertNil(next, "drainPending sealed the queue — no more items")
    }

    // MARK: - stop() — immediate cancel, synchronous

    func test_stopIsSynchronousAndImmediate() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "one", language: "en-US"))
        q.enqueue(.init(text: "two"))
        _ = await q.next()
        XCTAssertTrue(q.isSpeaking)

        q.stop()
        // No await between stop() and these reads — the cancellation
        // contract requires sync state updates.
        XCTAssertFalse(q.isSpeaking, "stop() must clear isSpeaking synchronously")
        XCTAssertEqual(q.queueDepth, 0)
        XCTAssertNil(q.currentLanguage)

        let item = await q.next()
        XCTAssertNil(item, "After stop, next() returns nil even if items were pending")
    }

    func test_stopUnblocksAwaitingConsumer() async {
        let q = AudioQueue()

        // Consumer suspends on next() before any enqueue.
        async let waiting = q.next()

        // Give the suspension a moment to engage.
        try? await Task.sleep(nanoseconds: 50_000_000)
        q.stop()
        let result = await waiting
        XCTAssertNil(result, "stop() must wake a suspended consumer with nil")
    }

    func test_enqueueAfterStopIsDropped() async {
        let q = AudioQueue()
        q.stop()
        q.enqueue(.init(text: "shouldnt show"))
        let item = await q.next()
        XCTAssertNil(item)
    }

    // MARK: - finish() vs drainPending() interaction

    func test_finishUnblocksAwaitingConsumerWhenQueueEmpty() async {
        let q = AudioQueue()
        async let waiting = q.next()
        try? await Task.sleep(nanoseconds: 50_000_000)
        q.finish()
        let result = await waiting
        XCTAssertNil(result, "finish() with empty queue wakes consumer with nil")
    }

    func test_finishDoesNotDropPending() async {
        let q = AudioQueue()
        q.enqueue(.init(text: "queued"))
        q.finish()
        let item = await q.next()
        XCTAssertEqual(item?.text, "queued")
    }
}

// MARK: - Apple system speech route regression

@MainActor
final class SystemSpeechPCMTests: XCTestCase {
    func test_appleSystemVoiceRendersNonEmptyPCM() async throws {
        let service = SystemSpeechService()
        guard let voice = service.defaultVoice else {
            throw XCTSkip("This simulator has no installed Apple speech voice.")
        }

        let result = try await service.synthesize(
            text: "IOSLocalLLM audio route check.",
            voice: voice,
            settings: VoiceSynthesisSettings()
        )

        XCTAssertEqual(result.engineKind, .appleSystem)
        XCTAssertTrue(result.requiresExternalPlayback)
        XCTAssertGreaterThan(result.sampleRate, 0)
        XCTAssertGreaterThan(result.duration, 0)
        XCTAssertGreaterThan(result.pcmBuffer?.frameLength ?? 0, 0)
    }

    /// iOS 17+ lists the novelty (Albert, Zarvox, …) and Eloquence (Eddy,
    /// Flo, …) voices in speechVoices(). They are robotic by design and must
    /// never be offered or auto-picked — the "synthetic robotic voice for
    /// everything" bug was Albert winning the alphabetical tie-break.
    func test_roboticSynthVoicesNeverOffered() {
        let service = SystemSpeechService()
        for v in service.availableVoices {
            let id = v.id.lowercased()
            XCTAssertFalse(id.contains("eloquence"), "Eloquence voice offered: \(v.id)")
            XCTAssertFalse(id.contains(".speech.synthesis.voice"), "Legacy MacinTalk voice offered: \(v.id)")
        }
        if let def = service.defaultVoice {
            let raw = def.id.replacingOccurrences(of: "system:", with: "")
            if let av = AVSpeechSynthesisVoice(identifier: raw) {
                XCTAssertFalse(SystemSpeechService.isRoboticSynth(av),
                               "Default voice is a robotic synth: \(def.id)")
            }
        }
    }
}
