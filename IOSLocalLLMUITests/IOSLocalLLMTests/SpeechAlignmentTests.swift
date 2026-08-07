import AVFoundation
import XCTest
@testable import IOSLocalLLM

@MainActor
final class SpeechAlignmentTests: XCTestCase {

    func testPredDurMapsToMonotonicWordTimings() {
        let text = "Hello world"
        let phonemeIDs = PhonemizerEN.shared.phonemeIDs(for: text)
        XCTAssertGreaterThan(phonemeIDs.count, 3)
        let frames: [Float] = phonemeIDs.map { _ in 4 }
        let result = PredDurWordAligner.align(
            text: text,
            phonemeIDs: phonemeIDs,
            predDurFrames: frames,
            validTokenCount: phonemeIDs.count,
            audioDuration: 2.0,
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.accuracy, .engineDerived)
        let words = result?.segments.first?.words ?? []
        XCTAssertFalse(words.isEmpty)
        assertInvariants(words: words, audioDuration: 2.0, timelineOffset: 0, transcript: text)
    }

    func testSilenceTokenValidationRejectsNegative() {
        let bad = [
            SpeechWordTiming(word: "a", startTime: 0.5, endTime: 0.2, utf16Range: NSRange(location: 0, length: 1))
        ]
        XCTAssertFalse(PredDurWordAligner.validate(timings: bad, audioDuration: 1, timelineOffset: 0))
    }

    func testPunctuationPauseLongerThanComma() {
        let provider = EstimatedSpeechTimingProvider.shared
        let comma = EstimatedSpeechTimingProvider.Token(
            text: "yes,", utf16Range: NSRange(location: 0, length: 4), trailingPunctuation: ","
        )
        let period = EstimatedSpeechTimingProvider.Token(
            text: "end.", utf16Range: NSRange(location: 0, length: 4), trailingPunctuation: "."
        )
        XCTAssertGreaterThan(provider.pauseDuration(for: period), provider.pauseDuration(for: comma))
    }

    func testWaveformSilenceDetectionFindsRegions() {
        let buffer = makeToneBuffer(duration: 1.0, silentMiddle: true)
        let env = AcousticSpeechAlignmentProvider.rmsEnvelope(buffer, windowMs: 20)
        let regions = AcousticSpeechAlignmentProvider.speechRegions(envelope: env, duration: 1.0)
        XCTAssertFalse(regions.isEmpty)
        for r in regions {
            XCTAssertGreaterThanOrEqual(r.start, 0)
            XCTAssertGreaterThanOrEqual(r.end, r.start)
            XCTAssertLessThanOrEqual(r.end, 1.05)
        }
    }

    func testSentenceSegmentation() {
        let clauses = AcousticSpeechAlignmentProvider.splitClauses("One. Two! Three?")
        XCTAssertEqual(clauses.count, 3)
    }

    func testAcousticAlignClampsToDuration() async {
        let text = "Hello there friend"
        let buffer = makeToneBuffer(duration: 1.5, silentMiddle: false)
        let result = await AcousticSpeechAlignmentProvider.shared.align(
            text: text,
            audio: buffer,
            synthesisMetadata: SynthesisMetadata(sourceText: text, engineKind: .appleSystem),
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertFalse(result.segments.isEmpty)
        if let words = result.segments.first?.words {
            assertInvariants(words: words, audioDuration: 1.5, timelineOffset: 0, transcript: text)
        }
        XCTAssertNotEqual(result.accuracy, .exact)
    }

    func testZeroDurationAudioPhraseFallback() async {
        let buffer = makeToneBuffer(duration: 0.01, silentMiddle: false)
        let result = await AcousticSpeechAlignmentProvider.shared.align(
            text: "Hi",
            audio: buffer,
            synthesisMetadata: nil,
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        // Very short audio → empty or phrase-level, never crash.
        XCTAssertTrue(result.accuracy == .phraseLevel || result.segments.isEmpty || result.accuracy == .estimated)
    }

    func testEmptyTranscript() async {
        let buffer = makeToneBuffer(duration: 0.5, silentMiddle: false)
        let result = await AcousticSpeechAlignmentProvider.shared.align(
            text: "   ",
            audio: buffer,
            synthesisMetadata: nil
        )
        XCTAssertTrue(result.segments.isEmpty || result.accuracy == .phraseLevel)
    }

    func testOneWordResponse() async throws {
        let segments = try await EstimatedSpeechTimingProvider.shared.timings(
            for: "Hi",
            audioDuration: 0.5,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertEqual(segments.first?.words.count, 1)
    }

    func testVeryLongResponseMonotonic() async throws {
        let text = Array(repeating: "word", count: 80).joined(separator: " ")
        let segments = try await EstimatedSpeechTimingProvider.shared.timings(
            for: text,
            audioDuration: 20,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 1.0,
            utf16BaseOffset: 0
        )
        let words = segments.first?.words ?? []
        assertInvariants(words: words, audioDuration: 20, timelineOffset: 1.0, transcript: text)
    }

    func testRepeatedWords() async throws {
        let text = "the the the"
        let segments = try await EstimatedSpeechTimingProvider.shared.timings(
            for: text,
            audioDuration: 1.2,
            engineCapabilities: [],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertEqual(segments.first?.words.count, 3)
    }

    func testTurkishArabicEnglishEmojiMixed() async throws {
        let text = "Merhaba 🌍 مرحبا hello"
        let segments = try await EstimatedSpeechTimingProvider.shared.timings(
            for: text,
            audioDuration: 2,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        let words = segments.first?.words ?? []
        assertInvariants(words: words, audioDuration: 2, timelineOffset: 0, transcript: text)
    }

    func testStaleAlignmentRejectedByGeneration() {
        let c = SpeechPlaybackCoordinator.shared
        c.beginUtterance(karaokeSeed: "Old")
        c.interrupt()
        XCTAssertEqual(c.snapshot.phase, .interrupted)
        XCTAssertNil(c.snapshot.activeUTF16Range)
        c.reset()
    }

    func testPhraseGrouperMergesShortWords() {
        let words = [
            SpeechWordTiming(word: "a", startTime: 0, endTime: 0.05, utf16Range: NSRange(location: 0, length: 1)),
            SpeechWordTiming(word: "tiny", startTime: 0.05, endTime: 0.1, utf16Range: NSRange(location: 2, length: 4)),
            SpeechWordTiming(word: "word", startTime: 0.1, endTime: 0.4, utf16Range: NSRange(location: 7, length: 4))
        ]
        let seg = SpeechSegment(
            text: "a tiny word",
            startTime: 0,
            endTime: 0.4,
            words: words,
            timingSource: .estimated
        )
        let range = KaraokePhraseGrouper.phraseRange(
            containing: words[0].utf16Range,
            time: 0.06,
            segments: [seg],
            minimumWordDuration: 0.12
        )
        XCTAssertNotNil(range)
        XCTAssertEqual(range?.location, 0)
        XCTAssertGreaterThan(range?.length ?? 0, 1)
    }

    func testKittenCapabilitiesIncludeEngineDerived() {
        let caps = TTSEngineCapabilities.capabilities(for: .kittenTTS)
        XCTAssertTrue(caps.contains(.engineDerivedDurations))
        XCTAssertFalse(caps.contains(.exactWordTimings))
    }

    func testKokoroAndAppleHaveNoExactTimings() {
        XCTAssertFalse(TTSEngineCapabilities.capabilities(for: .kokoro).contains(.exactWordTimings))
        XCTAssertFalse(TTSEngineCapabilities.capabilities(for: .appleSystem).contains(.exactWordTimings))
        XCTAssertFalse(TTSEngineCapabilities.capabilities(for: .kokoro).contains(.engineDerivedDurations))
    }

    // MARK: - Helpers

    private func assertInvariants(
        words: [SpeechWordTiming],
        audioDuration: TimeInterval,
        timelineOffset: TimeInterval,
        transcript: String
    ) {
        let len = (transcript as NSString).length
        var prev = timelineOffset - 0.0001
        for w in words {
            XCTAssertGreaterThanOrEqual(w.startTime, timelineOffset - 0.001)
            XCTAssertGreaterThanOrEqual(w.endTime, w.startTime)
            XCTAssertLessThanOrEqual(w.endTime, timelineOffset + audioDuration + 0.05)
            XCTAssertGreaterThanOrEqual(w.startTime, prev - 0.001)
            XCTAssertGreaterThanOrEqual(w.utf16Range.location, 0)
            XCTAssertLessThanOrEqual(w.utf16Range.location + w.utf16Range.length, len)
            prev = w.startTime
        }
    }

    private func makeToneBuffer(duration: TimeInterval, silentMiddle: Bool) -> AVAudioPCMBuffer {
        let rate = 24_000.0
        let frames = AVAudioFrameCount(max(1, rate * duration))
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        let data = buffer.floatChannelData![0]
        let n = Int(frames)
        for i in 0..<n {
            let t = Double(i) / rate
            if silentMiddle, t > duration * 0.35, t < duration * 0.65 {
                data[i] = 0
            } else {
                data[i] = Float(sin(t * 440 * 2 * .pi) * 0.2)
            }
        }
        return buffer
    }
}
