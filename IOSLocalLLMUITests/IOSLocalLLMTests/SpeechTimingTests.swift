import XCTest
import SwiftUI
@testable import IOSLocalLLM

@MainActor
final class SpeechTimingTests: XCTestCase {

    private let provider = EstimatedSpeechTimingProvider.shared

    func testWordTimingEstimationAllocatesAcrossDuration() async throws {
        let text = "Hello world, this is a test."
        let segments = try await provider.timings(
            for: text,
            audioDuration: 2.0,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertEqual(segments.count, 1)
        let seg = try XCTUnwrap(segments.first)
        XCTAssertEqual(seg.timingSource, .estimated)
        XCTAssertFalse(seg.words.isEmpty)
        XCTAssertEqual(seg.startTime, 0, accuracy: 0.001)
        XCTAssertEqual(seg.endTime, 2.0, accuracy: 0.02)
        XCTAssertEqual(seg.words.last?.endTime ?? -1, 2.0, accuracy: 0.02)
        // Words are ordered in time.
        for i in 1..<seg.words.count {
            XCTAssertGreaterThanOrEqual(seg.words[i].startTime, seg.words[i - 1].startTime)
        }
    }

    func testPunctuationPauseAllocation() {
        let comma = EstimatedSpeechTimingProvider.Token(
            text: "yes,",
            utf16Range: NSRange(location: 0, length: 4),
            trailingPunctuation: ","
        )
        let period = EstimatedSpeechTimingProvider.Token(
            text: "end.",
            utf16Range: NSRange(location: 0, length: 4),
            trailingPunctuation: "."
        )
        XCTAssertGreaterThan(provider.pauseDuration(for: period), provider.pauseDuration(for: comma))
    }

    func testPlaybackTimeMapsToActiveWord() async throws {
        let text = "One two three"
        let segments = try await provider.timings(
            for: text,
            audioDuration: 3.0,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        let mid = SpeechProgressMapper.resolve(
            time: 1.5,
            segments: segments,
            transcriptUTF16Length: (text as NSString).length
        )
        XCTAssertNotNil(mid.wordIndex)
        XCTAssertNotNil(mid.activeRange)
        let end = SpeechProgressMapper.resolve(
            time: 3.0,
            segments: segments,
            transcriptUTF16Length: (text as NSString).length
        )
        XCTAssertEqual(end.spokenEnd, (text as NSString).length)
    }

    func testSentenceLevelFallbackEmptyWords() {
        let seg = SpeechSegment(
            text: "Hello there",
            startTime: 0,
            endTime: 1,
            words: [],
            timingSource: .phraseLevel
        )
        let mapped = SpeechProgressMapper.resolve(
            time: 0.5,
            segments: [seg],
            transcriptUTF16Length: 11
        )
        XCTAssertEqual(mapped.segmentIndex, 0)
    }

    func testActiveRangeNeverExceedsTranscriptBounds() async throws {
        let text = "Bound check"
        let segments = try await provider.timings(
            for: text,
            audioDuration: 1.0,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        let len = (text as NSString).length
        let mapped = SpeechProgressMapper.resolve(time: 0.4, segments: segments, transcriptUTF16Length: len)
        if let range = mapped.activeRange {
            XCTAssertGreaterThanOrEqual(range.location, 0)
            XCTAssertLessThanOrEqual(range.location + range.length, len)
        }
        XCTAssertLessThanOrEqual(mapped.spokenEnd, len)
    }

    func testEmptyTextThrows() async {
        do {
            _ = try await provider.timings(
                for: "   ",
                audioDuration: 1,
                engineCapabilities: [],
                timelineOffset: 0,
                utf16BaseOffset: 0
            )
            XCTFail("Expected emptyText")
        } catch SpeechTimingError.emptyText {
            // expected
        } catch {
            XCTFail("Unexpected \(error)")
        }
    }

    func testZeroDurationUsesFloor() async throws {
        let segments = try await provider.timings(
            for: "Hi",
            audioDuration: 0,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertGreaterThan(segments.first?.endTime ?? 0, 0)
    }

    func testUnicodeTurkishArabicEmoji() async throws {
        let text = "Merhaba dünya 🌍 مرحبا"
        let segments = try await provider.timings(
            for: text,
            audioDuration: 2.5,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 0,
            utf16BaseOffset: 0
        )
        XCTAssertFalse(segments.isEmpty)
        let tokens = provider.tokenize(text)
        XCTAssertFalse(tokens.isEmpty)
        // UTF-16 ranges must stay inside the string.
        let len = (text as NSString).length
        for token in tokens {
            XCTAssertLessThanOrEqual(token.utf16Range.location + token.utf16Range.length, len)
        }
    }

    func testTimelineOffsetShiftsWordTimes() async throws {
        let segments = try await provider.timings(
            for: "Later chunk",
            audioDuration: 1.0,
            engineCapabilities: [.completePCMBuffers],
            timelineOffset: 4.0,
            utf16BaseOffset: 10
        )
        let first = try XCTUnwrap(segments.first?.words.first)
        XCTAssertGreaterThanOrEqual(first.startTime, 4.0 - 0.001)
        XCTAssertEqual(first.utf16Range.location, 10)
    }

    func testStaleGenerationRejected() {
        let coordinator = SpeechPlaybackCoordinator.shared
        coordinator.beginUtterance(karaokeSeed: "Hello")
        let generationBefore = coordinator.snapshot.currentTime
        coordinator.interrupt()
        XCTAssertEqual(coordinator.snapshot.phase, .interrupted)
        XCTAssertNil(coordinator.snapshot.activeUTF16Range)
        _ = generationBefore
    }

    func testInterruptClearsActiveHighlight() {
        let coordinator = SpeechPlaybackCoordinator.shared
        coordinator.beginUtterance(karaokeSeed: "Testing interrupt")
        coordinator.interrupt()
        XCTAssertEqual(coordinator.snapshot.normalizedLevel, 0)
        XCTAssertNil(coordinator.snapshot.activeWordIndex)
        coordinator.reset()
    }

    func testKaraokeBuilderKeepsFullTranscript() {
        let text = String(repeating: "word ", count: 40)
        let attr = KaraokeTextBuilder.build(
            text: text,
            spokenUTF16End: 20,
            activeRange: NSRange(location: 20, length: 4),
            ink: .primary,
            inkSecondary: .secondary,
            accent: .accentColor
        )
        XCTAssertEqual(String(attr.characters), text)
    }

    func testCapabilitiesHaveNoExactWordTimings() {
        for kind in VoiceEngineKind.allCases {
            let caps = TTSEngineCapabilities.capabilities(for: kind)
            XCTAssertFalse(caps.contains(.exactWordTimings))
            XCTAssertTrue(caps.contains(.completePCMBuffers))
            XCTAssertTrue(caps.contains(.playbackMetering))
        }
        XCTAssertTrue(TTSEngineCapabilities.capabilities(for: .kittenTTS).contains(.engineDerivedDurations))
    }
}
