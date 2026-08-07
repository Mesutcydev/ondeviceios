import XCTest
@testable import IOSLocalLLM

final class KaraokeTimelineTests: XCTestCase {

    func testPhraseBuilderGroupsWords() {
        let words: [SpeechWordTiming] = (0..<12).map { i in
            SpeechWordTiming(
                word: i == 2 ? "one," : "w\(i)",
                startTime: Double(i) * 0.2,
                endTime: Double(i) * 0.2 + 0.18,
                utf16Range: NSRange(location: i * 3, length: 2)
            )
        }
        let seg = SpeechSegment(
            text: "group",
            startTime: 0,
            endTime: 2.4,
            words: words,
            timingSource: .estimated
        )
        let timeline = KaraokePhraseBuilder.build(from: [seg])
        XCTAssertFalse(timeline.phrases.isEmpty)
        XCTAssertLessThan(timeline.phrases.count, words.count)
        // Binary search finds a phrase mid-stream.
        XCTAssertNotNil(timeline.phraseIndex(at: 1.0))
        let idx0 = timeline.phraseIndex(at: 0.05)
        let idx1 = timeline.phraseIndex(at: 2.0)
        XCTAssertNotNil(idx0)
        XCTAssertNotNil(idx1)
        if let a = idx0, let b = idx1 {
            XCTAssertLessThanOrEqual(a, b)
        }
    }

    func testNeverMovesToEarlierPhraseIndexConceptually() {
        let phrases = [
            KaraokePhrase(text: "a", startTime: 0, endTime: 0.6, utf16Range: NSRange(location: 0, length: 1)),
            KaraokePhrase(text: "b", startTime: 0.6, endTime: 1.2, utf16Range: NSRange(location: 2, length: 1)),
            KaraokePhrase(text: "c", startTime: 1.2, endTime: 1.8, utf16Range: NSRange(location: 4, length: 1))
        ]
        let timeline = KaraokeTimeline(phrases: phrases)
        XCTAssertEqual(timeline.phraseIndex(at: 0.3), 0)
        XCTAssertEqual(timeline.phraseIndex(at: 0.9), 1)
        XCTAssertEqual(timeline.phraseIndex(at: 1.5), 2)
        XCTAssertEqual(timeline.nextBoundary(after: 0.3), 0.6)
    }

    func testSpokenEndIsMonotonicWithTime() {
        let phrases = [
            KaraokePhrase(text: "hello", startTime: 0, endTime: 0.8, utf16Range: NSRange(location: 0, length: 5)),
            KaraokePhrase(text: "world", startTime: 0.8, endTime: 1.6, utf16Range: NSRange(location: 6, length: 5))
        ]
        let timeline = KaraokeTimeline(phrases: phrases)
        let early = timeline.spokenUTF16End(at: 0.2, transcriptLength: 11)
        let late = timeline.spokenUTF16End(at: 1.4, transcriptLength: 11)
        XCTAssertLessThanOrEqual(early, late)
    }

    func testWordlessSegmentsInsertSpaceBetweenRanges() {
        let segments = [
            SpeechSegment(
                text: "Hello",
                startTime: 0,
                endTime: 0.5,
                words: [],
                timingSource: .phraseLevel
            ),
            SpeechSegment(
                text: "world",
                startTime: 0.5,
                endTime: 1.0,
                words: [],
                timingSource: .phraseLevel
            )
        ]
        let timeline = KaraokePhraseBuilder.build(from: segments)
        XCTAssertEqual(timeline.phrases.count, 2)
        XCTAssertEqual(timeline.phrases[0].utf16Range.location, 0)
        XCTAssertEqual(timeline.phrases[0].utf16Range.length, 5)
        // Second phrase starts after an assumed separator so it maps into
        // "Hello world" rather than overlapping "Helloworld".
        XCTAssertEqual(timeline.phrases[1].utf16Range.location, 6)
        XCTAssertEqual(timeline.phrases[1].utf16Range.length, 5)
    }

    func testPhraseLevelWordCarriesUTF16BaseOffset() {
        let base = 12
        let text = "Next sentence."
        let seg = SpeechSegment(
            text: text,
            startTime: 1.0,
            endTime: 2.0,
            words: [
                SpeechWordTiming(
                    word: text,
                    startTime: 1.0,
                    endTime: 2.0,
                    utf16Range: NSRange(location: base, length: (text as NSString).length)
                )
            ],
            timingSource: .phraseLevel
        )
        let timeline = KaraokePhraseBuilder.build(from: [seg])
        XCTAssertEqual(timeline.phrases.first?.utf16Range.location, base)
        XCTAssertEqual(timeline.phrases.first?.utf16Range.length, (text as NSString).length)
    }
}
