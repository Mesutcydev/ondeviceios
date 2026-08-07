import XCTest
@testable import IOSLocalLLM
import AVFoundation

final class AudioVisualClockTests: XCTestCase {

    func testDisplayedSampleAppliesLatency() {
        var snap = AudioPlaybackSnapshot.idle
        snap.isPlaying = true
        snap.sampleRate = 24_000
        snap.currentPlayerSample = 24_000 // 1.0s
        snap.scheduledSampleCount = 48_000
        snap.renderHostTime = AVAudioTime.hostTime(forSeconds: 100)
        snap.outputPresentationLatency = 0.050 // 50 ms → 1200 samples

        let target = AVAudioTime.hostTime(forSeconds: 100) // same as render
        let sample = AudioVisualClock.displayedSample(snapshot: snap, targetHostTime: target)
        XCTAssertEqual(sample, 24_000 - 1_200)
    }

    func testDisplayedSampleExtrapolatesToTargetHost() {
        var snap = AudioPlaybackSnapshot.idle
        snap.isPlaying = true
        snap.sampleRate = 24_000
        snap.currentPlayerSample = 0
        snap.scheduledSampleCount = 240_000
        snap.renderHostTime = AVAudioTime.hostTime(forSeconds: 50)
        snap.outputPresentationLatency = 0

        let target = AVAudioTime.hostTime(forSeconds: 50.100) // +100 ms
        let sample = AudioVisualClock.displayedSample(snapshot: snap, targetHostTime: target)
        XCTAssertEqual(sample, 2_400)
    }

    func testDisplayedSampleClampsToScheduledRange() {
        var snap = AudioPlaybackSnapshot.idle
        snap.isPlaying = true
        snap.sampleRate = 16_000
        snap.currentPlayerSample = 15_000
        snap.scheduledSampleCount = 16_000
        snap.renderHostTime = AVAudioTime.hostTime(forSeconds: 10)
        snap.outputPresentationLatency = 0

        let target = AVAudioTime.hostTime(forSeconds: 11) // +1s would overshoot
        let sample = AudioVisualClock.displayedSample(snapshot: snap, targetHostTime: target)
        XCTAssertEqual(sample, 16_000)
    }

    func testCueBinarySearchAndSpokenEnd() {
        let phrases = [
            KaraokePhrase(
                id: UUID(),
                text: "hello",
                startTime: 0,
                endTime: 0.5,
                utf16Range: NSRange(location: 0, length: 5)
            ),
            KaraokePhrase(
                id: UUID(),
                text: "world",
                startTime: 0.5,
                endTime: 1.0,
                utf16Range: NSRange(location: 6, length: 5)
            )
        ]
        let track = KaraokeCueTrack.from(phrases: phrases, sampleRate: 24_000, timingSource: .estimated)
        XCTAssertEqual(track.cues.count, 2)
        // Ranges are [start, end) in samples.
        XCTAssertEqual(track.cue(at: 0)?.id, 0)
        XCTAssertEqual(track.cue(at: 11_999)?.id, 0)
        XCTAssertEqual(track.cue(at: 12_000)?.id, 1)
        // Mid-phrase interpolates through the active cue's UTF-16 span.
        XCTAssertEqual(track.spokenUTF16End(at: 6_000, transcriptLength: 11), 2)
        XCTAssertEqual(track.spokenUTF16End(at: 12_000, transcriptLength: 11), 6)
        XCTAssertEqual(track.spokenUTF16End(at: 24_000, transcriptLength: 11), 11)
    }

    func testGenerationInvalidationIgnoresStaleSnapshot() {
        var snap = AudioPlaybackSnapshot.idle
        snap.generation = 3
        snap.isPlaying = true
        snap.sampleRate = 24_000
        snap.currentPlayerSample = 8_000
        snap.scheduledSampleCount = 24_000
        // A consumer must compare generation — clock itself still converts.
        let sample = AudioVisualClock.displayedSample(
            snapshot: snap,
            targetHostTime: snap.renderHostTime
        )
        XCTAssertEqual(sample, 8_000)
        XCTAssertEqual(snap.generation, 3)
    }
}
