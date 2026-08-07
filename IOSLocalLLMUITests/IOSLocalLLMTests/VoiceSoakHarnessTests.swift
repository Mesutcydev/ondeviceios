import XCTest
@testable import IOSLocalLLM

#if DEBUG

@MainActor
final class VoiceSoakHarnessTests: XCTestCase {

    func testShortMockSoakReachesQuiescentState() async {
        let harness = VoiceSoakHarness.shared
        harness.stop()
        VoiceDiagnosticsCenter.shared.reset()
        SpeechPlaybackCoordinator.shared.reset()

        // ~3 cycles via a short deadline-style run.
        harness.run(durationMinutes: 0.05) // 3 seconds
        let deadline = Date().addingTimeInterval(8)
        while harness.isRunning && Date() < deadline {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        harness.stop()
        SpeechPlaybackCoordinator.shared.reset()

        XCTAssertGreaterThan(harness.cyclesCompleted, 0)
        let report = VoiceDiagnosticsCenter.shared.snapshotReport(
            conversationModel: "Ternary Bonsai 8B",
            ttsEngine: "mock",
            voiceID: "test",
            route: "Speaker",
            alignment: "estimated",
            phase: "idle"
        )
        XCTAssertTrue(report.contains("Conversation model"))
        XCTAssertTrue(VoiceDiagnosticsCenter.shared.counters.isQuiescent
                      || VoiceDiagnosticsCenter.shared.counters.activeDisplayLinks == 0)
    }
}

#endif
