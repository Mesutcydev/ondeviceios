import XCTest
@testable import IOSLocalLLM

final class TurnEndpointerTests: XCTestCase {

    func test_observeListening_emitsTurnStartedOnce() {
        let endpointer = TurnEndpointer()
        let t0 = Date()

        XCTAssertEqual(
            endpointer.observeListening(probability: 0.9, completion: 0.2, now: t0),
            .none,
            "speech must persist before onset"
        )
        XCTAssertEqual(
            endpointer.observeListening(probability: 0.9, completion: 0.2, now: t0.addingTimeInterval(0.25)),
            .turnStarted
        )
        XCTAssertEqual(
            endpointer.observeListening(probability: 0.9, completion: 0.2, now: t0.addingTimeInterval(0.40)),
            .none,
            "turnStarted is one-shot per listening reset"
        )
    }

    func test_observeListening_endsTurnAfterSilence() {
        let endpointer = TurnEndpointer()
        let t0 = Date()

        _ = endpointer.observeListening(probability: 0.9, completion: 1.0, now: t0)
        _ = endpointer.observeListening(probability: 0.9, completion: 1.0, now: t0.addingTimeInterval(0.25))

        XCTAssertEqual(
            endpointer.observeListening(probability: 0.1, completion: 1.0, now: t0.addingTimeInterval(0.30)),
            .none
        )
        XCTAssertEqual(
            endpointer.observeListening(probability: 0.1, completion: 1.0, now: t0.addingTimeInterval(0.90)),
            .turnEnded
        )
    }
}
