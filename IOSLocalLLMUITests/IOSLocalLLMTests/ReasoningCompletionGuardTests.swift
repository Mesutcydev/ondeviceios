import XCTest
@testable import IOSLocalLLM

final class ReasoningCompletionGuardTests: XCTestCase {
    func test_openThinkWithoutAnswer_needsRecovery() {
        let text = "<think>working through the problem"
        XCTAssertTrue(ReasoningCompletionGuard.needsFinalAnswerRecovery(text))
    }

    func test_closedThinkWithAnswer_doesNotNeedRecovery() {
        let text = "<think>private</think>The answer is 42."
        XCTAssertFalse(ReasoningCompletionGuard.needsFinalAnswerRecovery(text))
    }

    func test_visibleTextBeforeOpenThink_doesNotNeedRecovery() {
        let text = "The answer is 42.\n<think>extra"
        XCTAssertFalse(ReasoningCompletionGuard.needsFinalAnswerRecovery(text))
    }

    func test_closeForDisplay_closesOpenThinkBlock() {
        let closed = ReasoningCompletionGuard.closeForDisplay("<think>partial")
        XCTAssertTrue(closed.hasSuffix("</think>"))
    }
}
