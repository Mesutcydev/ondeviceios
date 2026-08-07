import XCTest
@testable import IOSLocalLLM

// MARK: - FastVLMPromptBuilderTests
//
// Covers the Qwen2 chat-template prompt construction for FastVLM.
// Validates that each task produces a properly formatted prompt with
// the expected tokens, image placeholder, and task-specific content.

final class FastVLMPromptBuilderTests: XCTestCase {

    // MARK: - Chat template format

    func test_formatChatTemplate_usesCorrectTokens() {
        let result = FastVLMPromptBuilder.formatChatTemplate(
            system: "You are helpful.",
            user: "Hello"
        )
        // Must contain Qwen2 special tokens.
        XCTAssertTrue(result.contains("<|im_start|>"))
        XCTAssertTrue(result.contains("<|im_end|>"))
        // Must contain both system and user roles.
        XCTAssertTrue(result.contains("system"))
        XCTAssertTrue(result.contains("user"))
        // Must end with assistant turn open.
        XCTAssertTrue(result.hasSuffix("assistant\n"))
    }

    func test_formatChatTemplate_preservesSystemContent() {
        let result = FastVLMPromptBuilder.formatChatTemplate(
            system: "Custom system message.",
            user: "Custom user message."
        )
        XCTAssertTrue(result.contains("Custom system message."))
        XCTAssertTrue(result.contains("Custom user message."))
    }

    // MARK: - Build prompt

    func test_buildPrompt_describeImage_containsImageToken() {
        let prompt = FastVLMPromptBuilder.buildPrompt(for: .describeImage)
        XCTAssertTrue(prompt.contains(FastVLMConfig.imageToken),
                      "describeImage prompt must include <image> token")
        XCTAssertTrue(prompt.contains("<|im_start|>"))
        XCTAssertTrue(prompt.contains("<|im_end|>"))
    }

    func test_buildPrompt_extractCode_containsCodeKeywords() {
        let prompt = FastVLMPromptBuilder.buildPrompt(for: .extractCode)
        XCTAssertTrue(prompt.contains("code"),
                      "extractCode prompt must mention code")
        XCTAssertTrue(prompt.contains(FastVLMConfig.imageToken))
    }

    func test_buildPrompt_reviewCode_containsReviewKeywords() {
        let prompt = FastVLMPromptBuilder.buildPrompt(for: .reviewCode)
        XCTAssertTrue(prompt.contains("review") || prompt.contains("Review"),
                      "reviewCode prompt must mention review")
    }

    func test_buildPrompt_answerQuestion_containsQuestionText() {
        let prompt = FastVLMPromptBuilder.buildPrompt(
            for: .answerQuestion("What is this?")
        )
        XCTAssertTrue(prompt.contains("What is this?"))
    }

    // MARK: - promptParts

    func test_promptParts_extractCode_returnsNonEmpty() {
        let (system, user) = FastVLMPromptBuilder.promptParts(for: .extractCode)
        XCTAssertFalse(system.isEmpty)
        XCTAssertFalse(user.isEmpty)
    }

    func test_promptParts_describeImage_returnsNonEmpty() {
        let (system, user) = FastVLMPromptBuilder.promptParts(for: .describeImage)
        XCTAssertFalse(system.isEmpty)
        XCTAssertFalse(user.isEmpty)
    }

    func test_promptParts_allTasks_haveImageTokenInUser() {
        let tasks: [FastVLMTask] = [
            .extractCode, .reviewCode, .describeImage,
            .answerQuestion("test")
        ]
        for task in tasks {
            let (_, user) = FastVLMPromptBuilder.promptParts(for: task)
            XCTAssertTrue(user.contains(FastVLMConfig.imageToken),
                          "Task \(task) user prompt must include <image> token")
        }
    }

    // MARK: - Token estimation

    func test_estimatedPromptTokens_isPositive() {
        for task: FastVLMTask in [.extractCode, .reviewCode, .describeImage,
                                   .answerQuestion("test question")] {
            let tokens = FastVLMPromptBuilder.estimatedPromptTokens(for: task)
            XCTAssertGreaterThan(tokens, 0,
                                 "Token estimate for \(task) must be positive")
        }
    }

    func test_estimatedPromptTokens_includesImagePatches() {
        let tokens = FastVLMPromptBuilder.estimatedPromptTokens(for: .describeImage)
        // Must be at least the 256 image patches.
        XCTAssertGreaterThanOrEqual(tokens, FastVLMConfig.encoderNumPatches)
    }
}

