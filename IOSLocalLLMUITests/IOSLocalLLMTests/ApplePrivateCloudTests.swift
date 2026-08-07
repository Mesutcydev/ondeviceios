import XCTest
@testable import IOSLocalLLM

// Always-compiled PCC metadata + the cumulative→delta streaming logic. These
// run on the GA build (no FM_PCC); the FoundationModels-touching runtime is
// validated separately on a physical iOS 27 device.
final class ApplePrivateCloudTests: XCTestCase {

    // MARK: streamDelta — cumulative snapshots → append-ready deltas

    func test_streamDelta_emitsOnlyNewSuffix() {
        XCTAssertEqual(ApplePrivateCloud.streamDelta(previous: "Hel", full: "Hello"), "lo")
    }

    func test_streamDelta_firstChunkIsWholeString() {
        XCTAssertEqual(ApplePrivateCloud.streamDelta(previous: "", full: "Hi"), "Hi")
    }

    func test_streamDelta_noGrowthEmitsEmpty() {
        XCTAssertEqual(ApplePrivateCloud.streamDelta(previous: "Hello", full: "Hello"), "")
    }

    func test_streamDelta_reassemblesFullText() {
        // Driving a sequence of cumulative snapshots must reconstruct the text
        // exactly — no dropped or duplicated characters.
        let snapshots = ["", "The", "The quick", "The quick brown fox"]
        var previous = ""
        var assembled = ""
        for full in snapshots {
            assembled += ApplePrivateCloud.streamDelta(previous: previous, full: full)
            previous = full
        }
        XCTAssertEqual(assembled, "The quick brown fox")
    }

    func test_streamDelta_modelRewriteFallsBackToFull() {
        // If a snapshot doesn't extend the previous one, emit it whole.
        XCTAssertEqual(ApplePrivateCloud.streamDelta(previous: "abc", full: "xyz"), "xyz")
    }

    // MARK: model-category classification (no PCC field on AssistantModel)

    func test_executionLocation_pccIdRecognized() {
        XCTAssertEqual(ModelExecutionLocation.of(assistantModelID: ApplePrivateCloud.modelID),
                       .applePrivateCloud)
    }

    func test_executionLocation_localModelDefault() {
        XCTAssertEqual(ModelExecutionLocation.of(assistantModelID: "qwen2.5-coder-1.5b"),
                       .localDownloaded)
        XCTAssertFalse(ModelExecutionLocation.localDownloaded.isCloud)
        XCTAssertTrue(ModelExecutionLocation.applePrivateCloud.isCloud)
    }

    // MARK: status gating

    func test_status_canSend() {
        XCTAssertTrue(ApplePCCStatus.ready.canSend)
        XCTAssertTrue(ApplePCCStatus.approachingLimit.canSend)
        XCTAssertFalse(ApplePCCStatus.limitReached.canSend)
        XCTAssertFalse(ApplePCCStatus.unsupportedOS.canSend)
        XCTAssertFalse(ApplePCCStatus.offline.canSend)
    }

    // MARK: automatic reasoning routing

    func test_automaticReasoning_shortConversationUsesLight() {
        XCTAssertEqual(
            ApplePrivateCloud.automaticReasoningLevel(for: "Summarize this in one sentence."),
            .light
        )
    }

    func test_automaticReasoning_codeTaskUsesModerate() {
        XCTAssertEqual(
            ApplePrivateCloud.automaticReasoningLevel(
                for: "Refactor this Swift function to avoid duplicate work."
            ),
            .moderate
        )
    }

    func test_automaticReasoning_complexAnalysisUsesDeep() {
        XCTAssertEqual(
            ApplePrivateCloud.automaticReasoningLevel(
                for: "Find the root cause of this race condition and propose a migration plan."
            ),
            .deep
        )
    }

    func test_resolvedReasoning_explicitChoiceAlwaysWins() {
        XCTAssertEqual(
            ApplePrivateCloud.resolvedReasoningLevel(
                requested: .light,
                prompt: "Develop a detailed threat model and architecture."
            ),
            .light
        )
    }

    func test_requestResolvesAutomaticReasoningWithoutChangingStoredChoice() {
        let request = ApplePCCRequest(
            prompt: "Implement this TypeScript API client.",
            reasoning: .automatic,
            maximumResponseTokens: 1_024,
            temperature: 0.25
        )

        XCTAssertEqual(request.reasoning, .automatic)
        XCTAssertEqual(request.resolvedReasoning, .moderate)
        XCTAssertEqual(request.maximumResponseTokens, 1_024)
        XCTAssertEqual(request.temperature, 0.25)
    }

    // MARK: consent + real-context budgeting

    func test_privacyConsentRequiresCurrentDisclosureVersion() {
        XCTAssertFalse(ApplePrivateCloud.hasCurrentPrivacyConsent(version: 0))
        XCTAssertTrue(
            ApplePrivateCloud.hasCurrentPrivacyConsent(
                version: ApplePrivateCloud.privacyDisclosureVersion
            )
        )
    }

    func test_inputBudgetReservesResponseAndFramingTokens() {
        XCTAssertEqual(
            ApplePrivateCloud.inputBudget(
                contextSize: 8_192,
                maximumResponseTokens: 2_048
            ),
            5_632
        )
    }

    func test_inputBudgetNeverDropsBelowNewestPromptReserve() {
        XCTAssertEqual(
            ApplePrivateCloud.inputBudget(
                contextSize: 1_000,
                maximumResponseTokens: 2_000
            ),
            512
        )
    }

    // MARK: prompt construction

    func test_promptBuilderSeparatesSystemInstructionsFromDialog() {
        let input = ApplePrivateCloudPromptBuilder.build(messages: [
            ChatMessage(role: .system, content: "Be concise."),
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi"),
            ChatMessage(role: .user, content: "Continue"),
        ])

        XCTAssertEqual(input.instructions, "Be concise.")
        XCTAssertEqual(
            input.prompt,
            "User:\nHello\n\nAssistant:\nHi\n\nUser:\nContinue\n\nAssistant:"
        )
    }

    func test_promptBuilderUsesHiddenModelContent() {
        let input = ApplePrivateCloudPromptBuilder.build(messages: [
            ChatMessage(
                role: .user,
                content: "Summarize the attachment",
                modelContent: "Summarize the attachment\n\n[DOCUMENT]\nGrounding text"
            ),
        ])

        XCTAssertTrue(input.prompt.contains("[DOCUMENT]\nGrounding text"))
        XCTAssertFalse(input.prompt.hasPrefix("Summarize the attachment\n\nAssistant:"))
    }

    func test_promptBuilderOmitsEmptyStreamingPlaceholderAndAddsOneCue() {
        let input = ApplePrivateCloudPromptBuilder.build(messages: [
            ChatMessage(role: .user, content: "Question"),
            ChatMessage(role: .assistant, content: "", isStreaming: true),
        ])

        XCTAssertEqual(input.prompt, "User:\nQuestion\n\nAssistant:")
    }

    func test_messageAttributionSurvivesConversationPersistence() {
        let message = ChatMessage(
            role: .assistant,
            content: "Cloud response",
            generationModelID: ApplePrivateCloud.modelID,
            generationExecutionLocation: .applePrivateCloud
        )

        let restored = StoredMessage(message).chatMessage

        XCTAssertEqual(restored.generationModelID, ApplePrivateCloud.modelID)
        XCTAssertEqual(restored.generationExecutionLocation, .applePrivateCloud)
    }

    // MARK: facade is inert on a GA build

    func test_facade_buildAndRuntimeGatesStayConsistent() {
        if ApplePrivateCloud.isCompiledIn {
            // An iOS 27 SDK build may still run on iOS 18–26, where the
            // stricter runtime gate remains false.
            XCTAssertEqual(ApplePrivateCloud.unavailableError, .unsupportedOS)
        } else {
            XCTAssertFalse(ApplePrivateCloud.isSupportedOnCurrentOS)
            XCTAssertEqual(ApplePrivateCloud.unavailableError, .notCompiledIn)
        }
    }

    func test_error_descriptionsDoNotLeakContent() {
        // Sanity: user-facing strings are generic, never echo a prompt.
        XCTAssertEqual(ApplePCCError.quotaExceeded.errorDescription,
                       "Today's Apple Private Cloud usage limit has been reached. Local models remain available.")
        XCTAssertNotNil(ApplePCCError.offline.errorDescription)
    }
}
