import XCTest
@testable import IOSLocalLLM

// MARK: - StoredConversationTests
//
// Covers the StoredConversation model: markdown export, JSON round-trip,
// search/matching, and title auto-generation logic from ConversationStore.

final class StoredConversationTests: XCTestCase {

    // MARK: - Initialization defaults

    func test_defaultInit_hasTitleNewChat() {
        let conv = StoredConversation()
        XCTAssertEqual(conv.title, "New Chat")
    }

    func test_defaultInit_messagesEmpty() {
        let conv = StoredConversation()
        XCTAssertTrue(conv.messages.isEmpty)
    }

    func test_defaultInit_usesAppDefaultModel() {
        let conv = StoredConversation()
        XCTAssertNil(conv.assistantModelID)
    }

    func test_defaultInit_datesAreCloseToNow() {
        let conv = StoredConversation()
        let now = Date()
        XCTAssertLessThan(abs(conv.createdAt.timeIntervalSince(now)), 5)
        XCTAssertLessThan(abs(conv.updatedAt.timeIntervalSince(now)), 5)
    }

    // MARK: - Markdown export

    func test_markdownExport_includesTitle() {
        let conv = StoredConversation(title: "My Chat")
        let md = conv.markdownExport
        XCTAssertTrue(md.contains("# My Chat"))
    }

    func test_markdownExport_includesDates() {
        let conv = StoredConversation(title: "Test")
        let md = conv.markdownExport
        XCTAssertTrue(md.contains("Created"))
        XCTAssertTrue(md.contains("Last updated"))
    }

    func test_markdownExport_skipsSystemMessages() {
        var conv = StoredConversation(title: "Test")
        conv.messages = [
            StoredMessage(ChatMessage(role: .system, content: "hidden prompt")),
            StoredMessage(ChatMessage(role: .user, content: "Hello")),
        ]
        let md = conv.markdownExport
        XCTAssertFalse(md.contains("hidden prompt"))
        XCTAssertTrue(md.contains("Hello"))
    }

    func test_storedMessage_preservesGenerationMetrics() {
        let original = ChatMessage(
            role: .assistant,
            content: "Done",
            generationTokensPerSecond: 18.75,
            generationDuration: 4.2
        )

        let restored = StoredMessage(original).chatMessage

        XCTAssertEqual(restored.generationTokensPerSecond, 18.75)
        XCTAssertEqual(restored.generationDuration, 4.2)
    }

    func test_storedMessage_preservesHiddenModelContext() {
        let original = ChatMessage(
            role: .user,
            content: "Summarize the important parts.",
            modelContent: "[ATTACHED FILE]\nFull private source text"
        )

        let restored = StoredMessage(original).chatMessage

        XCTAssertEqual(restored.content, "Summarize the important parts.")
        XCTAssertEqual(
            restored.contentForModel,
            "[ATTACHED FILE]\nFull private source text"
        )
    }

    func test_markdownExport_includesUserAndAssistantLabels() {
        var conv = StoredConversation(title: "Test")
        conv.messages = [
            StoredMessage(ChatMessage(role: .user, content: "Q1")),
            StoredMessage(ChatMessage(role: .assistant, content: "A1")),
        ]
        let md = conv.markdownExport
        XCTAssertTrue(md.contains("### You"))
        XCTAssertTrue(md.contains("### Assistant"))
    }

    func test_shareMarkdown_includesVisibleConversationOnly() {
        let markdown = ConversationShareFormatter.markdown(
            title: "Planning",
            messages: [
                ChatMessage(role: .system, content: "Hidden system prompt"),
                ChatMessage(role: .user, content: "First question"),
                ChatMessage(role: .tool, content: "Hidden tool result"),
                ChatMessage(role: .assistant, content: "First answer"),
            ]
        )

        XCTAssertTrue(markdown.hasPrefix("# Planning"))
        XCTAssertTrue(markdown.contains("### You\n\nFirst question"))
        XCTAssertTrue(markdown.contains("### Assistant\n\nFirst answer"))
        XCTAssertFalse(markdown.contains("Hidden system prompt"))
        XCTAssertFalse(markdown.contains("Hidden tool result"))
    }

    func test_shareMarkdown_omitsEmptyStreamingPlaceholder() {
        let markdown = ConversationShareFormatter.markdown(
            title: "",
            messages: [
                ChatMessage(role: .user, content: "Hello"),
                ChatMessage(role: .assistant, content: " \n", isStreaming: true),
            ]
        )

        XCTAssertEqual(markdown, "# Conversation\n\n### You\n\nHello")
    }

    // MARK: - JSON export

    func test_jsonExport_roundTrips() {
        var conv = StoredConversation(title: "Round Trip Test")
        conv.assistantModelID = "downloaded:example/assistant"
        conv.messages = [
            StoredMessage(ChatMessage(
                role: .user, content: "Hello world",
                timestamp: Date(timeIntervalSince1970: 1_000_000)
            )),
        ]
        guard let data = conv.jsonExport else {
            XCTFail("jsonExport returned nil")
            return
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let decoded = try? decoder.decode(StoredConversation.self,
                                                from: data) else {
            XCTFail("Failed to decode exported JSON")
            return
        }
        XCTAssertEqual(decoded.title, "Round Trip Test")
        XCTAssertEqual(decoded.messages.count, 1)
        XCTAssertEqual(decoded.messages.first?.content, "Hello world")
        XCTAssertEqual(decoded.messages.first?.role, "user")
        XCTAssertEqual(decoded.assistantModelID, "downloaded:example/assistant")
    }

    func test_jsonExport_emptyMessages_isValid() {
        let conv = StoredConversation(title: "Empty")
        let data = conv.jsonExport
        XCTAssertNotNil(data)
        // Should decode back cleanly.
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try? decoder.decode(StoredConversation.self,
                                          from: data!)
        XCTAssertEqual(decoded?.title, "Empty")
        XCTAssertTrue(decoded?.messages.isEmpty ?? false)
    }

    func test_legacyJSON_withoutContextMemory_decodes() throws {
        let id = UUID()
        let json = """
        {
          "id": "\(id.uuidString)",
          "title": "Legacy",
          "messages": [],
          "createdAt": "2001-01-01T00:00:00Z",
          "updatedAt": "2001-01-01T00:00:00Z"
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(
            StoredConversation.self,
            from: Data(json.utf8)
        )
        XCTAssertNil(decoded.contextMemory)
        XCTAssertNil(decoded.assistantModelID)
    }

    // MARK: - Context compaction

    func test_contextCompactor_keepsRecentTurnsAndAddsMemory() {
        let system = ChatMessage(role: .system, content: "Be helpful.")
        let dialog = (0..<8).flatMap { index in
            [
                ChatMessage(
                    role: .user,
                    content: "Question \(index): " + String(repeating: "detail ", count: 80)
                ),
                ChatMessage(
                    role: .assistant,
                    content: "Answer \(index): " + String(repeating: "result ", count: 80)
                ),
            ]
        }
        let prepared = ConversationContextCompactor.prepare(
            messages: [system] + dialog,
            existingMemory: nil,
            maxTokens: 700
        )

        XCTAssertTrue(prepared.didCompact)
        XCTAssertNotNil(prepared.memory)
        XCTAssertTrue(prepared.messages.contains {
            $0.role == .system && $0.content.contains("[CONVERSATION MEMORY]")
        })
        XCTAssertEqual(prepared.messages.last?.id, dialog.last?.id)
        XCTAssertLessThan(prepared.messages.count, dialog.count + 1)
    }

    func test_contextCompactor_usesStoredBoundaryIncrementally() {
        let oldUser = ChatMessage(role: .user, content: "Original goal")
        let oldAssistant = ChatMessage(role: .assistant, content: "Original decision")
        let memory = ConversationContextMemory(
            summary: "- User: Original goal\n- Assistant: Original decision",
            compactedThroughMessageID: oldAssistant.id,
            updatedAt: .now
        )
        let recent = (0..<6).flatMap { index in
            [
                ChatMessage(
                    role: .user,
                    content: "Follow-up \(index) " + String(repeating: "context ", count: 70)
                ),
                ChatMessage(
                    role: .assistant,
                    content: "Response \(index) " + String(repeating: "answer ", count: 70)
                ),
            ]
        }
        let prepared = ConversationContextCompactor.prepare(
            messages: [oldUser, oldAssistant] + recent,
            existingMemory: memory,
            maxTokens: 600
        )

        XCTAssertFalse(prepared.messages.contains { $0.id == oldUser.id })
        XCTAssertEqual(
            prepared.messages.filter {
                $0.content.contains("[CONVERSATION MEMORY]")
            }.count,
            1
        )
        XCTAssertTrue(prepared.memory?.summary.contains("Original goal") ?? false)
        XCTAssertEqual(prepared.messages.last?.id, recent.last?.id)
    }

    func test_contextCompactor_invalidBoundaryFallsBackToTranscript() {
        let messages = [
            ChatMessage(role: .user, content: "Hello"),
            ChatMessage(role: .assistant, content: "Hi"),
        ]
        let stale = ConversationContextMemory(
            summary: "Unrelated stale memory",
            compactedThroughMessageID: UUID(),
            updatedAt: .now
        )
        let prepared = ConversationContextCompactor.prepare(
            messages: messages,
            existingMemory: stale,
            maxTokens: 4_000
        )

        XCTAssertNil(prepared.memory)
        XCTAssertEqual(prepared.messages.map(\.id), messages.map(\.id))
    }

    // MARK: - matches(_:)

    func test_matches_emptyQuery_returnsTrue() {
        let conv = StoredConversation(title: "Test")
        XCTAssertTrue(conv.matches(""))
        XCTAssertTrue(conv.matches("  "))
    }

    func test_matches_titleMatch_returnsTrue() {
        let conv = StoredConversation(title: "Swift Concurrency Chat")
        XCTAssertTrue(conv.matches("swift"))
        XCTAssertTrue(conv.matches("Concurrency"))
    }

    func test_matches_messageBodyMatch_returnsTrue() {
        var conv = StoredConversation(title: "Test")
        conv.messages = [
            StoredMessage(ChatMessage(role: .user, content: "How do I use async/await?")),
        ]
        XCTAssertTrue(conv.matches("async"))
        XCTAssertTrue(conv.matches("await"))
    }

    func test_matches_noMatch_returnsFalse() {
        var conv = StoredConversation(title: "Test")
        conv.messages = [
            StoredMessage(ChatMessage(role: .user, content: "Hello")),
        ]
        XCTAssertFalse(conv.matches("Goodbye"))
    }

    func test_matches_caseInsensitive() {
        var conv = StoredConversation(title: "Test")
        conv.messages = [
            StoredMessage(ChatMessage(role: .user, content: "SWIFT")),
        ]
        XCTAssertTrue(conv.matches("swift"))
        XCTAssertTrue(conv.matches("Swift"))
        XCTAssertTrue(conv.matches("SWIFT"))
    }

    // MARK: - ConversationStore auto-title

    func test_conversationStore_update_autoTitlesFromFirstUserMessage() {
        let store = ConversationStore()
        let conv = store.create(title: "New Chat")

        store.update(id: conv.id, messages: [
            ChatMessage(role: .user,
                        content: "How do I implement a binary search tree?"),
            ChatMessage(role: .assistant,
                        content: "Here's how you do it..."),
        ])

        let updated = store.conversations.first(where: { $0.id == conv.id })
        XCTAssertNotNil(updated)
        // Title should be first 40 chars of the first user message.
        XCTAssertTrue(
            updated?.title.hasPrefix("How do I implement") ?? false,
            "Expected auto-title from first user message, got \(updated?.title ?? "nil")"
        )
    }

    func test_conversationStore_update_preservesManualTitle() {
        let store = ConversationStore()
        let conv = store.create(title: "Custom Title")

        store.update(id: conv.id, messages: [
            ChatMessage(role: .user, content: "Some question"),
        ])

        let updated = store.conversations.first(where: { $0.id == conv.id })
        XCTAssertEqual(updated?.title, "Custom Title",
                       "Manual title should not be overwritten")
    }

    func test_conversationStore_savesConversationModel() {
        let store = ConversationStore()
        let id = UUID()

        store.saveConversation(
            id: id,
            messages: [ChatMessage(role: .user, content: "Hello")],
            assistantModelID: "llama-3.2-1b"
        )

        XCTAssertEqual(
            store.conversations.first(where: { $0.id == id })?.assistantModelID,
            "llama-3.2-1b"
        )
    }

}
