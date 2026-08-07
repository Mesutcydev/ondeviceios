import Foundation

// MARK: - Persistent conversation storage
// Saves to ~/Documents/conversations.json via Codable.
// Each conversation is a named list of messages.

final class ConversationStore: ObservableObject {
    static let shared = ConversationStore()

    @Published private(set) var conversations: [StoredConversation] = []

    private let fileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("conversations.json")
    }()

    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var saveTask: Task<Void, Never>?
    private var persistRetryCount = 0

    init() {
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
        load()
    }

    // MARK: - CRUD

    func create(title: String = "New Chat") -> StoredConversation {
        let conv = StoredConversation(title: title)
        conversations.insert(conv, at: 0)
        scheduleSave()
        return conv
    }

    func update(id: UUID, messages: [ChatMessage]) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[idx].messages = messages.map(StoredMessage.init)
        conversations[idx].updatedAt = .now
        if conversations[idx].title == "New Chat", let first = messages.first(where: { $0.role == .user }) {
            conversations[idx].title = String(first.content.prefix(40))
        }
        scheduleSave()
    }

    func delete(id: UUID) {
        conversations.removeAll { $0.id == id }
        // Tombstone the id so an enabled iCloud sync propagates the deletion
        // to CloudKit instead of pulling the remote record back next sync.
        CloudSyncTombstones.mark(id)
        scheduleSave()
    }

    func messages(for id: UUID) -> [ChatMessage] {
        conversations.first(where: { $0.id == id })?.messages.map(\.chatMessage) ?? []
    }

    // Creates a new conversation or updates an existing one by id.
    func saveConversation(
        id: UUID,
        messages: [ChatMessage],
        contextMemory: ConversationContextMemory? = nil,
        assistantModelID: String? = nil
    ) {
        if conversations.contains(where: { $0.id == id }) {
            update(id: id, messages: messages)
            if let idx = conversations.firstIndex(where: { $0.id == id }) {
                conversations[idx].contextMemory = contextMemory
                if let assistantModelID {
                    conversations[idx].assistantModelID = assistantModelID
                }
            }
        } else {
            var conv = StoredConversation()
            conv.id = id
            conv.messages = messages.map(StoredMessage.init)
            conv.contextMemory = contextMemory
            conv.assistantModelID = assistantModelID
            if let first = messages.first(where: { $0.role == .user }) {
                conv.title = String(first.content.prefix(40))
            }
            conversations.insert(conv, at: 0)
        }
        scheduleSave()
    }

    func deleteConversations(at offsets: IndexSet) {
        offsets.map { conversations[$0].id }.forEach { delete(id: $0) }
    }

    /// Used by CloudSyncService to overwrite the local set after a pull.
    func replaceAll(with newSet: [StoredConversation]) {
        conversations = newSet.sorted { $0.updatedAt > $1.updatedAt }
        scheduleSave()
    }

    // MARK: - Persistence

    private func load() {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return }
        let stored: [StoredConversation]
        do {
            let data = try Data(contentsOf: fileURL)
            stored = try decoder.decode([StoredConversation].self, from: data)
        } catch {
            Diagnostics.shared.error(
                "conversation load failed: \(error.localizedDescription)",
                category: "conversations"
            )
            return
        }
        // One-time cleanup: older builds let a thinking model's raw "<think>…"
        // output become the conversation title (visible in History / Home).
        // Strip any reasoning so the list reads cleanly.
        var changed = false
        conversations = stored.map { conv in
            guard conv.title.contains("<think>") else { return conv }
            var c = conv
            c.title = Self.stripReasoningFromTitle(conv.title)
            changed = true
            return c
        }
        if changed { scheduleSave() }
    }

    /// Removes a `<think>…</think>` (or unclosed `<think>…`) block from a
    /// title; falls back to "Chat" when nothing usable remains.
    private static func stripReasoningFromTitle(_ raw: String) -> String {
        var s = raw
        if let close = s.range(of: "</think>", options: [.backwards]) {
            s = String(s[close.upperBound...])
        } else if let open = s.range(of: "<think>") {
            s = String(s[..<open.lowerBound])
        }
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return s.isEmpty ? "Chat" : String(s.prefix(60))
    }

    private func scheduleSave() {
        saveTask?.cancel()
        // Snapshot NOW, on the caller (the main actor, where every create/
        // update/delete/setTitle mutation happens). The debounced write then
        // encodes this immutable copy off-main instead of reading the live
        // @Published `conversations` array — which previously raced concurrent
        // main-actor mutations (a non-Sendable Array read+write data race that
        // can corrupt the COW buffer / crash). Each new mutation re-snapshots
        // and cancels the prior task, so the latest state still wins.
        let snapshot = conversations
        saveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(500))
            guard !Task.isCancelled else { return }
            self?.persist(snapshot)
        }
    }

    private func persist(_ snapshot: [StoredConversation]) {
        do {
            let data = try encoder.encode(snapshot)
            // At-rest encryption: .completeUnlessOpen means the file is encrypted
            // until first unlock after boot, then stays accessible while the
            // device is unlocked. Best compromise between background access
            // (we need to read at launch even from sleep) and protection.
            try data.write(to: fileURL, options: [.atomic, .completeFileProtectionUnlessOpen])
            persistRetryCount = 0
            SpotlightIndexer.reindexAll(snapshot)
        } catch {
            Diagnostics.shared.error(
                "conversation persist failed: \(error.localizedDescription)",
                category: "conversations"
            )
            if persistRetryCount < 2 {
                persistRetryCount += 1
                saveTask = Task { [weak self] in
                    try? await Task.sleep(for: .milliseconds(750))
                    guard !Task.isCancelled else { return }
                    self?.persist(snapshot)
                }
            } else {
                persistRetryCount = 0
                DispatchQueue.main.async {
                    ToastCenter.shared.error(
                        "Couldn't save conversations",
                        detail: "Your latest changes may be lost if the app closes. Try again."
                    )
                }
            }
        }
    }

    /// Cancel the debounce and persist immediately. Call on scene-phase
    /// background so the user's most recent turn isn't lost if the app is
    /// suspended/terminated within the 500 ms save window.
    func flush() {
        saveTask?.cancel()
        persist(conversations)
    }

    /// Clear all in-memory conversations for a privacy wipe. Cancels any
    /// pending debounced save so a stray flush()/scheduleSave can't rewrite
    /// conversations.json — or rebuild the Spotlight index — from memory after
    /// WipeAllDataService has deleted the file and the index on disk.
    func clearAllForWipe() {
        saveTask?.cancel()
        saveTask = nil
        conversations = []
    }

    /// Public title update — used by ConversationTitler.
    func setTitle(_ title: String, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[idx].title = title
        conversations[idx].updatedAt = .now
        scheduleSave()
    }

    /// Persists model-facing long-conversation memory without rewriting the
    /// visible transcript (which may currently contain a streaming message).
    func setContextMemory(_ memory: ConversationContextMemory?, for id: UUID) {
        guard let idx = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[idx].contextMemory = memory
        conversations[idx].updatedAt = .now
        scheduleSave()
    }

    // MARK: - Pagination

    /// Returns the most recent `pageSize` messages for a conversation, plus
    /// a flag indicating whether older messages exist. Use to lazy-load long
    /// chat histories instead of loading 500+ messages on conversation open.
    func recentMessages(for id: UUID, pageSize: Int = 200) -> (messages: [ChatMessage], hasMore: Bool) {
        guard let conv = conversations.first(where: { $0.id == id }) else {
            return ([], false)
        }
        let all = conv.messages.map(\.chatMessage)
        if all.count <= pageSize { return (all, false) }
        return (Array(all.suffix(pageSize)), true)
    }

    /// Returns the next-older window of messages for incremental scroll-back.
    /// `before` is the index from the end of the conversation (so callers
    /// can paginate without keeping cursors).
    func olderMessages(for id: UUID, before: Int, pageSize: Int = 200) -> [ChatMessage] {
        guard let conv = conversations.first(where: { $0.id == id }) else { return [] }
        let all = conv.messages.map(\.chatMessage)
        let endIdx = Swift.max(0, all.count - before)
        let startIdx = Swift.max(0, endIdx - pageSize)
        return Array(all[startIdx..<endIdx])
    }
}

// MARK: - Codable models

struct StoredConversation: Identifiable, Codable {
    var id: UUID
    var title: String
    var messages: [StoredMessage]
    /// Model selected for this thread. Optional keeps older conversation files
    /// decodable; nil means use the user's default assistant model.
    var assistantModelID: String?
    /// Bounded model-facing memory made from turns older than the live context
    /// window. The full transcript remains in `messages` for display/export.
    var contextMemory: ConversationContextMemory?
    var createdAt: Date
    var updatedAt: Date

    init(title: String = "New Chat") {
        self.id = UUID()
        self.title = title
        self.messages = []
        self.assistantModelID = nil
        self.contextMemory = nil
        self.createdAt = .now
        self.updatedAt = .now
    }

    // MARK: - Export

    /// Plain markdown export — pastes nicely into Notes, Obsidian, GitHub, etc.
    var markdownExport: String {
        var out = "# \(title)\n\n"
        let df = DateFormatter()
        df.dateStyle = .medium; df.timeStyle = .short
        out += "_Created \(df.string(from: createdAt)) · "
        out += "Last updated \(df.string(from: updatedAt))_\n\n---\n\n"
        for m in messages where m.role != "system" {
            let heading = m.role == "user" ? "### You" : "### Assistant"
            out += "\(heading)\n\n\(m.content)\n\n"
        }
        return out
    }

    /// Structured JSON export for round-tripping back into the app.
    var jsonExport: Data? {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? encoder.encode(self)
    }

    /// True when any message body matches `query` (case-insensitive),
    /// or when the title matches. Empty query returns true.
    func matches(_ query: String) -> Bool {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if q.isEmpty { return true }
        if title.lowercased().contains(q) { return true }
        return messages.contains { $0.content.lowercased().contains(q) }
    }
}

struct StoredMessage: Codable {
    var id: UUID
    var role: String        // "user" | "assistant" | "system"
    var content: String
    /// Hidden grounding used by the model for later turns. Optional keeps
    /// conversations written by older builds fully decodable.
    var modelContent: String?
    var timestamp: Date
    var generationTokensPerSecond: Double?
    var generationDuration: TimeInterval?
    var generationModelID: String?
    var generationExecutionLocation: ModelExecutionLocation?

    init(_ msg: ChatMessage) {
        self.id = msg.id
        self.role = msg.role.rawValue
        self.content = msg.content
        self.modelContent = msg.modelContent
        self.timestamp = msg.timestamp
        self.generationTokensPerSecond = msg.generationTokensPerSecond
        self.generationDuration = msg.generationDuration
        self.generationModelID = msg.generationModelID
        self.generationExecutionLocation = msg.generationExecutionLocation
    }

    var chatMessage: ChatMessage {
        ChatMessage(
            id: id,
            role: ChatMessage.Role(rawValue: role) ?? .user,
            content: content,
            modelContent: modelContent,
            timestamp: timestamp,
            generationTokensPerSecond: generationTokensPerSecond,
            generationDuration: generationDuration,
            generationModelID: generationModelID,
            generationExecutionLocation: generationExecutionLocation
        )
    }
}

// MARK: - Conversation sharing

enum ConversationShareFormatter {
    /// A clean, portable transcript for the system share sheet.
    ///
    /// System prompts and tool plumbing are deliberately omitted because they
    /// are not part of the visible conversation. Streaming placeholders with
    /// no content are omitted as well.
    static func markdown(title: String, messages: [ChatMessage]) -> String {
        let turns = messages.compactMap { message -> String? in
            let content = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { return nil }

            let heading: String
            switch message.role {
            case .user:
                heading = String(localized: "You")
            case .assistant:
                heading = String(localized: "Assistant")
            case .system, .tool:
                return nil
            }
            return "### \(heading)\n\n\(content)"
        }

        let cleanTitle = title.trimmingCharacters(in: .whitespacesAndNewlines)
        let displayTitle = cleanTitle.isEmpty ? String(localized: "Conversation") : cleanTitle
        return (["# \(displayTitle)"] + turns).joined(separator: "\n\n")
    }
}

// MARK: - Long-conversation context compaction

/// Persistent, model-facing memory for the part of a transcript that no
/// longer fits verbatim. `compactedThroughMessageID` makes compaction
/// incremental: later sends summarize only newly-expired turns.
struct ConversationContextMemory: Codable, Equatable {
    var summary: String
    var compactedThroughMessageID: UUID
    var updatedAt: Date
    var schemaVersion: Int? = 2
    var originalTokenCount: Int? = nil
    var compactedTokenCount: Int? = nil
    var state: ConversationRecoveryState? = nil
}

struct ConversationRecoveryState: Codable, Equatable {
    var goal: String?
    var constraints: [String]
    var decisions: [String]
    var openItems: [String]
    var toolState: [String]
    var artifacts: [String]
}

struct ConversationContextPreparation {
    var messages: [ChatMessage]
    var memory: ConversationContextMemory?
    var didCompact: Bool
}

enum ConversationContextCompactor {
    /// Start before the runtime's hard trim point, leaving room for attachment,
    /// web, tool, and chat-template overhead that is added after this pass.
    static let triggerFraction = 0.72
    static let recentDialogFraction = 0.46
    static let memoryFraction = 0.20

    static func prepare(
        messages: [ChatMessage],
        existingMemory: ConversationContextMemory?,
        maxTokens: Int
    ) -> ConversationContextPreparation {
        guard maxTokens > 0 else {
            return .init(messages: messages, memory: existingMemory, didCompact: false)
        }

        let systemMessages = messages.filter { $0.role == .system }
        let fullDialog = messages.filter { $0.role != .system }

        // If the stored boundary disappeared (for example after importing or
        // editing a transcript), discard the stale memory instead of hiding
        // unrelated messages behind an invalid cursor.
        let validMemory: ConversationContextMemory?
        let activeDialog: [ChatMessage]
        if let existingMemory,
           let boundary = fullDialog.firstIndex(where: {
               $0.id == existingMemory.compactedThroughMessageID
           }) {
            validMemory = existingMemory
            activeDialog = Array(fullDialog.suffix(from: boundary + 1))
        } else {
            validMemory = nil
            activeDialog = fullDialog
        }

        let current = assemble(
            systemMessages: systemMessages,
            memory: validMemory,
            dialog: activeDialog
        )
        let triggerTokens = max(1, Int(Double(maxTokens) * triggerFraction))
        guard estimateTokens(current) > triggerTokens else {
            return .init(messages: current, memory: validMemory, didCompact: false)
        }

        // Keep a generous recent suffix verbatim. Only compact a prefix ending
        // in a completed assistant/tool response so a user/assistant turn is
        // never split across memory and live context.
        let recentBudget = max(128, Int(Double(maxTokens) * recentDialogFraction))
        var recentCost = 0
        var provisionalCut = activeDialog.count
        for index in activeDialog.indices.reversed() {
            let cost = estimateTokens(activeDialog[index])
            if recentCost + cost > recentBudget, index < activeDialog.count - 2 {
                provisionalCut = index + 1
                break
            }
            recentCost += cost
            provisionalCut = index
        }

        guard provisionalCut > 0 else {
            return .init(messages: current, memory: validMemory, didCompact: false)
        }
        let completedCut = stride(from: provisionalCut, through: 1, by: -1)
            .first { cut in
                isSafeBoundary(activeDialog[cut - 1])
            }
        guard let cut = completedCut, activeDialog.count - cut >= 2 else {
            return .init(messages: current, memory: validMemory, didCompact: false)
        }

        let expired = Array(activeDialog[..<cut])
        let recent = Array(activeDialog[cut...])
        guard let boundaryID = expired.last?.id else {
            return .init(messages: current, memory: validMemory, didCompact: false)
        }

        let state = mergedState(previous: validMemory?.state, expired: expired)
        let summary = mergedSummary(
            previous: validMemory?.summary,
            expired: expired,
            state: state,
            maxCharacters: max(600, Int(Double(maxTokens * 4) * memoryFraction))
        )
        var memory = ConversationContextMemory(
            summary: summary,
            compactedThroughMessageID: boundaryID,
            updatedAt: .now,
            schemaVersion: 2,
            originalTokenCount: estimateTokens(current),
            compactedTokenCount: nil,
            state: state
        )
        let compactedMessages = assemble(
            systemMessages: systemMessages,
            memory: memory,
            dialog: recent
        )
        memory.compactedTokenCount = estimateTokens(compactedMessages)
        return .init(
            messages: assemble(
                systemMessages: systemMessages,
                memory: memory,
                dialog: recent
            ),
            memory: memory,
            didCompact: true
        )
    }

    static func estimateTokens(_ messages: [ChatMessage]) -> Int {
        messages.reduce(0) { $0 + estimateTokens($1) }
    }

    private static func estimateTokens(_ message: ChatMessage) -> Int {
        message.contentForModel.count / 4 + 4
    }

    private static func assemble(
        systemMessages: [ChatMessage],
        memory: ConversationContextMemory?,
        dialog: [ChatMessage]
    ) -> [ChatMessage] {
        guard let memory else { return systemMessages + dialog }
        let memoryMessage = ChatMessage(
            id: memory.compactedThroughMessageID,
            role: .system,
            content: """
            [CONVERSATION MEMORY]
            This is a compact record of earlier turns. Use it as background \
            context, preserve stated goals and decisions, and defer to newer \
            verbatim messages if anything conflicts.

            \(memory.summary)
            """
        )
        return systemMessages + [memoryMessage] + dialog
    }

    private static func mergedSummary(
        previous: String?,
        expired: [ChatMessage],
        state: ConversationRecoveryState,
        maxCharacters: Int
    ) -> String {
        var sections: [String] = []
        if let previous, !previous.isEmpty {
            sections.append("Earlier memory:\n\(previous)")
        }
        sections.append("Newly compacted turns:\n" + expired.map(summaryLine).joined(separator: "\n"))
        sections.append(recoveryStateText(state))
        return bounded(sections.joined(separator: "\n\n"), maxCharacters: maxCharacters)
    }

    private static func summaryLine(_ message: ChatMessage) -> String {
        let label: String
        switch message.role {
        case .user: label = "User"
        case .assistant: label = "Assistant"
        case .tool: label = "Tool"
        case .system: label = "System"
        }
        var content = message.contentForModel
            .replacingOccurrences(of: "\\s+", with: " ",
                                  options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let lineLimit = message.role == .tool ? 240 : 700
        if content.count > lineLimit {
            content = bounded(content, maxCharacters: lineLimit)
        }
        if !message.imageThumbnails.isEmpty || message.imageThumbnailData != nil {
            content += " [image attached]"
        }
        return "- \(label): \(content)"
    }

    private static func isSafeBoundary(_ message: ChatMessage) -> Bool {
        if message.role == .tool { return true }
        guard message.role == .assistant else { return false }
        let value = message.contentForModel.lowercased()
        return !value.contains("tool_call")
            && !value.contains("[previous tool calls]")
    }

    private static func mergedState(
        previous: ConversationRecoveryState?,
        expired: [ChatMessage]
    ) -> ConversationRecoveryState {
        var state = previous ?? .init(
            goal: nil,
            constraints: [],
            decisions: [],
            openItems: [],
            toolState: [],
            artifacts: []
        )
        for message in expired {
            let content = message.contentForModel
                .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !content.isEmpty else { continue }
            if state.goal == nil, message.role == .user {
                state.goal = bounded(content, maxCharacters: 360)
            }
            let lower = content.lowercased()
            if lower.contains("must ") || lower.contains("should ")
                || lower.contains("don't ") || lower.contains("do not ") {
                appendUnique(bounded(content, maxCharacters: 320), to: &state.constraints, limit: 8)
            }
            if lower.contains("decided") || lower.contains("selected") || lower.contains("will use") {
                appendUnique(bounded(content, maxCharacters: 320), to: &state.decisions, limit: 8)
            }
            if lower.contains("todo") || lower.contains("remaining")
                || lower.contains("failed") || lower.contains("error") || content.hasSuffix("?") {
                appendUnique(bounded(content, maxCharacters: 320), to: &state.openItems, limit: 8)
            }
            if message.role == .tool || lower.contains("tool_call") || lower.contains("call_id=") {
                appendUnique(bounded(content, maxCharacters: 240), to: &state.toolState, limit: 6)
            }
            for artifact in artifactReferences(in: content) {
                appendUnique(artifact, to: &state.artifacts, limit: 12)
            }
        }
        return state
    }

    private static func recoveryStateText(_ state: ConversationRecoveryState) -> String {
        var lines = ["[RECOVERY STATE v2]"]
        if let goal = state.goal { lines.append("Goal: \(goal)") }
        func add(_ title: String, _ values: [String]) {
            guard !values.isEmpty else { return }
            lines.append("\(title):")
            lines.append(contentsOf: values.map { "- \($0)" })
        }
        add("Constraints", state.constraints)
        add("Decisions", state.decisions)
        add("Open items", state.openItems)
        add("Tool state", state.toolState)
        add("Artifacts", state.artifacts)
        return lines.joined(separator: "\n")
    }

    private static func artifactReferences(in text: String) -> [String] {
        let pattern = #"(?:https?://[^\s<>]+|/(?:[^\s/:]+/)+[^\s<>]+)"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).compactMap {
            Range($0.range, in: text).map {
                String(text[$0]).trimmingCharacters(in: .punctuationCharacters)
            }
        }
    }

    private static func appendUnique(_ value: String, to values: inout [String], limit: Int) {
        guard !value.isEmpty, !values.contains(value) else { return }
        values.append(value)
        if values.count > limit { values.removeFirst(values.count - limit) }
    }

    /// Preserve both the original goal at the beginning and the most recent
    /// compacted decisions at the end when memory itself reaches its cap.
    private static func bounded(_ text: String, maxCharacters: Int) -> String {
        guard text.count > maxCharacters else { return text }
        let marker = "\n[…older detail compacted…]\n"
        let available = max(0, maxCharacters - marker.count)
        let headCount = Int(Double(available) * 0.45)
        let tailCount = available - headCount
        return String(text.prefix(headCount)) + marker + String(text.suffix(tailCount))
    }
}
