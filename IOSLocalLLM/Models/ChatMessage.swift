import Foundation

// MARK: - Chat message

public struct ChatMessage: Identifiable, Codable, Hashable, Sendable {
    /// Native function-call metadata retained across an agent conversation.
    /// Keeping this structured lets MLX render the model's own tool template
    /// and preserves `tool_call_id` for Hermes/OpenCode follow-up turns.
    public struct ToolCallMetadata: Codable, Hashable, Sendable {
        public let id: String
        public let name: String
        public let argumentsJSON: String

        public init(id: String, name: String, argumentsJSON: String) {
            self.id = id
            self.name = name
            self.argumentsJSON = argumentsJSON
        }
    }

    public let id: UUID
    public let role: Role
    public var content: String
    /// Optional model-facing form of this message.
    ///
    /// File contents, web excerpts, Knowledge Base passages, and visual
    /// grounding can be much larger than the text that belongs in the chat
    /// bubble. Keep that hidden context on the turn so follow-up generations
    /// and restored conversations see the same source material.
    public var modelContent: String?
    public let timestamp: Date
    public var isStreaming: Bool
    /// True when the user tapped Stop while this message was streaming.
    /// Rendered as a small "[stopped]" badge in the bubble header.
    public var wasInterrupted: Bool
    /// JPEG-compressed thumbnail (~480 px long edge) the user attached
    /// to this message. Rendered above the text in the user bubble.
    /// Stored as Data so the message stays Codable/Hashable-friendly
    /// for any future persistence layer. The full-resolution image
    /// isn't kept after the OCR/VLM pass — only what's needed to show
    /// the user what they sent.
    public var imageThumbnailData: Data?
    /// Multiple image thumbnails for interleaved vision (feature #11).
    /// Each entry represents one attached image in order. The first image
    /// also populates `imageThumbnailData` for backward compatibility.
    public var imageThumbnails: [ImageAttachment] = []
    /// Token-by-token log probabilities for this assistant message.
    /// Populated when `logprobsEnabled` is true (feature #3).
    public var logprobSummary: LogprobSummary? = nil
    /// Whether the model generated this with structured JSON output mode.
    public var isJSONMode: Bool = false
    /// Final decode throughput reported by the local runtime.
    public var generationTokensPerSecond: Double? = nil
    /// Wall-clock time from creating the assistant turn until completion.
    public var generationDuration: TimeInterval? = nil
    /// Backend identity captured when this assistant turn starts. Optional so
    /// conversations written by older builds decode unchanged.
    public var generationModelID: String? = nil
    public var generationExecutionLocation: ModelExecutionLocation? = nil
    /// Calls produced by an assistant turn, if any.
    public var toolCalls: [ToolCallMetadata]? = nil
    /// Correlates a `.tool` result with the assistant call it answers.
    public var toolCallID: String? = nil

    /// A single image attachment in an interleaved message.
    public struct ImageAttachment: Codable, Hashable, Sendable {
        public let data: Data         // JPEG thumbnail
        public let caption: String?   // optional user caption

        public init(data: Data, caption: String? = nil) {
            self.data = data
            self.caption = caption
        }
    }

    public enum Role: String, Codable, Hashable, Sendable {
        case user
        case assistant
        case system
        case tool
    }

    public init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        modelContent: String? = nil,
        timestamp: Date = .now,
        isStreaming: Bool = false,
        wasInterrupted: Bool = false,
        imageThumbnailData: Data? = nil,
        imageThumbnails: [ImageAttachment] = [],
        logprobSummary: LogprobSummary? = nil,
        isJSONMode: Bool = false,
        generationTokensPerSecond: Double? = nil,
        generationDuration: TimeInterval? = nil,
        generationModelID: String? = nil,
        generationExecutionLocation: ModelExecutionLocation? = nil,
        toolCalls: [ToolCallMetadata]? = nil,
        toolCallID: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.modelContent = modelContent
        self.timestamp = timestamp
        self.isStreaming = isStreaming
        self.wasInterrupted = wasInterrupted
        self.imageThumbnailData = imageThumbnailData
        self.imageThumbnails = imageThumbnails
        self.logprobSummary = logprobSummary
        self.isJSONMode = isJSONMode
        self.generationTokensPerSecond = generationTokensPerSecond
        self.generationDuration = generationDuration
        self.generationModelID = generationModelID
        self.generationExecutionLocation = generationExecutionLocation
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    public static func == (lhs: ChatMessage, rhs: ChatMessage) -> Bool {
        lhs.id == rhs.id &&
        lhs.content == rhs.content &&
        lhs.modelContent == rhs.modelContent &&
        lhs.isStreaming == rhs.isStreaming &&
        lhs.wasInterrupted == rhs.wasInterrupted &&
        lhs.imageThumbnailData == rhs.imageThumbnailData &&
        lhs.imageThumbnails == rhs.imageThumbnails &&
        lhs.logprobSummary == rhs.logprobSummary &&
        lhs.isJSONMode == rhs.isJSONMode &&
        lhs.generationTokensPerSecond == rhs.generationTokensPerSecond &&
        lhs.generationDuration == rhs.generationDuration &&
        lhs.generationModelID == rhs.generationModelID &&
        lhs.generationExecutionLocation == rhs.generationExecutionLocation &&
        lhs.toolCalls == rhs.toolCalls &&
        lhs.toolCallID == rhs.toolCallID
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(content)
        hasher.combine(modelContent)
        hasher.combine(isStreaming)
        hasher.combine(wasInterrupted)
        hasher.combine(imageThumbnailData)
        hasher.combine(imageThumbnails)
        hasher.combine(logprobSummary)
        hasher.combine(isJSONMode)
        hasher.combine(generationTokensPerSecond)
        hasher.combine(generationDuration)
        hasher.combine(generationModelID)
        hasher.combine(generationExecutionLocation)
        hasher.combine(toolCalls)
        hasher.combine(toolCallID)
    }

    /// Content supplied to a model prompt. Views and exports continue to use
    /// `content`, keeping grounding payloads out of the visible transcript.
    public var contentForModel: String {
        modelContent ?? content
    }
}

// MARK: - Qwen3 chat template
// Qwen3 uses ChatML format with optional thinking tags.

extension [ChatMessage] {
    func qwen3Prompt(enableThinking: Bool = false) -> String {
        var parts: [String] = []
        for msg in self {
            switch msg.role {
            case .system:
                parts.append("<|im_start|>system\n\(msg.content)<|im_end|>")
            case .user:
                parts.append("<|im_start|>user\n\(msg.content)<|im_end|>")
            case .assistant:
                parts.append("<|im_start|>assistant\n\(msg.content)<|im_end|>")
            case .tool:
                parts.append("<|im_start|>tool\n\(msg.content)<|im_end|>")
            }
        }
        // Prompt the model to respond; /no_think suppresses chain-of-thought for speed
        let thinkTag = enableThinking ? "" : " /no_think"
        parts.append("<|im_start|>assistant\(thinkTag)\n")
        return parts.joined(separator: "\n")
    }
}
