import Darwin
import Foundation
import Network
import UIKit

enum LocalAPIValidation {
    static let portRange = 1024...65535

    static func validPort(_ value: Int) -> UInt16? {
        guard portRange.contains(value) else { return nil }
        return UInt16(value)
    }

    static func modelMatches(_ requested: String, id: String, repoID: String) -> Bool {
        requested == id || requested == repoID
    }

    static func isReachableLANInterface(_ name: String) -> Bool {
        let excludedPrefixes = ["lo", "utun", "ipsec", "awdl", "llw", "pdp_ip"]
        return !excludedPrefixes.contains(where: name.hasPrefix)
    }

    /// HTTP authentication schemes are case-insensitive, while the bearer
    /// token itself is not. Compare the token without an early exit so a
    /// client on the trusted LAN cannot learn key bytes from timing jitter.
    static func isAuthorized(headers: [String: String], key: String) -> Bool {
        let authorization = headers.first {
            $0.key.caseInsensitiveCompare("authorization") == .orderedSame
        }?.value
        if let authorization {
            let parts = authorization.split {
                $0 == " " || $0 == "\t"
            }
            if parts.count == 2,
               parts[0].caseInsensitiveCompare("Bearer") == .orderedSame,
               constantTimeEqual(String(parts[1]), key) {
                return true
            }
        }
        return headers.first {
            $0.key.caseInsensitiveCompare("x-api-key") == .orderedSame
        }.map { constantTimeEqual($0.value, key) } ?? false
    }

    private static func constantTimeEqual(_ lhs: String, _ rhs: String) -> Bool {
        let left = Array(lhs.utf8)
        let right = Array(rhs.utf8)
        let count = max(left.count, right.count)
        var difference = left.count ^ right.count
        for index in 0..<count {
            let leftByte = index < left.count ? left[index] : 0
            let rightByte = index < right.count ? right[index] : 0
            difference |= Int(leftByte ^ rightByte)
        }
        return difference == 0
    }
}

enum LocalAPIKeyStore {
    static let account = "localAPI.bearerKey"
    // Use a second namespace inside the cloned app so its bearer token is
    // independent from both the reference app and any web-tool credentials.
    static let keychainService = "com.mesutcydev.ondevicelas.localapi"

    static func key() -> String {
        if let existing = KeychainStore.get(
            account: account,
            serviceName: keychainService
        ), !existing.isEmpty {
            return existing
        }
        return rotate()
    }

    @discardableResult
    static func rotate() -> String {
        let value = "odl_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
            + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        KeychainStore.set(
            value,
            account: account,
            serviceName: keychainService
        )
        return value
    }
}

enum LocalAPIProtocolError: Error, Equatable {
    case malformed(String)
    case unsupported(String)
    case unknownModel
}

struct LocalAPIToolDefinition: Equatable, Sendable {
    let name: String
    let description: String?
    let parametersJSON: String
}

enum LocalAPIToolChoice: Equatable, Sendable {
    case auto
    case none
    case required
    case function(String)
}

struct LocalAPIToolCall: Equatable, Sendable {
    let id: String
    let name: String
    let argumentsJSON: String
}

enum LocalAPIReasoningPreference: Equatable, Sendable {
    case automatic
    case enabled
    case disabled

    func forceNoThinking(globalEnabled: Bool) -> Bool {
        switch self {
        case .disabled: true
        case .enabled: false
        case .automatic: !globalEnabled
        }
    }
}

struct LocalAPIChatRequest {
    let model: String
    let messages: [ChatMessage]
    let stream: Bool
    let streamIncludeUsage: Bool
    let maxTokens: Int?
    let reasoningPreference: LocalAPIReasoningPreference
    let temperature: Double?
    let topP: Double?
    let tools: [LocalAPIToolDefinition]
    let toolChoice: LocalAPIToolChoice
    let parallelToolCalls: Bool

    static func decodeOpenAI(_ data: Data) throws -> Self {
        let raw = try object(data)
        let tools = try decodeOpenAITools(raw["tools"])
        let toolChoice = try decodeOpenAIToolChoice(raw["tool_choice"], tools: tools)
        guard let model = raw["model"] as? String, !model.isEmpty,
              let rows = raw["messages"] as? [[String: Any]], !rows.isEmpty else {
            throw LocalAPIProtocolError.malformed("model and messages are required")
        }
        return Self(
            model: model,
            messages: try decodeMessages(rows),
            stream: raw["stream"] as? Bool ?? false,
            streamIncludeUsage: ((raw["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false,
            maxTokens: integer(raw["max_completion_tokens"] ?? raw["max_tokens"]),
            reasoningPreference: decodeReasoningPreference(raw),
            temperature: number(raw["temperature"]),
            topP: number(raw["top_p"]),
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: raw["parallel_tool_calls"] as? Bool ?? true
        )
    }

    static func decodeOpenAIResponses(_ data: Data) throws -> Self {
        let raw = try object(data)
        guard let model = raw["model"] as? String, !model.isEmpty,
              raw["input"] != nil else {
            throw LocalAPIProtocolError.malformed("model and input are required")
        }

        let tools = try decodeResponseTools(raw["tools"])
        let toolChoice = try decodeOpenAIResponsesToolChoice(
            raw["tool_choice"],
            tools: tools
        )

        var messages: [ChatMessage] = []
        if let instructions = raw["instructions"] as? String, !instructions.isEmpty {
            messages.append(ChatMessage(role: .system, content: instructions))
        }
        if let input = raw["input"] as? String {
            messages.append(ChatMessage(role: .user, content: input))
        } else if let rows = raw["input"] as? [[String: Any]], !rows.isEmpty {
            messages.append(contentsOf: try decodeResponseInput(rows))
        } else {
            throw LocalAPIProtocolError.unsupported(
                "input must be a string or a non-empty array of text messages."
            )
        }
        return Self(
            model: model,
            messages: messages,
            stream: raw["stream"] as? Bool ?? false,
            streamIncludeUsage: ((raw["stream_options"] as? [String: Any])?["include_usage"] as? Bool) ?? false,
            maxTokens: integer(raw["max_output_tokens"]),
            reasoningPreference: decodeReasoningPreference(raw),
            temperature: number(raw["temperature"]),
            topP: number(raw["top_p"]),
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: raw["parallel_tool_calls"] as? Bool ?? true
        )
    }

    static func decodeOllamaChat(_ data: Data) throws -> Self {
        let raw = try object(data)
        guard let rows = raw["messages"] as? [[String: Any]], !rows.isEmpty else {
            throw LocalAPIProtocolError.malformed("messages are required")
        }
        let model = raw["model"] as? String ?? ""
        let options = raw["options"] as? [String: Any] ?? [:]
        let tools = try decodeOpenAITools(raw["tools"])
        return Self(
            model: model,
            messages: try decodeOllamaMessages(rows),
            stream: raw["stream"] as? Bool ?? true,
            streamIncludeUsage: false,
            maxTokens: integer(options["num_predict"]),
            reasoningPreference: decodeReasoningPreference(raw),
            temperature: number(options["temperature"]),
            topP: number(options["top_p"]),
            tools: tools,
            toolChoice: tools.isEmpty ? .none : .auto,
            parallelToolCalls: raw["parallel_tool_calls"] as? Bool ?? true
        )
    }

    static func decodeOllamaGenerate(_ data: Data) throws -> Self {
        let raw = try object(data)
        guard let model = raw["model"] as? String, !model.isEmpty,
              let prompt = raw["prompt"] as? String else {
            throw LocalAPIProtocolError.malformed("model and prompt are required")
        }
        let options = raw["options"] as? [String: Any] ?? [:]
        var messages: [ChatMessage] = []
        if let system = raw["system"] as? String, !system.isEmpty {
            messages.append(ChatMessage(role: .system, content: system))
        }
        let images = try decodeOllamaImages(raw["images"])
        messages.append(ChatMessage(
            role: .user,
            content: prompt.isEmpty && !images.isEmpty
                ? "Describe the attached image."
                : prompt,
            imageThumbnailData: images.first?.data,
            imageThumbnails: images
        ))
        return Self(
            model: model,
            messages: messages,
            stream: raw["stream"] as? Bool ?? true,
            streamIncludeUsage: false,
            maxTokens: integer(options["num_predict"]),
            reasoningPreference: decodeReasoningPreference(raw),
            temperature: number(options["temperature"]),
            topP: number(options["top_p"]),
            tools: [],
            toolChoice: .none,
            parallelToolCalls: false
        )
    }

    static func decodeAnthropic(_ data: Data) throws -> Self {
        let raw = try object(data)
        guard let model = raw["model"] as? String, !model.isEmpty,
              let rows = raw["messages"] as? [[String: Any]], !rows.isEmpty,
              let maxTokens = integer(raw["max_tokens"]) else {
            throw LocalAPIProtocolError.malformed("model, messages, and max_tokens are required")
        }
        let tools = try decodeAnthropicTools(raw["tools"])
        let toolChoice = try decodeAnthropicToolChoice(raw["tool_choice"], tools: tools)
        var messages: [ChatMessage] = []
        if let system = try decodeAnthropicContent(raw["system"]), !system.isEmpty {
            messages.append(ChatMessage(role: .system, content: system))
        }
        messages.append(contentsOf: try decodeAnthropicMessages(rows))
        return Self(
            model: model,
            messages: messages,
            stream: raw["stream"] as? Bool ?? false,
            streamIncludeUsage: false,
            maxTokens: maxTokens,
            reasoningPreference: decodeReasoningPreference(raw),
            temperature: number(raw["temperature"]),
            topP: number(raw["top_p"]),
            tools: tools,
            toolChoice: toolChoice,
            parallelToolCalls: !((raw["tool_choice"] as? [String: Any])?["disable_parallel_tool_use"] as? Bool ?? false)
        )
    }

    private static func object(_ data: Data) throws -> [String: Any] {
        guard let value = try? JSONSerialization.jsonObject(with: data),
              let object = value as? [String: Any] else {
            throw LocalAPIProtocolError.malformed("request body must be a JSON object")
        }
        return object
    }

    private static func decodeReasoningPreference(
        _ raw: [String: Any]
    ) -> LocalAPIReasoningPreference {
        if let effort = (raw["reasoning_effort"] as? String)?.lowercased() {
            if effort == "none" { return .disabled }
            if ["minimal", "low", "medium", "high"].contains(effort) {
                return .enabled
            }
        }
        if let thinking = raw["thinking"] as? [String: Any],
           let type = (thinking["type"] as? String)?.lowercased() {
            if type == "disabled" || type == "none" { return .disabled }
            if type == "enabled" { return .enabled }
        }

        let template = raw["chat_template_kwargs"] as? [String: Any]
        let options = raw["options"] as? [String: Any]
        let explicitValues: [Bool?] = [
            raw["enable_thinking"] as? Bool,
            raw["think"] as? Bool,
            template?["enable_thinking"] as? Bool,
            options?["enable_thinking"] as? Bool,
            options?["think"] as? Bool
        ]
        if explicitValues.contains(where: { $0 == false }) { return .disabled }
        if explicitValues.contains(where: { $0 == true }) { return .enabled }
        return .automatic
    }

    private static func decodeOpenAITools(_ value: Any?) throws -> [LocalAPIToolDefinition] {
        guard let value else { return [] }
        guard let rows = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.malformed("tools must be an array")
        }
        return try rows.map { row in
            guard row["type"] as? String == "function",
                  let function = row["function"] as? [String: Any],
                  let name = function["name"] as? String,
                  !name.isEmpty else {
                throw LocalAPIProtocolError.malformed(
                    "Each tool must be a named function."
                )
            }
            let parameters = function["parameters"] ?? [
                "type": "object",
                "properties": [:]
            ]
            guard JSONSerialization.isValidJSONObject(parameters),
                  let data = try? JSONSerialization.data(
                    withJSONObject: parameters,
                    options: [.sortedKeys]
                  ),
                  let parametersJSON = String(data: data, encoding: .utf8) else {
                throw LocalAPIProtocolError.malformed(
                    "Tool '\(name)' has an invalid parameters schema."
                )
            }
            return LocalAPIToolDefinition(
                name: name,
                description: function["description"] as? String,
                parametersJSON: parametersJSON
            )
        }
    }

    private static func decodeResponseTools(_ value: Any?) throws -> [LocalAPIToolDefinition] {
        guard let value else { return [] }
        guard let rows = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.malformed("tools must be an array")
        }
        return try rows.map { row in
            guard row["type"] as? String == "function",
                  let name = row["name"] as? String,
                  !name.isEmpty else {
                throw LocalAPIProtocolError.malformed(
                    "Each Responses API tool must be a named function."
                )
            }
            let parameters = row["parameters"] ?? [
                "type": "object",
                "properties": [:]
            ]
            guard JSONSerialization.isValidJSONObject(parameters),
                  let data = try? JSONSerialization.data(
                    withJSONObject: parameters,
                    options: [.sortedKeys]
                  ),
                  let parametersJSON = String(data: data, encoding: .utf8) else {
                throw LocalAPIProtocolError.malformed(
                    "Tool '\(name)' has an invalid parameters schema."
                )
            }
            return LocalAPIToolDefinition(
                name: name,
                description: row["description"] as? String,
                parametersJSON: parametersJSON
            )
        }
    }

    private static func decodeOpenAIResponsesToolChoice(
        _ value: Any?,
        tools: [LocalAPIToolDefinition]
    ) throws -> LocalAPIToolChoice {
        guard let value else { return tools.isEmpty ? .none : .auto }
        if let string = value as? String {
            switch string {
            case "auto": return tools.isEmpty ? .none : .auto
            case "none": return .none
            case "required":
                guard !tools.isEmpty else {
                    throw LocalAPIProtocolError.malformed(
                        "tool_choice requires at least one tool."
                    )
                }
                return .required
            default:
                throw LocalAPIProtocolError.malformed(
                    "Unsupported Responses API tool_choice '\(string)'."
                )
            }
        }
        guard let object = value as? [String: Any],
              object["type"] as? String == "function",
              let name = object["name"] as? String,
              !name.isEmpty else {
            throw LocalAPIProtocolError.malformed(
                "Responses API tool_choice must be auto, none, required, or a named function."
            )
        }
        guard tools.contains(where: { $0.name == name }) else {
            throw LocalAPIProtocolError.malformed(
                "tool_choice references unknown function '\(name)'."
            )
        }
        return .function(name)
    }

    private static func decodeAnthropicTools(_ value: Any?) throws -> [LocalAPIToolDefinition] {
        guard let value else { return [] }
        guard let rows = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.malformed("tools must be an array")
        }
        return try rows.map { row in
            guard let name = row["name"] as? String, !name.isEmpty else {
                throw LocalAPIProtocolError.malformed(
                    "Each Anthropic tool must have a name."
                )
            }
            let schema = row["input_schema"] ?? [
                "type": "object",
                "properties": [:]
            ]
            guard JSONSerialization.isValidJSONObject(schema),
                  let data = try? JSONSerialization.data(
                    withJSONObject: schema,
                    options: [.sortedKeys]
                  ),
                  let parametersJSON = String(data: data, encoding: .utf8) else {
                throw LocalAPIProtocolError.malformed(
                    "Tool '\(name)' has an invalid input_schema."
                )
            }
            return LocalAPIToolDefinition(
                name: name,
                description: row["description"] as? String,
                parametersJSON: parametersJSON
            )
        }
    }

    private static func decodeAnthropicToolChoice(
        _ value: Any?,
        tools: [LocalAPIToolDefinition]
    ) throws -> LocalAPIToolChoice {
        guard let value else { return tools.isEmpty ? .none : .auto }
        guard let object = value as? [String: Any],
              let type = object["type"] as? String else {
            throw LocalAPIProtocolError.malformed(
                "Anthropic tool_choice must be an object."
            )
        }
        let choice: LocalAPIToolChoice
        switch type {
        case "auto": choice = .auto
        case "none": choice = .none
        case "any": choice = .required
        case "tool":
            guard let name = object["name"] as? String, !name.isEmpty else {
                throw LocalAPIProtocolError.malformed(
                    "A tool tool_choice requires a name."
                )
            }
            choice = .function(name)
        default:
            throw LocalAPIProtocolError.malformed(
                "Unsupported Anthropic tool_choice type '\(type)'."
            )
        }
        if case .function(let name) = choice,
           !tools.contains(where: { $0.name == name }) {
            throw LocalAPIProtocolError.malformed(
                "tool_choice references unknown tool '\(name)'."
            )
        }
        if tools.isEmpty, choice != .none {
            throw LocalAPIProtocolError.malformed(
                "tool_choice requires at least one tool."
            )
        }
        return choice
    }

    private static func decodeOpenAIToolChoice(
        _ value: Any?,
        tools: [LocalAPIToolDefinition]
    ) throws -> LocalAPIToolChoice {
        guard let value else { return tools.isEmpty ? .none : .auto }
        let choice: LocalAPIToolChoice
        if let string = value as? String {
            switch string {
            case "auto": choice = .auto
            case "none": choice = .none
            case "required": choice = .required
            default:
                throw LocalAPIProtocolError.malformed(
                    "tool_choice must be auto, none, required, or a named function."
                )
            }
        } else if let object = value as? [String: Any],
                  object["type"] as? String == "function",
                  let function = object["function"] as? [String: Any],
                  let name = function["name"] as? String,
                  !name.isEmpty {
            choice = .function(name)
        } else {
            throw LocalAPIProtocolError.malformed(
                "tool_choice must be auto, none, required, or a named function."
            )
        }

        switch choice {
        case .auto where tools.isEmpty, .required where tools.isEmpty:
            throw LocalAPIProtocolError.malformed(
                "tool_choice requires at least one tool."
            )
        case .function(let name) where !tools.contains(where: { $0.name == name }):
            throw LocalAPIProtocolError.malformed(
                "tool_choice references unknown function '\(name)'."
            )
        default:
            return choice
        }
    }

    private static func decodeMessages(_ rows: [[String: Any]]) throws -> [ChatMessage] {
        try rows.map { row in
            guard let roleString = row["role"] as? String else {
                throw LocalAPIProtocolError.malformed("Every message must include a role.")
            }
            switch roleString {
            case "system", "developer":
                guard let content = try decodeOpenAIContent(row["content"]) else {
                    throw LocalAPIProtocolError.unsupported(
                        "System message content must be text or an array of text parts."
                    )
                }
                return ChatMessage(role: .system, content: content)
            case "user":
                guard let decoded = try decodeOpenAIUserContent(row["content"]) else {
                    throw LocalAPIProtocolError.unsupported(
                        "User message content must contain text or an embedded image."
                    )
                }
                return ChatMessage(
                    role: .user,
                    content: decoded.text,
                    imageThumbnailData: decoded.images.first?.data,
                    imageThumbnails: decoded.images
                )
            case "assistant":
                let content = try decodeOpenAIContent(row["content"])
                let calls = row["tool_calls"] as? [[String: Any]] ?? []
                guard content?.isEmpty == false || !calls.isEmpty else {
                    throw LocalAPIProtocolError.unsupported(
                        "Assistant messages must contain text or tool_calls."
                    )
                }
                let metadata = try calls.map { call -> ChatMessage.ToolCallMetadata in
                    guard let function = call["function"] as? [String: Any],
                          let name = function["name"] as? String,
                          let arguments = encodedJSONObject(
                            function["arguments"] ?? function["parameters"]
                          ) else {
                        throw LocalAPIProtocolError.malformed(
                            "Assistant tool_calls must contain a function name and arguments."
                        )
                    }
                    let id = call["id"] as? String
                        ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
                    return .init(id: id, name: name, argumentsJSON: arguments)
                }
                let summaries = metadata.map {
                    "call_id=\($0.id) name=\($0.name) arguments=\($0.argumentsJSON)"
                }
                return ChatMessage(
                    role: .assistant,
                    content: content ?? "",
                    modelContent: summaries.isEmpty
                        ? nil
                        : ([content].compactMap { $0 }.filter { !$0.isEmpty }
                            + ["[Previous tool calls]\n" + summaries.joined(separator: "\n")])
                            .joined(separator: "\n\n"),
                    toolCalls: metadata.isEmpty ? nil : metadata
                )
            case "tool":
                guard let content = try decodeOpenAIContent(row["content"]) else {
                    throw LocalAPIProtocolError.unsupported(
                        "Tool message content must be text or an array of text parts."
                    )
                }
                let id = row["tool_call_id"] as? String ?? "unknown"
                return ChatMessage(
                    role: .tool,
                    content: content,
                    modelContent: "[Tool result for call_id=\(id)]\n\(content)",
                    toolCallID: id
                )
            default:
                throw LocalAPIProtocolError.unsupported("Message role '\(roleString)' is not supported.")
            }
        }
    }

    private static func decodeOpenAIContent(_ value: Any?) throws -> String? {
        guard let value, !(value is NSNull) else { return nil }
        if let text = value as? String { return text }
        guard let blocks = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.unsupported(
                "Message content must be text or an array of text parts."
            )
        }
        return try blocks.map { block in
            let type = block["type"] as? String
            guard ["text", "input_text", "output_text"].contains(type),
                  let text = block["text"] as? String else {
                throw LocalAPIProtocolError.unsupported(
                    "Only text content parts are supported in chat messages."
                )
            }
            return text
        }.joined(separator: "\n")
    }

    private struct DecodedUserContent {
        var text: String
        var images: [ChatMessage.ImageAttachment]
    }

    /// Decode the multimodal subset shared by OpenAI Chat Completions and
    /// Responses. Network access remains explicit: the local server accepts
    /// embedded data URLs/base64, but never fetches an arbitrary remote URL
    /// merely because a LAN client placed it in a request.
    private static func decodeOpenAIUserContent(_ value: Any?) throws -> DecodedUserContent? {
        guard let value, !(value is NSNull) else { return nil }
        if let text = value as? String {
            return .init(text: text, images: [])
        }
        guard let blocks = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.unsupported(
                "User content must be text or an array of text/image parts."
            )
        }

        var text: [String] = []
        var images: [ChatMessage.ImageAttachment] = []
        for block in blocks {
            switch block["type"] as? String {
            case "text", "input_text", "output_text":
                if let value = block["text"] as? String, !value.isEmpty {
                    text.append(value)
                }
            case "image_url", "input_image", "image":
                let source: Any? = {
                    if let row = block["image_url"] as? [String: Any] {
                        return row["url"]
                    }
                    return block["image_url"] ?? block["image"] ?? block["data"]
                }()
                images.append(.init(data: try decodeEmbeddedImage(source)))
            default:
                throw LocalAPIProtocolError.unsupported(
                    "Only text and embedded image content parts are supported."
                )
            }
        }
        try validateImageCount(images.count)
        guard !text.isEmpty || !images.isEmpty else { return nil }
        return .init(
            text: text.isEmpty ? "Describe the attached image." : text.joined(separator: "\n"),
            images: images
        )
    }

    private static func decodeOllamaMessages(_ rows: [[String: Any]]) throws -> [ChatMessage] {
        var messages = try decodeMessages(rows)
        for index in rows.indices where index < messages.count {
            guard messages[index].role == .user else { continue }
            let images = try decodeOllamaImages(rows[index]["images"])
            guard !images.isEmpty else { continue }
            messages[index].imageThumbnailData = images.first?.data
            messages[index].imageThumbnails = images
            if messages[index].content.isEmpty {
                messages[index].content = "Describe the attached image."
            }
        }
        return messages
    }

    private static func decodeOllamaImages(_ value: Any?) throws -> [ChatMessage.ImageAttachment] {
        guard let value else { return [] }
        guard let encoded = value as? [String] else {
            throw LocalAPIProtocolError.malformed("Ollama images must be an array of base64 strings.")
        }
        try validateImageCount(encoded.count)
        return try encoded.map { .init(data: try decodeEmbeddedImage($0)) }
    }

    private static let maximumImagesPerMessage = 4
    private static let maximumEmbeddedImageBytes = 12 * 1_024 * 1_024

    private static func validateImageCount(_ count: Int) throws {
        guard count <= maximumImagesPerMessage else {
            throw LocalAPIProtocolError.unsupported(
                "At most \(maximumImagesPerMessage) images are supported per message."
            )
        }
    }

    private static func decodeEmbeddedImage(_ value: Any?) throws -> Data {
        guard let source = value as? String, !source.isEmpty else {
            throw LocalAPIProtocolError.malformed("Image content is missing its base64 data.")
        }

        let base64: Substring
        if source.lowercased().hasPrefix("data:") {
            guard let comma = source.firstIndex(of: ",") else {
                throw LocalAPIProtocolError.malformed("Image data URL is malformed.")
            }
            let metadata = source[..<comma].lowercased()
            guard metadata.hasPrefix("data:image/"), metadata.contains(";base64") else {
                throw LocalAPIProtocolError.unsupported(
                    "Only base64-encoded image data URLs are supported."
                )
            }
            base64 = source[source.index(after: comma)...]
        } else if source.lowercased().hasPrefix("http://")
                    || source.lowercased().hasPrefix("https://") {
            throw LocalAPIProtocolError.unsupported(
                "Remote image URLs are not fetched by this local server. Send a data URL or base64 image instead."
            )
        } else {
            base64 = Substring(source)
        }

        // Base64 expands bytes by roughly 4/3. Reject before allocating a
        // potentially hostile LAN payload, then verify the decoded image.
        guard base64.utf8.count <= (maximumEmbeddedImageBytes * 4 / 3) + 8,
              let data = Data(base64Encoded: String(base64), options: [.ignoreUnknownCharacters]),
              !data.isEmpty,
              data.count <= maximumEmbeddedImageBytes,
              let image = UIImage(data: data),
              image.size.width > 0,
              image.size.height > 0 else {
            throw LocalAPIProtocolError.malformed(
                "Embedded image is invalid or exceeds the 12 MB limit."
            )
        }
        return data
    }

    private static func decodeResponseInput(_ rows: [[String: Any]]) throws -> [ChatMessage] {
        try rows.map { row in
            switch row["type"] as? String {
            case nil, "message":
                guard let roleString = row["role"] as? String else {
                    throw LocalAPIProtocolError.unsupported(
                        "Responses API message items must include a role."
                    )
                }
                let role: ChatMessage.Role
                switch roleString {
                case "developer", "system": role = .system
                case "user": role = .user
                case "assistant": role = .assistant
                default:
                    throw LocalAPIProtocolError.unsupported("Message role '\(roleString)' is not supported.")
                }
                if role == .user {
                    guard let decoded = try decodeOpenAIUserContent(row["content"]) else {
                        throw LocalAPIProtocolError.unsupported(
                            "Responses API user messages must contain text or an embedded image."
                        )
                    }
                    return ChatMessage(
                        role: .user,
                        content: decoded.text,
                        imageThumbnailData: decoded.images.first?.data,
                        imageThumbnails: decoded.images
                    )
                }
                guard let content = try decodeResponseContent(row["content"]),
                      !content.isEmpty else {
                    throw LocalAPIProtocolError.unsupported(
                        "Responses API system and assistant messages must contain text."
                    )
                }
                return ChatMessage(role: role, content: content)
            case "function_call":
                guard let callID = row["call_id"] as? String,
                      let name = row["name"] as? String,
                      !callID.isEmpty, !name.isEmpty,
                      let arguments = encodedJSONObject(row["arguments"]) else {
                    throw LocalAPIProtocolError.malformed(
                        "Responses API function_call items require call_id, name, and JSON arguments."
                    )
                }
                return ChatMessage(
                    role: .assistant,
                    content: "",
                    modelContent: "[Previous tool call]\ncall_id=\(callID) name=\(name) arguments=\(arguments)",
                    toolCalls: [.init(id: callID, name: name, argumentsJSON: arguments)]
                )
            case "function_call_output":
                guard let callID = row["call_id"] as? String,
                      !callID.isEmpty,
                      let output = encodedResponseToolOutput(row["output"]) else {
                    throw LocalAPIProtocolError.malformed(
                        "Responses API function_call_output items require call_id and output."
                    )
                }
                return ChatMessage(
                    role: .tool,
                    content: output,
                    modelContent: "[Tool result for call_id=\(callID)]\n\(output)",
                    toolCallID: callID
                )
            default:
                throw LocalAPIProtocolError.unsupported(
                    "Unsupported Responses API input item."
                )
            }
        }
    }

    private static func encodedResponseToolOutput(_ value: Any?) -> String? {
        if let text = value as? String { return text }
        guard let value,
              JSONSerialization.isValidJSONObject(value),
              let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    private static func decodeResponseContent(_ value: Any?) throws -> String? {
        if let text = value as? String { return text }
        guard let blocks = value as? [[String: Any]] else { return nil }
        return try blocks.map { block in
            let type = block["type"] as? String
            guard ["input_text", "output_text", "text"].contains(type),
                  let text = block["text"] as? String else {
                throw LocalAPIProtocolError.unsupported(
                    "Only input_text and output_text content blocks are supported."
                )
            }
            return text
        }.joined(separator: "\n")
    }

    private static func decodeAnthropicContent(_ value: Any?) throws -> String? {
        guard let value else { return nil }
        if let text = value as? String { return text }
        guard let blocks = value as? [[String: Any]] else {
            throw LocalAPIProtocolError.unsupported("Only text content is supported.")
        }
        return try blocks.map { block in
            guard block["type"] as? String == "text",
                  let text = block["text"] as? String else {
                throw LocalAPIProtocolError.unsupported("Only Anthropic text content blocks are supported.")
            }
            return text
        }.joined(separator: "\n")
    }

    private static func decodeAnthropicMessages(
        _ rows: [[String: Any]]
    ) throws -> [ChatMessage] {
        var messages: [ChatMessage] = []
        for row in rows {
            guard let role = row["role"] as? String else {
                throw LocalAPIProtocolError.malformed(
                    "Every Anthropic message must include a role."
                )
            }
            if let text = row["content"] as? String {
                guard !text.isEmpty else {
                    throw LocalAPIProtocolError.unsupported(
                        "Anthropic messages must not be empty."
                    )
                }
                switch role {
                case "user": messages.append(ChatMessage(role: .user, content: text))
                case "assistant": messages.append(ChatMessage(role: .assistant, content: text))
                default:
                    throw LocalAPIProtocolError.unsupported(
                        "Message role '\(role)' is not supported."
                    )
                }
                continue
            }

            guard let blocks = row["content"] as? [[String: Any]], !blocks.isEmpty else {
                throw LocalAPIProtocolError.unsupported(
                    "Anthropic messages must contain text, tool_use, or tool_result blocks."
                )
            }
            switch role {
            case "assistant":
                var text: [String] = []
                var calls: [ChatMessage.ToolCallMetadata] = []
                for block in blocks {
                    switch block["type"] as? String {
                    case "text":
                        if let value = block["text"] as? String, !value.isEmpty {
                            text.append(value)
                        }
                    case "tool_use":
                        guard let id = block["id"] as? String,
                              let name = block["name"] as? String,
                              !id.isEmpty, !name.isEmpty,
                              let input = encodedJSONObject(block["input"]) else {
                            throw LocalAPIProtocolError.malformed(
                                "Anthropic tool_use requires id, name, and input."
                            )
                        }
                        calls.append(.init(id: id, name: name, argumentsJSON: input))
                    default:
                        throw LocalAPIProtocolError.unsupported(
                            "Unsupported Anthropic assistant content block."
                        )
                    }
                }
                guard !text.isEmpty || !calls.isEmpty else {
                    throw LocalAPIProtocolError.unsupported(
                        "Anthropic assistant messages must contain text or tool_use."
                    )
                }
                let visible = text.joined(separator: "\n")
                let summaries = calls.map {
                    "call_id=\($0.id) name=\($0.name) arguments=\($0.argumentsJSON)"
                }
                messages.append(ChatMessage(
                    role: .assistant,
                    content: visible,
                    modelContent: summaries.isEmpty
                        ? nil
                        : ([visible].filter { !$0.isEmpty }
                            + ["[Previous tool calls]\n" + summaries.joined(separator: "\n")])
                            .joined(separator: "\n\n"),
                    toolCalls: calls.isEmpty ? nil : calls
                ))
            case "user":
                var pendingText: [String] = []
                var pendingImages: [ChatMessage.ImageAttachment] = []
                func appendPendingUser() throws {
                    guard !pendingText.isEmpty || !pendingImages.isEmpty else { return }
                    try validateImageCount(pendingImages.count)
                    messages.append(ChatMessage(
                        role: .user,
                        content: pendingText.isEmpty
                            ? "Describe the attached image."
                            : pendingText.joined(separator: "\n"),
                        imageThumbnailData: pendingImages.first?.data,
                        imageThumbnails: pendingImages
                    ))
                    pendingText.removeAll(keepingCapacity: true)
                    pendingImages.removeAll(keepingCapacity: true)
                }
                for block in blocks {
                    switch block["type"] as? String {
                    case "text":
                        if let value = block["text"] as? String, !value.isEmpty {
                            pendingText.append(value)
                        }
                    case "image":
                        guard let source = block["source"] as? [String: Any] else {
                            throw LocalAPIProtocolError.malformed(
                                "Anthropic image blocks require a source."
                            )
                        }
                        switch source["type"] as? String {
                        case "base64":
                            pendingImages.append(.init(
                                data: try decodeEmbeddedImage(source["data"])
                            ))
                        case "url":
                            // Reuse the shared decoder so HTTP(S) receives the
                            // same explicit local-only refusal as OpenAI.
                            pendingImages.append(.init(
                                data: try decodeEmbeddedImage(source["url"])
                            ))
                        default:
                            throw LocalAPIProtocolError.unsupported(
                                "Anthropic images must use a base64 source."
                            )
                        }
                    case "tool_result":
                        try appendPendingUser()
                        guard let id = block["tool_use_id"] as? String,
                              !id.isEmpty,
                              let output = try decodeAnthropicContent(block["content"]) else {
                            throw LocalAPIProtocolError.malformed(
                                "Anthropic tool_result requires tool_use_id and content."
                            )
                        }
                        messages.append(ChatMessage(
                            role: .tool,
                            content: output,
                            modelContent: "[Tool result for call_id=\(id)]\n\(output)",
                            toolCallID: id
                        ))
                    default:
                        throw LocalAPIProtocolError.unsupported(
                            "Unsupported Anthropic user content block."
                        )
                    }
                }
                try appendPendingUser()
            default:
                throw LocalAPIProtocolError.unsupported(
                    "Message role '\(role)' is not supported."
                )
            }
        }
        return messages
    }

    private static func decodeAnthropicMessageContent(_ value: Any?) throws -> String? {
        if let text = value as? String { return text }
        guard let blocks = value as? [[String: Any]] else { return nil }
        return try blocks.compactMap { block -> String? in
            switch block["type"] as? String {
            case "text":
                return block["text"] as? String
            case "tool_use":
                guard let id = block["id"] as? String,
                      let name = block["name"] as? String,
                      let input = encodedJSONObject(block["input"]) else {
                    throw LocalAPIProtocolError.malformed(
                        "Anthropic tool_use requires id, name, and input."
                    )
                }
                return "[Previous tool call]\ncall_id=\(id) name=\(name) arguments=\(input)"
            case "tool_result":
                guard let id = block["tool_use_id"] as? String,
                      let content = try decodeAnthropicContent(block["content"]) else {
                    throw LocalAPIProtocolError.malformed(
                        "Anthropic tool_result requires tool_use_id and content."
                    )
                }
                return "[Tool result available for call_id=\(id)]\n\(content)"
            default:
                throw LocalAPIProtocolError.unsupported(
                    "Unsupported Anthropic message content block."
                )
            }
        }.joined(separator: "\n\n")
    }

    private static func encodedJSONObject(_ value: Any?) -> String? {
        if let string = value as? String {
            guard let data = string.data(using: .utf8),
                  (try? JSONSerialization.jsonObject(with: data)) != nil else {
                return nil
            }
            return string
        }
        let object = value ?? [:]
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]
              ) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func integer(_ value: Any?) -> Int? {
        (value as? NSNumber)?.intValue
    }

    private static func number(_ value: Any?) -> Double? {
        (value as? NSNumber)?.doubleValue
    }
}

struct LocalAPIReasoningResult: Equatable, Sendable {
    let reasoning: String
    let content: String
}

struct LocalAPIReasoningDelta: Equatable, Sendable {
    let reasoning: String
    let content: String

    var isEmpty: Bool { reasoning.isEmpty && content.isEmpty }
}

struct LocalAPIReasoningFilter {
    private static let openingTags = [
        "<think>", "<thinking>", "<reasoning>", "<|think|>", "<|thinking|>"
    ]
    private static let closingTags = [
        "</think>", "</thinking>", "</reasoning>", "<|end_think|>", "<|end_thinking|>"
    ]
    private static let implicitOpening = "Thinking Process:"

    /// Text held back while a later tag could still reclassify it as CoT.
    /// Bounded on the streaming path — see `speculationLimit`.
    private static let speculationLimit = 64

    private var pending = ""
    /// Hold outside-reasoning text until a tag boundary confirms whether it
    /// was CoT (orphaned `</think>`) or visible answer.
    private var speculativeContent = ""
    private var insideReasoning = false
    private var decidingImplicitOpening = true
    private var reasoning = ""
    private var emittedReasoningCount = 0
    private var assumedPrefilledOpening = false
    private var sawClosingTag = false
    private var speculationClosed = false
    /// `parse` holds the whole completion already, so it can wait for a tag
    /// that proves how to classify the text and still promote a no-tag
    /// completion to `content`. A live stream cannot: text withheld for that
    /// proof is text the client does not see until generation ends, which is
    /// the whole answer arriving in one delta. Streaming commits as it goes.
    private let defersClassification: Bool

    /// Qwen3 / similar chat templates pre-fill the opening `<think>` token
    /// when thinking is enabled, so the model emits reasoning text plus a
    /// closing tag and no opening one. Pass `prefilledOpening` only when the
    /// template really did that — assuming it for a no-thinking completion
    /// makes a plain answer look like unterminated CoT.
    init(prefilledOpening: Bool = false) {
        self.init(prefilledOpening: prefilledOpening, defersClassification: false)
    }

    private init(prefilledOpening: Bool, defersClassification: Bool) {
        self.defersClassification = defersClassification
        if prefilledOpening {
            insideReasoning = true
            decidingImplicitOpening = false
            assumedPrefilledOpening = true
        }
    }

    /// Reasoning is withheld only until a closing tag confirms it, and only
    /// when the caller can afford to wait for that confirmation.
    private var holdsReasoning: Bool {
        defersClassification && assumedPrefilledOpening && !sawClosingTag
    }

    mutating func consume(_ chunk: String) -> LocalAPIReasoningDelta {
        pending += chunk
        var visible = ""

        while true {
            if insideReasoning {
                // Whichever marker appears first decides: a redundant opening
                // tag echoed mid-block is dropped, while the closing tag ends
                // the reasoning section. Checking one tag class before the
                // other can swallow the close into `reasoning_content`.
                let openingRange = firstTag(in: pending, tags: Self.openingTags)
                let closingRange = firstTag(in: pending, tags: Self.closingTags)
                if let openingRange,
                   closingRange == nil || openingRange.lowerBound <= closingRange!.lowerBound {
                    appendReasoning(String(pending[..<openingRange.lowerBound]))
                    pending = String(pending[openingRange.upperBound...])
                    continue
                }
                if let closingRange {
                    appendReasoning(String(pending[..<closingRange.lowerBound]))
                    pending = String(pending[closingRange.upperBound...])
                    insideReasoning = false
                    sawClosingTag = true
                    continue
                }
                let keepCount = longestSuffixPrefixLength(
                    in: pending,
                    tags: Self.closingTags + Self.openingTags
                )
                if keepCount == 0 {
                    appendReasoning(pending)
                    pending = ""
                } else {
                    appendReasoning(String(pending.dropLast(keepCount)))
                    pending = String(pending.suffix(keepCount))
                }
                break
            }

            if decidingImplicitOpening {
                switch implicitOpeningDecision(for: pending) {
                case .waiting:
                    break
                case .matched(let remainder):
                    pending = remainder
                    decidingImplicitOpening = false
                    insideReasoning = true
                    // Explicit "Thinking Process:" means reasoning has started.
                    continue
                case .notMatched:
                    decidingImplicitOpening = false
                }
                if decidingImplicitOpening { break }
            }

            let openingRange = firstTag(in: pending, tags: Self.openingTags)
            let closingRange = firstTag(in: pending, tags: Self.closingTags)
            if let openingRange,
               closingRange == nil || openingRange.lowerBound <= closingRange!.lowerBound {
                visible += speculativeContent
                speculativeContent = ""
                visible += String(pending[..<openingRange.lowerBound])
                pending = String(pending[openingRange.upperBound...])
                insideReasoning = true
                continue
            }

            // Orphaned closing tag (template pre-filled the open, or the
            // model skipped it). Treat everything before the close as
            // reasoning — same contract as `parse`'s post-pass. Speculative
            // buffering is what makes this work on a live token stream.
            if let closingRange {
                appendReasoning(speculativeContent + String(pending[..<closingRange.lowerBound]))
                speculativeContent = ""
                pending = String(pending[closingRange.upperBound...])
                insideReasoning = false
                sawClosingTag = true
                continue
            }

            let keepCount = longestSuffixPrefixLength(
                in: pending,
                tags: Self.openingTags + Self.closingTags
            )
            if keepCount == 0 {
                speculativeContent += pending
                pending = ""
            } else {
                speculativeContent += String(pending.dropLast(keepCount))
                pending = String(pending.suffix(keepCount))
            }
            // An orphaned close only reclassifies text that is still held, and
            // on a live stream holding is what stalls the client. Give the tag
            // a short window to appear, then commit the text as answer and
            // stop speculating — re-arming the window would throttle every
            // later token into the same 64-character batches.
            // Past a closing tag there is no orphaned-close case left to catch,
            // so the answer phase streams from its first token.
            if !defersClassification,
               speculationClosed || sawClosingTag
                || speculativeContent.count > Self.speculationLimit {
                visible += speculativeContent
                speculativeContent = ""
                speculationClosed = true
            }
            break
        }
        return LocalAPIReasoningDelta(
            reasoning: takeReasoningDelta(),
            content: visible
        )
    }

    mutating func finish() -> LocalAPIReasoningDelta {
        if insideReasoning {
            appendReasoning(pending)
            pending = ""
            // Prefilled open but the model never closed: treat the held text
            // as the final answer (thinking disabled / no-CoT completion).
            // Only reachable while the reasoning was still held — a stream has
            // already emitted it and cannot recall it.
            if holdsReasoning {
                let promoted = (speculativeContent + reasoning)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                speculativeContent = ""
                reasoning = ""
                emittedReasoningCount = 0
                insideReasoning = false
                return LocalAPIReasoningDelta(reasoning: "", content: promoted)
            }
            return LocalAPIReasoningDelta(
                reasoning: takeReasoningDelta(),
                content: ""
            )
        }
        let result = speculativeContent + pending
        speculativeContent = ""
        pending = ""
        return LocalAPIReasoningDelta(
            reasoning: takeReasoningDelta(),
            content: result
        )
    }

    var reasoningText: String {
        reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func parse(
        _ text: String,
        prefilledOpening: Bool = false
    ) -> LocalAPIReasoningResult {
        var filter = Self(
            prefilledOpening: prefilledOpening,
            defersClassification: true
        )
        let streamed = filter.consume(text)
        let trailing = filter.finish()
        var result = LocalAPIReasoningResult(
            reasoning: filter.reasoningText,
            content: (streamed.content + trailing.content)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        )

        // Some Ornith/MLX templates omit the opening marker but retain the
        // closing marker. The marker is authoritative: keep everything
        // before the final close in the separate reasoning field, matching
        // the reasoning_content contract used by OpenAI-compatible servers.
        // Prefer the live filter result when prefilledOpening already routed
        // the stream correctly; only apply this salvage when reasoning is
        // still empty (e.g. close-only text without prefilled mode).
        if result.reasoning.isEmpty,
           let close = lastTag(in: text, tags: Self.closingTags) {
            let prefix = String(text[..<close.lowerBound])
            let suffix = String(text[close.upperBound...])
            result = LocalAPIReasoningResult(
                reasoning: normalizedReasoning(prefix),
                content: suffix.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        return result
    }

    static func removingReasoning(from text: String) -> String {
        parse(text).content
    }

    private mutating func takeReasoningDelta() -> String {
        // Hold CoT until the closing tag confirms this really is reasoning
        // (prefilled templates) so `finish()` can still promote a no-tag
        // completion into `content`.
        if holdsReasoning { return "" }
        guard reasoning.count > emittedReasoningCount else { return "" }
        let start = reasoning.index(
            reasoning.startIndex,
            offsetBy: emittedReasoningCount
        )
        emittedReasoningCount = reasoning.count
        return String(reasoning[start...])
    }

    private func firstTag(in text: String, tags: [String]) -> Range<String.Index>? {
        Self.firstTag(in: text, tags: tags, options: [.caseInsensitive])
    }

    private static func lastTag(in text: String, tags: [String]) -> Range<String.Index>? {
        tags.compactMap { tag in
            text.range(of: tag, options: [.caseInsensitive, .backwards])
        }.max { lhs, rhs in
            lhs.lowerBound < rhs.lowerBound
        }
    }

    private static func firstTag(
        in text: String,
        tags: [String],
        options: String.CompareOptions
    ) -> Range<String.Index>? {
        tags.compactMap { tag in
            text.range(of: tag, options: options)
        }.min { lhs, rhs in
            lhs.lowerBound < rhs.lowerBound
        }
    }

    private enum ImplicitOpeningDecision {
        case waiting
        case matched(String)
        case notMatched
    }

    private func implicitOpeningDecision(for text: String) -> ImplicitOpeningDecision {
        let whitespaceEnd = text.firstIndex {
            !$0.isWhitespace
        } ?? text.endIndex
        let prefix = String(text[whitespaceEnd...])
        guard !prefix.isEmpty else { return .waiting }

        let marker = Self.implicitOpening
        let lowerPrefix = prefix.lowercased()
        let lowerMarker = marker.lowercased()
        if lowerPrefix.count < lowerMarker.count,
           lowerMarker.hasPrefix(lowerPrefix) {
            return .waiting
        }
        guard lowerPrefix.hasPrefix(lowerMarker) else { return .notMatched }

        let markerEnd = prefix.index(prefix.startIndex, offsetBy: marker.count)
        let remainder = String(prefix[markerEnd...])
        return .matched(remainder)
    }

    private static func normalizedReasoning(_ text: String) -> String {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for marker in [
            "Thinking Process:",
            "<think>", "<thinking>", "<reasoning>", "<|think|>", "<|thinking|>"
        ] {
            guard let range = result.range(
                of: marker,
                options: [.caseInsensitive, .anchored]
            ) else { continue }
            result = String(result[range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            break
        }
        return result
    }

    private mutating func appendReasoning(_ text: String) {
        guard !text.isEmpty else { return }
        reasoning += text
    }

    private func longestSuffixPrefixLength(in text: String, tags: [String]) -> Int {
        let maximum = min(text.count, (tags.map(\.count).max() ?? 1) - 1)
        guard maximum > 0 else { return 0 }
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length)).lowercased()
            if tags.contains(where: { $0.lowercased().hasPrefix(suffix) }) {
                return length
            }
        }
        return 0
    }
}
enum LocalAPIToolCalling {
    /// Tool JSON must be buffered until it is complete so it can be emitted
    /// using the provider's native tool-call shape. Ordinary prose should not
    /// be buffered: doing so makes streaming clients appear frozen.
    static func shouldBufferForToolDecision(_ output: String) -> Bool {
        let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }

        let toolPrefixes = ["{", "<tool_call", "```json", "```"]
        let couldBeToolCall = toolPrefixes.contains {
            trimmed.hasPrefix($0) || $0.hasPrefix(trimmed)
        }
        guard couldBeToolCall else { return false }

        // Some models begin an ordinary answer with JSON or a fenced block.
        // Once enough of the prefix is present to show that it is not using
        // our tool-call envelope, stream it as prose instead of buffering the
        // entire generation and making agent UIs appear frozen.
        let decisionPrefix = String(trimmed.prefix(160))
        if decisionPrefix.count >= 64,
           !decisionPrefix.contains("\"tool_calls\""),
           !decisionPrefix.contains("\"name\""),
           !decisionPrefix.contains("<tool_call") {
            return false
        }
        return true
    }

    static func messages(
        from messages: [ChatMessage],
        tools: [LocalAPIToolDefinition],
        choice: LocalAPIToolChoice,
        parallelToolCalls: Bool,
        parser: ModelToolParser = .hermes,
        nativeTemplate: Bool = false
    ) -> [ChatMessage] {
        guard !tools.isEmpty, choice != .none else { return messages }

        let toolList = tools.map { tool in
            var line = "- \(tool.name)"
            if let description = tool.description, !description.isEmpty {
                line += ": \(description)"
            }
            return line + "\n  parameters: \(tool.parametersJSON)"
        }.joined(separator: "\n")

        let choiceInstruction: String
        let hasToolResult = messages.contains {
            $0.role == .tool
                || $0.content.contains("[Tool result")
                || $0.content.contains("TOOL RESULT:")
        }
        switch choice {
        case .auto:
            choiceInstruction = hasToolResult
                ? "A requested tool has finished and its result is present. Answer naturally using it now. Call another tool only if the result is insufficient."
                : "Call a tool when it is needed to answer the user. Otherwise answer normally."
        case .required:
            choiceInstruction = "You must call at least one available tool."
        case .function(let name):
            choiceInstruction = "You must call the function named \(name)."
        case .none:
            return messages
        }
        let countInstruction = parallelToolCalls
            ? "You may include multiple calls when they can run in parallel."
            : "Return exactly one tool call at a time."

        // MLX can render the model package's tokenizer chat template with a
        // native `tools` payload. In that mode do not duplicate schemas or
        // prescribe a competing wire syntax in plain text; only retain the
        // behavioral choice/parallel guidance that templates do not encode.
        if nativeTemplate {
            let instruction = """
            External functions are available through the native tool interface. \(choiceInstruction)
            \(countInstruction)
            Never invent a function name. Preserve tool call IDs when using returned results.
            """
            var result = messages
            if let systemIndex = result.firstIndex(where: { $0.role == .system }) {
                var system = result[systemIndex]
                system.content += "\n\n" + instruction
                result[systemIndex] = system
            } else {
                result.insert(ChatMessage(role: .system, content: instruction), at: 0)
            }
            return result
        }

        let responseInstruction: String
        switch parser {
        case .hermes:
            responseInstruction = "For a call, output <tool_call>{\"name\":\"function_name\",\"arguments\":{\"argument\":\"value\"}}</tool_call>. For an ordinary answer, respond naturally without a tool-call wrapper."
        case .qwen3XML, .qwen3Coder:
            responseInstruction = "For a call, output <tool_call><function=function_name><parameter=argument>value</parameter></function></tool_call>. For an ordinary answer, respond naturally without a tool-call wrapper."
        case .foundationModels:
            responseInstruction = "Use the Foundation Models tool-call representation selected by the server. For an ordinary answer, respond naturally without a tool-call wrapper."
        case .none:
            responseInstruction = "For a call, output exactly {\"tool_calls\":[{\"name\":\"function_name\",\"arguments\":{\"argument\":\"value\"}}]}."
        }
        let protocolPrompt = """
        You can call external functions. \(choiceInstruction)
        \(countInstruction)
        Use this model family's native \(parser.displayName) tool syntax. \(responseInstruction)
        Never invent a function name or execute the function yourself. The client executes calls and sends results back with the original call ID. After a tool result is present, answer the user normally unless another call is genuinely necessary.

        Available functions:
        \(toolList)
        """

        let systemInstructions = messages
            .filter { $0.role == .system }
            .map(\.content)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        var result = messages.filter { $0.role != .system }
        if let userIndex = result.lastIndex(where: { $0.role == .user }) {
            var user = result[userIndex]
            var sections: [String] = []
            if !systemInstructions.isEmpty {
                sections.append("[Agent instructions]\n\(systemInstructions)")
            }
            sections.append("[Tool calling instructions]\n\(protocolPrompt)")
            sections.append("[Current user or tool message]\n\(user.content)")
            user.content = sections.joined(separator: "\n\n")
            result[userIndex] = user
        } else {
            let content = systemInstructions.isEmpty
                ? protocolPrompt
                : "[Agent instructions]\n\(systemInstructions)\n\n\(protocolPrompt)"
            result.append(ChatMessage(role: .user, content: content))
        }
        return result
    }

    static func nativeConstraintConfiguration(
        tools: [LocalAPIToolDefinition],
        choice: LocalAPIToolChoice,
        parallelToolCalls: Bool,
        reasoningEnabled: Bool,
        parser: ModelToolParser = .hermes
    ) -> MLXToolCallConstraintConfiguration? {
        guard !tools.isEmpty, choice != .none else { return nil }
        guard parser != .qwen3XML, parser != .qwen3Coder else { return nil }

        let names: [String]
        let decision: MLXToolCallConstraintConfiguration.Decision
        switch choice {
        case .auto:
            names = tools.map(\.name)
            decision = .automatic
        case .required:
            names = tools.map(\.name)
            decision = .required
        case .function(let name):
            names = tools.filter { $0.name == name }.map(\.name)
            decision = .required
        case .none:
            return nil
        }

        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: names,
            decision: decision,
            allowParallelCalls: parallelToolCalls,
            allowReasoningPrefixes: reasoningEnabled
        )
        return configuration.isSuitableForNativeConstraint ? configuration : nil
    }

    static func parse(
        _ output: String,
        tools: [LocalAPIToolDefinition],
        parallelToolCalls: Bool,
        validateSchemas: Bool = true,
        parser: ModelToolParser = .hermes,
        maximumCalls: Int? = nil,
        attemptRepair: Bool = true
    ) -> [LocalAPIToolCall] {
        let toolsByName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })
        guard !toolsByName.isEmpty else { return [] }

        let callLimit = max(1, maximumCalls ?? (parallelToolCalls ? 4 : 1))
        var parsedCalls: [LocalAPIToolCall] = []
        /// Well-formed calls to a known tool that only failed schema validation.
        var schemaRejectedCalls: [LocalAPIToolCall] = []
        let cleanedOutput = LocalAPIReasoningFilter.removingReasoning(from: output)
        if parser == .qwen3XML || parser == .qwen3Coder {
            parsedCalls.append(contentsOf: xmlFunctionCalls(
                in: cleanedOutput,
                toolsByName: toolsByName,
                validateSchemas: validateSchemas
            ))
            if parsedCalls.count >= callLimit {
                return Array(parsedCalls.prefix(callLimit))
            }
        }
        for data in candidateJSONObjects(in: cleanedOutput) {
            guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            let rawCalls: [[String: Any]]
            if let calls = object["tool_calls"] as? [[String: Any]] {
                rawCalls = calls
            } else {
                rawCalls = [object]
            }

            for rawCall in rawCalls {
                let function = rawCall["function"] as? [String: Any] ?? rawCall
                guard let name = function["name"] as? String,
                      let tool = toolsByName[name],
                      let argumentsJSON = normalizedArguments(
                        function["arguments"]
                            ?? function["parameters"]
                            ?? function["args"]
                      ) else {
                    continue
                }
                let call = LocalAPIToolCall(
                    id: nonEmptyString(rawCall["id"] as? String)
                        ?? "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                    name: name,
                    argumentsJSON: argumentsJSON
                )
                guard !validateSchemas
                        || argumentsMatchSchema(argumentsJSON, schemaJSON: tool.parametersJSON) else {
                    schemaRejectedCalls.append(call)
                    continue
                }
                parsedCalls.append(call)
                if parsedCalls.count >= callLimit { return Array(parsedCalls.prefix(callLimit)) }
            }
        }
        if parsedCalls.isEmpty,
           attemptRepair,
           let repaired = repairedToolEnvelope(cleanedOutput),
           repaired != cleanedOutput {
            let repairedCalls = parse(
                repaired,
                tools: tools,
                parallelToolCalls: parallelToolCalls,
                validateSchemas: validateSchemas,
                parser: parser,
                maximumCalls: maximumCalls,
                attemptRepair: false
            )
            if !repairedCalls.isEmpty { return repairedCalls }
        }
        // A named call to a real tool that only fails its schema is still a
        // tool call. Dropping it turned one extra key into a 502 and told the
        // caller to shorten their prompt, which never helped.
        if parsedCalls.isEmpty, !schemaRejectedCalls.isEmpty {
            RuntimeLogCenter.emit(
                "Tool call arguments did not match the declared schema; forwarding unvalidated · "
                    + "tools=\(schemaRejectedCalls.map(\.name).joined(separator: ","))",
                level: .error,
                subsystem: "api"
            )
            return Array(schemaRejectedCalls.prefix(callLimit))
        }
        if parsedCalls.isEmpty, containsToolCallEnvelope(output) {
            RuntimeLogCenter.emit(
                "Tool call parsing failed · parser=\(parser.rawValue) · "
                    + "tools=\(tools.map(\.name).joined(separator: ",")) · "
                    + "raw=\(output.prefix(600))",
                level: .error,
                subsystem: "api"
            )
        }
        return Array(parsedCalls.prefix(callLimit))
    }

    static func textResponse(from output: String) -> String? {
        let cleanedOutput = LocalAPIReasoningFilter.removingReasoning(from: output)
        for data in candidateJSONObjects(in: cleanedOutput) {
            guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  object["response_type"] as? String == "text",
                  let content = object["content"] as? String else {
                continue
            }
            return content
        }
        return nil
    }

    static func containsToolCallEnvelope(_ output: String) -> Bool {
        let normalized = LocalAPIReasoningFilter.removingReasoning(from: output).lowercased()
        return normalized.contains("\"tool_calls\"")
            || normalized.contains("<tool_call")
            || (normalized.contains("\"name\"") && normalized.contains("\"arguments\""))
    }

    static let invalidToolCallMessage =
        "The on-device model did not produce a tool call this request could use. "
            + "It named no available tool, or its call could not be parsed. "
            + "The raw model output is in the app's runtime log under the 'api' subsystem."

    private static func normalizedArguments(_ value: Any?) -> String? {
        if let string = value as? String,
           let data = string.data(using: .utf8),
           (try? JSONSerialization.jsonObject(with: data)) != nil {
            return string
        }
        let object = value ?? [:]
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]
              ) else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    private static func nonEmptyString(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }

    /// A single conservative repair pass for common streaming/quantization
    /// mistakes. Schema validation still runs after repair; malformed output
    /// is never silently downgraded to ordinary assistant text.
    private static func repairedToolEnvelope(_ output: String) -> String? {
        var repaired = output
            .replacingOccurrences(of: "“", with: "\"")
            .replacingOccurrences(of: "”", with: "\"")
            .replacingOccurrences(of: "‘", with: "'")
            .replacingOccurrences(of: "’", with: "'")
        repaired = repaired.replacingOccurrences(
            of: #",\s*([}\]])"#,
            with: "$1",
            options: .regularExpression
        )
        return repaired
    }

    /// Qwen3 XML-family templates emit function and parameter tags instead of
    /// JSON. Parse that native form, then normalize it into OpenAI tool calls.
    private static func xmlFunctionCalls(
        in output: String,
        toolsByName: [String: LocalAPIToolDefinition],
        validateSchemas: Bool
    ) -> [LocalAPIToolCall] {
        let blockPattern = #"(?s)<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>"#
        guard let blockRegex = try? NSRegularExpression(pattern: blockPattern) else { return [] }
        let outputRange = NSRange(output.startIndex..<output.endIndex, in: output)
        return blockRegex.matches(in: output, range: outputRange).compactMap { match in
            guard let nameRange = Range(match.range(at: 1), in: output),
                  let bodyRange = Range(match.range(at: 2), in: output) else { return nil }
            let name = String(output[nameRange])
            guard let tool = toolsByName[name] else { return nil }
            let body = String(output[bodyRange])
            let parameterPattern = #"(?s)<parameter=([^>\s]+)>(.*?)</parameter>"#
            guard let parameterRegex = try? NSRegularExpression(pattern: parameterPattern) else { return nil }
            let bodyRangeNS = NSRange(body.startIndex..<body.endIndex, in: body)
            var arguments: [String: Any] = [:]
            for parameter in parameterRegex.matches(in: body, range: bodyRangeNS) {
                guard let keyRange = Range(parameter.range(at: 1), in: body),
                      let valueRange = Range(parameter.range(at: 2), in: body) else { continue }
                let key = String(body[keyRange])
                let raw = String(body[valueRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                if let integer = Int(raw) {
                    arguments[key] = integer
                } else if let number = Double(raw) {
                    arguments[key] = number
                } else if raw == "true" || raw == "false" {
                    arguments[key] = raw == "true"
                } else if let data = raw.data(using: .utf8),
                   let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) {
                    arguments[key] = value
                } else {
                    arguments[key] = raw
                }
            }
            guard let argumentsJSON = normalizedArguments(arguments),
                  !validateSchemas || argumentsMatchSchema(argumentsJSON, schemaJSON: tool.parametersJSON) else {
                return nil
            }
            return LocalAPIToolCall(
                id: "call_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                name: name,
                argumentsJSON: argumentsJSON
            )
        }
    }

    private static func argumentsMatchSchema(_ argumentsJSON: String, schemaJSON: String) -> Bool {
        guard let argumentsData = argumentsJSON.data(using: .utf8),
              let arguments = try? JSONSerialization.jsonObject(with: argumentsData),
              let schemaData = schemaJSON.data(using: .utf8),
              let schema = try? JSONSerialization.jsonObject(with: schemaData) else {
            return false
        }
        return validate(arguments, against: schema)
    }

    private static func validate(_ value: Any, against schemaValue: Any) -> Bool {
        guard let schema = schemaValue as? [String: Any] else { return true }

        if let enumValues = schema["enum"] as? [Any],
           !enumValues.contains(where: { jsonValuesEqual(value, $0) }) {
            return false
        }
        if let anyOf = schema["anyOf"] as? [Any],
           !anyOf.contains(where: { validate(value, against: $0) }) {
            return false
        }
        if let oneOf = schema["oneOf"] as? [Any],
           oneOf.filter({ validate(value, against: $0) }).count != 1 {
            return false
        }
        if let allOf = schema["allOf"] as? [Any],
           !allOf.allSatisfy({ validate(value, against: $0) }) {
            return false
        }

        if let type = schema["type"] as? String,
           !matchesJSONType(value, type: type) {
            return false
        }
        if let types = schema["type"] as? [String],
           !types.contains(where: { matchesJSONType(value, type: $0) }) {
            return false
        }

        if let object = value as? [String: Any] {
            if let required = schema["required"] as? [String],
               required.contains(where: { object[$0] == nil }) {
                return false
            }
            if let properties = schema["properties"] as? [String: Any] {
                for (key, propertySchema) in properties {
                    if let property = object[key],
                       !validate(property, against: propertySchema) {
                        return false
                    }
                }
            }
            if schema["additionalProperties"] as? Bool == false,
               let properties = schema["properties"] as? [String: Any],
               object.keys.contains(where: { properties[$0] == nil }) {
                return false
            }
        }
        if let array = value as? [Any],
           let itemSchema = schema["items"] {
            if array.contains(where: { !validate($0, against: itemSchema) }) {
                return false
            }
        }
        return true
    }

    private static func matchesJSONType(_ value: Any, type: String) -> Bool {
        switch type {
        case "object": return value is [String: Any]
        case "array": return value is [Any]
        case "string": return value is String
        case "boolean": return value is Bool
        case "number":
            guard let number = value as? NSNumber else { return false }
            return CFGetTypeID(number) != CFBooleanGetTypeID()
        case "integer":
            guard let number = value as? NSNumber,
                  CFGetTypeID(number) != CFBooleanGetTypeID() else { return false }
            return number.doubleValue.rounded() == number.doubleValue
        case "null": return value is NSNull
        default: return true
        }
    }

    private static func jsonValuesEqual(_ lhs: Any, _ rhs: Any) -> Bool {
        guard let lhsData = try? JSONSerialization.data(
            withJSONObject: lhs,
            options: [.fragmentsAllowed, .sortedKeys]
        ), let rhsData = try? JSONSerialization.data(
            withJSONObject: rhs,
            options: [.fragmentsAllowed, .sortedKeys]
        ) else { return false }
        return lhsData == rhsData
    }

    private static func candidateJSONObjects(in output: String) -> [Data] {
        let bytes = Array(output.utf8)
        var candidates: [Data] = []
        var start: Int?
        var depth = 0
        var inString = false
        var escaped = false

        for (index, byte) in bytes.enumerated() {
            if inString {
                if escaped {
                    escaped = false
                } else if byte == 0x5C {
                    escaped = true
                } else if byte == 0x22 {
                    inString = false
                }
                continue
            }
            if byte == 0x22 {
                inString = true
            } else if byte == 0x7B {
                if depth == 0 { start = index }
                depth += 1
            } else if byte == 0x7D, depth > 0 {
                depth -= 1
                if depth == 0, let startIndex = start {
                    candidates.append(Data(bytes[startIndex...index]))
                    start = nil
                }
            }
        }
        return candidates
    }
}

enum LocalAPIInferencePolicy {
    static let minimumDeadlineSeconds = 90
    static let maximumDeadlineSeconds = 900
    static let toolTimeoutMessage =
        "The on-device model did not finish a valid tool call before the generation deadline. Please retry with a smaller context or tool set."
    static let responseTimeoutMessage =
        "The on-device model did not finish before the generation deadline. The partial response was not reported as complete."

    /// Slow 9B models can legitimately need several minutes for a 2K–4K
    /// completion. Scale the watchdog with the accepted output budget instead
    /// of cutting every request off after 90 seconds (roughly 900 tokens on a
    /// typical phone). The client-disconnect monitor remains immediate.
    static func deadline(maxTokens: Int, toolCallingEnabled: Bool) -> Duration {
        let secondsPerToken = toolCallingEnabled ? 0.15 : 0.25
        let scaled = Int(ceil(Double(maxTokens) * secondsPerToken))
        return .seconds(
            min(maximumDeadlineSeconds, max(minimumDeadlineSeconds, scaled))
        )
    }

    /// `runtimeMaximum` is resolved from the loaded backend/model and is also
    /// what `/v1/models` advertises. Tool calling does not get a second hidden
    /// cap: a coding agent may legitimately return prose/code instead of a
    /// call, and that visible completion owns the caller's full budget.
    static func maxTokens(requested: Int?, runtimeMaximum: Int) -> Int {
        min(max(1, requested ?? runtimeMaximum), max(1, runtimeMaximum))
    }
}

enum LocalAPIAssistantOutputValidator {
    static let noAnswerMessage =
        "Model stopped without text or a valid tool call. Retry, or switch to a compatible model."

    static func hasAnswer(text: String, toolCalls: [LocalAPIToolCall]) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !toolCalls.isEmpty
    }
}

struct LocalAPIUsage: Equatable, Sendable {
    let promptTokens: Int
    let completionTokens: Int

    var totalTokens: Int { promptTokens + completionTokens }

    init(_ result: AssistantGenerationResult) {
        promptTokens = result.promptTokenCount
        completionTokens = result.completionTokenCount
    }

    var object: [String: Int] {
        [
            "prompt_tokens": promptTokens,
            "completion_tokens": completionTokens,
            "total_tokens": totalTokens
        ]
    }
}

enum LocalAPIResponse {
    static func openAIChunk(
        id: String,
        model: String,
        text: String,
        role: String? = nil,
        finishReason: String? = nil,
        reasoningContent: String? = nil
    ) -> Data {
        var delta: [String: Any] = text.isEmpty ? [:] : ["content": text]
        if let role {
            delta["role"] = role
        }
        if let reasoningContent, !reasoningContent.isEmpty {
            delta["reasoning_content"] = reasoningContent
        }
        let choice: [String: Any] = [
            "index": 0,
            "delta": delta,
            "finish_reason": finishReason as Any
        ]
        return json([
            "id": id, "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model, "choices": [choice]
        ])
    }

    static func openAIToolCallChunk(
        id: String,
        model: String,
        calls: [LocalAPIToolCall],
        finishReason: String? = nil
    ) -> Data {
        let delta: [String: Any]
        if calls.isEmpty {
            delta = [:]
        } else {
            delta = [
                "role": "assistant",
                "tool_calls": calls.enumerated().map { index, call in
                    [
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": call.argumentsJSON
                        ]
                    ] as [String: Any]
                }
            ]
        }
        return json([
            "id": id,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [[
                "index": 0,
                "delta": delta,
                "finish_reason": finishReason as Any
            ]]
        ])
    }

    static func openAIUsageChunk(
        id: String,
        model: String,
        usage: LocalAPIUsage
    ) -> Data {
        json([
            "id": id,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [],
            "usage": usage.object
        ])
    }

    static func openAIStreamTerminator(
        id: String,
        model: String,
        finishReason: String,
        usage: LocalAPIUsage?,
        includeUsage: Bool
    ) -> Data {
        var data = Data("data: ".utf8)
        data.append(openAIChunk(
            id: id,
            model: model,
            text: "",
            finishReason: finishReason
        ))
        data.append(Data("\n\n".utf8))
        if includeUsage, let usage {
            data.append(Data("data: ".utf8))
            data.append(openAIUsageChunk(id: id, model: model, usage: usage))
            data.append(Data("\n\n".utf8))
        }
        data.append(Data("data: [DONE]\n\n".utf8))
        return data
    }

    static func openAIChatCompletion(
        id: String,
        model: String,
        text: String,
        toolCalls: [LocalAPIToolCall],
        reasoningContent: String = "",
        finishReason: String? = nil,
        usage: LocalAPIUsage? = nil
    ) -> Data {
        var message: [String: Any] = ["role": "assistant"]
        if toolCalls.isEmpty {
            message["content"] = text
        } else {
            message["content"] = NSNull()
            message["tool_calls"] = toolCalls.map { call in
                [
                    "id": call.id,
                    "type": "function",
                    "function": [
                        "name": call.name,
                        "arguments": call.argumentsJSON
                    ]
                ]
            }
        }
        if !reasoningContent.isEmpty {
            message["reasoning_content"] = reasoningContent
        }
        var payload: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [[
                "index": 0,
                "message": message,
                "finish_reason": finishReason
                    ?? (toolCalls.isEmpty ? "stop" : "tool_calls")
            ]]
        ]
        if let usage {
            payload["usage"] = usage.object
        }
        return json(payload)
    }

    static func ollamaChat(
        model: String,
        text: String,
        done: Bool,
        reasoningContent: String = ""
    ) -> Data {
        var message: [String: Any] = ["role": "assistant", "content": text]
        if !reasoningContent.isEmpty {
            message["thinking"] = reasoningContent
        }
        return json([
            "model": model,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "message": message,
            "done": done,
            "done_reason": done ? "stop" : NSNull()
        ])
    }

    static func ollamaToolCalls(
        model: String,
        calls: [LocalAPIToolCall],
        done: Bool
    ) -> Data {
        let toolCalls: [[String: Any]] = calls.map { call in
            let arguments = (try? JSONSerialization.jsonObject(
                with: Data(call.argumentsJSON.utf8)
            )) ?? [:]
            return [
                "function": [
                    "name": call.name,
                    "arguments": arguments
                ]
            ]
        }
        return json([
            "model": model,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "message": [
                "role": "assistant",
                "content": "",
                "tool_calls": toolCalls
            ],
            "done": done,
            "done_reason": done ? "stop" : NSNull()
        ])
    }

    static func ollamaGenerate(
        model: String,
        text: String,
        done: Bool,
        reasoningContent: String = ""
    ) -> Data {
        var payload: [String: Any] = [
            "model": model,
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "response": text,
            "done": done,
            "done_reason": done ? "stop" : NSNull()
        ]
        if !reasoningContent.isEmpty {
            payload["thinking"] = reasoningContent
        }
        return json(payload)
    }

    static func anthropicMessage(
        id: String,
        model: String,
        text: String,
        reasoningContent: String = ""
    ) -> Data {
        var payload: [String: Any] = [
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [["type": "text", "text": text]],
            "stop_reason": "end_turn",
            "stop_sequence": NSNull(),
            "usage": ["input_tokens": 0, "output_tokens": 0]
        ]
        // Keep the visible Anthropic text block clean for clients such as
        // Hermes, while exposing the parsed trace under the same optional
        // field used by OpenAI-compatible adapters.
        if !reasoningContent.isEmpty {
            payload["reasoning_content"] = reasoningContent
        }
        return json(payload)
    }

    static func anthropicToolMessage(
        id: String,
        model: String,
        calls: [LocalAPIToolCall]
    ) -> Data {
        let content: [[String: Any]] = calls.map { call in
            [
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": (try? JSONSerialization.jsonObject(
                    with: Data(call.argumentsJSON.utf8)
                )) ?? [:]
            ]
        }
        return json([
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": NSNull(),
            "usage": ["input_tokens": 0, "output_tokens": 0]
        ])
    }

    static func openAIResponse(
        id: String,
        model: String,
        text: String,
        toolCalls: [LocalAPIToolCall] = [],
        status: String = "completed",
        reasoningContent: String = ""
    ) -> Data {
        let output: [[String: Any]]
        let outputText: String
        if toolCalls.isEmpty {
            outputText = text
            output = [[
                "id": "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                "type": "message",
                "status": status,
                "role": "assistant",
                "content": [[
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                ]]
            ]]
        } else {
            outputText = ""
            output = toolCalls.map { call in
                [
                    "id": "fc_\(call.id)",
                    "type": "function_call",
                    "status": status,
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.argumentsJSON
                ]
            }
        }
        var payload: [String: Any] = [
            "id": id,
            "object": "response",
            "created_at": Int(Date().timeIntervalSince1970),
            "status": status,
            "model": model,
            "output": output,
            "output_text": outputText,
            "error": NSNull(),
            "incomplete_details": NSNull(),
            "usage": [
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            ]
        ]
        if !reasoningContent.isEmpty {
            payload["reasoning_content"] = reasoningContent
        }
        return json(payload)
    }

    static func openAIResponseEvent(_ type: String, fields: [String: Any]) -> Data {
        var object = fields
        object["type"] = type
        return Data("event: \(type)\ndata: ".utf8) + json(object) + Data("\n\n".utf8)
    }

    static func anthropicEvent(_ name: String, object: Any) -> Data {
        let payload = json(object)
        return Data("event: \(name)\ndata: ".utf8) + payload + Data("\n\n".utf8)
    }

    static func json(_ object: Any) -> Data {
        (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
    }
}

actor LocalAPIServer {
    enum Event: Sendable {
        case ready
        case failed(String)
    }

    private enum RequestReadError: Error {
        case timedOut
        case tooLarge
        case connectionClosed
    }

    private var listener: NWListener?
    private var listenerID: UUID?
    private struct ActiveInference {
        let id: UUID
        let signal: LocalAPIInferenceSignal
        let connection: NWConnection
        var timedOut = false
    }
    private var activeInference: ActiveInference?
    // A 12 MB embedded image expands to ~16 MB in base64 plus JSON overhead.
    // Keep a finite authenticated-LAN cap while allowing one full-resolution
    // screenshot/photo request to reach the decoder.
    private let maxRequestBytes = 20 * 1024 * 1024
    private let requestTimeout: TimeInterval = 30
    var onEvent: (@Sendable (Event) -> Void)?

    func setEventHandler(_ handler: @escaping @Sendable (Event) -> Void) {
        onEvent = handler
    }

    func start(port: UInt16) throws {
        guard listener == nil else { return }
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        params.includePeerToPeer = true
        guard let endpointPort = NWEndpoint.Port(rawValue: port) else {
            throw LocalAPIProtocolError.malformed("Invalid port")
        }
        let newListener = try NWListener(using: params, on: endpointPort)
        let newListenerID = UUID()
        newListener.stateUpdateHandler = { [weak self] state in
            Task { await self?.listenerChanged(state, listenerID: newListenerID) }
        }
        newListener.newConnectionHandler = { [weak self] connection in
            Task { await self?.handle(connection) }
        }
        listener = newListener
        listenerID = newListenerID
        newListener.start(queue: .global(qos: .userInitiated))
    }

    func stop() {
        listener?.cancel()
        listener = nil
        listenerID = nil
        guard let activeInference else { return }
        activeInference.signal.finish()
        activeInference.connection.cancel()
        Task { @MainActor in
            CodingAssistantService.shared.stopGeneration()
        }
    }

    private func listenerChanged(_ state: NWListener.State, listenerID: UUID) {
        guard self.listenerID == listenerID else { return }
        switch state {
        case .ready:
            onEvent?(.ready)
        case .failed(let error):
            listener?.cancel()
            listener = nil
            self.listenerID = nil
            onEvent?(.failed(error.localizedDescription))
        default:
            break
        }
    }

    private func handle(_ connection: NWConnection) async {
        let connectionMonitor = LocalAPIConnectionMonitor()
        connection.stateUpdateHandler = { state in
            switch state {
            case .failed, .cancelled:
                connectionMonitor.markDisconnected()
            default:
                break
            }
        }
        connection.start(queue: .global(qos: .userInitiated))
        defer { connectionMonitor.stop() }
        do {
            let request = try await readRequest(connection)
            await route(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor
            )
        } catch let error as RequestReadError {
            let response: (Int, String) = switch error {
            case .timedOut: (408, "Request timed out")
            case .tooLarge: (413, "Request body is too large")
            case .connectionClosed: (400, "Connection closed before a complete request was received")
            }
            await respond(
                connection,
                status: response.0,
                contentType: "application/json",
                data: LocalAPIResponse.json([
                    "error": [
                        "type": "invalid_request_error",
                        "message": response.1
                    ]
                ])
            )
        } catch {
            await respond(
                connection,
                status: 400,
                contentType: "application/json",
                data: LocalAPIResponse.json([
                    "error": [
                        "type": "invalid_request_error",
                        "message": "Malformed HTTP request"
                    ]
                ])
            )
        }
    }

    private func readRequest(_ connection: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        let deadline = Date().addingTimeInterval(requestTimeout)
        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw RequestReadError.timedOut }
            let chunk = try await receive(connection, timeout: remaining)
            guard !chunk.isEmpty else { throw RequestReadError.connectionClosed }
            buffer.append(chunk)
            guard buffer.count <= maxRequestBytes else {
                throw RequestReadError.tooLarge
            }
            if let request = HTTPRequest(data: buffer) { return request }
        }
    }

    private func receive(_ connection: NWConnection, timeout: TimeInterval) async throws -> Data {
        try await withCheckedThrowingContinuation { continuation in
            let once = LocalAPIResumeOnce(continuation)
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) {
                data, _, _, error in
                if let error { once.resume(throwing: error) }
                else { once.resume(returning: data ?? Data()) }
            }
            let timeoutTask = Task<Void, Never> {
                do {
                    try await Task.sleep(for: .seconds(timeout))
                } catch {
                    return
                }
                if once.resume(throwing: RequestReadError.timedOut) {
                    connection.cancel()
                }
            }
            once.setTimeoutTask(timeoutTask)
        }
    }

    private func route(
        _ request: HTTPRequest,
        connection: NWConnection,
        connectionMonitor: LocalAPIConnectionMonitor
    ) async {
        // Browser clients send a preflight before requests carrying
        // Authorization / x-api-key. Answer it before authentication so web
        // agents can reach the same local endpoint as terminal clients.
        if request.method == "OPTIONS" {
            await respond(
                connection,
                status: 204,
                contentType: "text/plain",
                data: Data()
            )
            return
        }
        let key = LocalAPIKeyStore.key()
        let authorized = LocalAPIValidation.isAuthorized(
            headers: request.headers,
            key: key
        )
        guard authorized else {
            RuntimeLogCenter.emit(
                "\(request.method) \(request.path) rejected (unauthorized)",
                level: .warning,
                subsystem: "api"
            )
            await error(connection, status: 401, message: "Invalid or missing API key", dialect: request.path == "/v1/messages" ? .anthropic : .openAIChat)
            return
        }
        RuntimeLogCenter.emit(
            "\(request.method) \(request.path)",
            subsystem: "api"
        )

        if request.method == "GET", request.path.hasPrefix("/v1/models/") {
            await retrieveOpenAIModel(request, connection: connection)
            return
        }

        switch (request.method, request.path) {
        case ("GET", "/"), ("GET", "/v1"):
            await describeAPI(connection)
        case ("GET", "/v1/models"):
            await listOpenAIModels(connection)
        case ("POST", "/v1/chat/completions"):
            await run(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor,
                dialect: .openAIChat
            )
        case ("POST", "/v1/responses"):
            await run(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor,
                dialect: .openAIResponses
            )
        case ("POST", "/v1/messages"):
            await run(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor,
                dialect: .anthropic
            )
        case ("GET", "/api/tags"):
            await listOllamaModels(connection)
        case ("GET", "/api/ps"):
            await listRunningOllamaModels(connection)
        case ("GET", "/api/version"):
            await showOllamaVersion(connection)
        case ("POST", "/api/show"):
            await showOllamaModel(request, connection: connection)
        case ("POST", "/api/chat"):
            await run(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor,
                dialect: .ollamaChat
            )
        case ("POST", "/api/generate"):
            await run(
                request,
                connection: connection,
                connectionMonitor: connectionMonitor,
                dialect: .ollamaGenerate
            )
        default:
            await error(connection, status: 404, message: "Endpoint not supported", dialect: request.path == "/v1/messages" ? .anthropic : .openAIChat)
        }
    }

    private enum Dialect {
        case openAIChat, openAIResponses, anthropic, ollamaChat, ollamaGenerate

        /// The Ollama dialects stream newline-delimited JSON, which has no
        /// comment syntax to hide a heartbeat in.
        var usesServerSentEvents: Bool {
            switch self {
            case .openAIChat, .openAIResponses, .anthropic: true
            case .ollamaChat, .ollamaGenerate: false
            }
        }
    }

    private func run(
        _ request: HTTPRequest,
        connection: NWConnection,
        connectionMonitor: LocalAPIConnectionMonitor,
        dialect: Dialect
    ) async {
        guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true else {
            await error(connection, status: 415, message: "Content-Type must be application/json", dialect: dialect)
            return
        }
        guard let body = request.body else {
            await error(connection, status: 400, message: "JSON body required", dialect: dialect)
            return
        }
        let decoded: LocalAPIChatRequest
        do {
            switch dialect {
            case .openAIChat: decoded = try .decodeOpenAI(body)
            case .openAIResponses: decoded = try .decodeOpenAIResponses(body)
            case .anthropic: decoded = try .decodeAnthropic(body)
            case .ollamaChat: decoded = try .decodeOllamaChat(body)
            case .ollamaGenerate: decoded = try .decodeOllamaGenerate(body)
            }
        } catch let error as LocalAPIProtocolError {
            let message: String
            switch error {
            case .malformed(let value), .unsupported(let value): message = value
            case .unknownModel: message = "Unknown model"
            }
            await self.error(connection, status: 400, message: message, dialect: dialect)
            return
        } catch {
            await self.error(connection, status: 400, message: "Malformed request", dialect: dialect)
            return
        }

        // A completed token stream can reach the HTTP layer a fraction before
        // the native generation task has unwound its final cache cleanup. Wait
        // for that drain so a back-to-back Hermes request is not mistaken for
        // an unloaded/busy model.
        _ = await CodingAssistantService.shared.waitForGenerationToFinish(
            timeout: .seconds(5)
        )
        let snapshot = await MainActor.run { () -> (String, String, Bool, Bool, Bool, Bool, Bool, ModelCapabilityProfile, Bool, Bool, Int) in
            let service = CodingAssistantService.shared
            let settings = AppSettings.shared
            return (
                service.activeModel.id,
                service.activeModel.repoID,
                service.state == .ready,
                settings.localAPIToolCallingEnabled,
                settings.localAPIReasoningEnabled,
                settings.localAPIParallelToolCallsEnabled,
                settings.localAPIStrictToolSchemasEnabled,
                ModelCapabilityProfile.resolve(for: service.activeModel),
                service.activeModel.runtime != .llamaCpp,
                service.isVisionChatCapable,
                service.localAPIEffectiveMaximumOutputTokens
            )
        }
        RuntimeLogCenter.emit(
            "Inference request accepted for model \(snapshot.0)",
            subsystem: "api"
        )
        let usesCurrentOllamaModel = decoded.model.isEmpty
            && (dialect == .ollamaChat || dialect == .ollamaGenerate)
        guard usesCurrentOllamaModel
                || LocalAPIValidation.modelMatches(decoded.model, id: snapshot.0, repoID: snapshot.1) else {
            await error(connection, status: 404, message: "Model '\(decoded.model)' is not loaded", dialect: dialect)
            return
        }
        guard snapshot.2 else {
            await error(connection, status: 503, message: "The active model is not loaded or is busy", dialect: dialect)
            return
        }
        let requestContainsImages = decoded.messages.contains {
            !$0.imageThumbnails.isEmpty || $0.imageThumbnailData != nil
        }
        guard !requestContainsImages || snapshot.9 else {
            await error(
                connection,
                status: 400,
                message: "The active model is text-only. Load a vision model with its matching mmproj projector.",
                dialect: dialect
            )
            return
        }
        guard let lease = await acquireInferenceLease() else {
            await error(connection, status: 503, message: "Model busy", dialect: dialect)
            return
        }

        let toolCallingEnabled = (
            dialect == .openAIChat
                || dialect == .openAIResponses
                || dialect == .anthropic
                || dialect == .ollamaChat
        )
            && snapshot.3
            && !decoded.tools.isEmpty
            && decoded.toolChoice != .none
        let configuredParallelLimit = await MainActor.run {
            LocalAPIParallelToolCallsSetting(
                rawValue: AppSettings.shared.localAPIParallelToolCallsLimit
            )?.maximumCalls ?? LocalAPIParallelToolCallsSetting.automaticTwo.maximumCalls
        }
        let parallelLimit = snapshot.7.effectiveParallelLimit(
            requestAllowsParallel: decoded.parallelToolCalls,
            globalEnabled: snapshot.5,
            configuredMaximumCalls: configuredParallelLimit
        )
        let inferenceID = UUID()
        let inferenceSignal = LocalAPIInferenceSignal()
        activeInference = ActiveInference(
            id: inferenceID,
            signal: inferenceSignal,
            connection: connection
        )
        let disconnectTask = Task { [weak self] in
            await connectionMonitor.wait()
            await self?.cancelInference(
                inferenceID: inferenceID,
                reason: "client disconnected"
            )
        }
        let maxTokens = LocalAPIInferencePolicy.maxTokens(
            requested: decoded.maxTokens,
            runtimeMaximum: snapshot.10
        )
        let inferenceDeadline = LocalAPIInferencePolicy.deadline(
            maxTokens: maxTokens,
            toolCallingEnabled: toolCallingEnabled
        )
        let deadlineReason = toolCallingEnabled
            ? "tool-selection deadline"
            : "response deadline"
        let deadlineTask = Task { [weak self] in
            do {
                try await Task.sleep(for: inferenceDeadline)
                await self?.cancelInference(
                    inferenceID: inferenceID,
                    reason: deadlineReason,
                    timedOut: true
                )
            } catch {
                // Normal completion cancels this deadline task.
            }
        }
        defer {
            if activeInference?.id == inferenceID {
                activeInference = nil
            }
            disconnectTask.cancel()
            connectionMonitor.stop()
            deadlineTask.cancel()
            RemoteInferenceGate.shared.release(lease)
        }

        let requestedTemperature = decoded.temperature.map { min(max(0, $0), 2) }
        // Tool selection is a protocol decision, not creative prose. A
        // deterministic pass materially reduces malformed envelopes on small
        // Qwen/MLX models while ordinary chat keeps the caller's sampler.
        let temperature = toolCallingEnabled ? 0 : requestedTemperature
        let topP = decoded.topP.map { min(max(0, $0), 1) }
        let preparedContext = ConversationContextCompactor.prepare(
            messages: decoded.messages,
            existingMemory: nil,
            maxTokens: max(
                1_024,
                snapshot.7.configuredContextLength
                    - maxTokens
            )
        )
        if preparedContext.didCompact {
            RuntimeLogCenter.emit(
                "Context compacted locally · model=\(snapshot.0) · parser=\(snapshot.7.toolParser.rawValue)",
                subsystem: "api"
            )
        }
        let usesNativeToolTemplate = toolCallingEnabled
            && snapshot.8
            && snapshot.7.toolParser != .none
        let forceNoThinking = decoded.reasoningPreference.forceNoThinking(
            globalEnabled: snapshot.4
        )
        let inferenceMessages = LocalAPIToolCalling.messages(
            from: preparedContext.messages,
            tools: toolCallingEnabled ? decoded.tools : [],
            choice: toolCallingEnabled ? decoded.toolChoice : .none,
            parallelToolCalls: parallelLimit > 1,
            parser: snapshot.7.toolParser,
            nativeTemplate: usesNativeToolTemplate
        )
        let legacyToolConstraint = toolCallingEnabled && !usesNativeToolTemplate
            ? LocalAPIToolCalling.nativeConstraintConfiguration(
                tools: decoded.tools,
                choice: decoded.toolChoice,
                parallelToolCalls: parallelLimit > 1,
                reasoningEnabled: !forceNoThinking,
                parser: snapshot.7.toolParser
            )
            : nil
        let nativeTools = usesNativeToolTemplate
            ? decoded.tools.map {
                AssistantNativeToolDefinition(
                    name: $0.name,
                    description: $0.description,
                    parametersJSON: $0.parametersJSON
                )
            }
            : []
        let nativeToolFormat: AssistantNativeToolFormat? = switch snapshot.7.toolParser {
        case .hermes: .hermesJSON
        case .qwen3XML, .qwen3Coder: .qwenXML
        // Foundation Models performs tool selection through its own runtime;
        // it does not use the MLX chat-template format adapter.
        case .foundationModels: nil
        case .none: nil
        }
        let stream = await inferenceStream(
            messages: inferenceMessages,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            jsonMode: false,
            forceNoThinking: forceNoThinking,
            toolConstraint: legacyToolConstraint,
            nativeTools: nativeTools,
            nativeToolFormat: usesNativeToolTemplate ? nativeToolFormat : nil,
            signal: inferenceSignal
        )

        // Only start the stream inside a reasoning block when the chat
        // template really pre-filled `<think>`. Assuming it from the model
        // family alone made every no-thinking completion look like unterminated
        // CoT, which the filter then had to withhold until the generation ended
        // — the whole answer arrived in one delta.
        let thinkingEnabled = await MainActor.run {
            CodingAssistantService.shared.resolvedThinkingEnabled(
                forceNoThinking: forceNoThinking
            )
        }
        let prefilledReasoningOpen = snapshot.7.streamAssumesPrefilledReasoningOpen
            && thinkingEnabled

        if decoded.stream {
            let requestID = dialect == .openAIResponses
                ? "resp_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
                : "chatcmpl-\(UUID().uuidString)"
            if toolCallingEnabled {
                var output = ""
                output.reserveCapacity(min(maxTokens * 4, 256 * 1_024))
                do {
                    for await token in stream {
                        output += token
                        // Do not emit model tokens until the complete tool
                        // decision is parsed. This prevents a printed
                        // {"tool_calls": ...} envelope from escaping as
                        // assistant prose and allows parallel calls to arrive
                        // in one native response.
                        if output.count > 100_000 {
                            await cancelInference()
                            break
                        }
                    }
                    let calls = LocalAPIToolCalling.parse(
                        output,
                        tools: decoded.tools,
                        parallelToolCalls: parallelLimit > 1,
                        validateSchemas: snapshot.6,
                        parser: snapshot.7.toolParser,
                        maximumCalls: parallelLimit
                    )
                    let parsedOutput = LocalAPIReasoningFilter.parse(
                        output,
                        prefilledOpening: prefilledReasoningOpen
                    )
                    let cleanedOutput = LocalAPIToolCalling.textResponse(from: output)
                        ?? parsedOutput.content
                    if calls.isEmpty {
                        let requiredTool = decoded.toolChoice != .auto
                        let invalidEnvelope = LocalAPIToolCalling.containsToolCallEnvelope(output)
                        let timedOut = activeInference?.id == inferenceID
                            && activeInference?.timedOut == true
                        if requiredTool || invalidEnvelope || timedOut {
                            // A tool decision cannot be streamed, so nothing has
                            // been written yet and the failure can still carry a
                            // real status — the same 502 the non-streaming path
                            // returns, instead of a 200 wrapping an error event.
                            await error(
                                connection,
                                status: 502,
                                message: timedOut
                                    ? LocalAPIInferencePolicy.toolTimeoutMessage
                                    : LocalAPIToolCalling.invalidToolCallMessage,
                                dialect: dialect
                            )
                            return
                        }
                        guard LocalAPIAssistantOutputValidator.hasAnswer(
                            text: cleanedOutput,
                            toolCalls: calls
                        ) else {
                            RuntimeLogCenter.emit(
                                "Model returned no assistant answer · model=\(snapshot.0)",
                                level: .error,
                                subsystem: "api"
                            )
                            await error(
                                connection,
                                status: 502,
                                message: LocalAPIAssistantOutputValidator.noAnswerMessage,
                                dialect: dialect
                            )
                            return
                        }
                        await writeStreamingHeaders(
                            connection,
                            dialect: dialect,
                            model: snapshot.0
                        )
                        try await sendStreamingTextCompletion(
                            connection,
                            dialect: dialect,
                            model: snapshot.0,
                            requestID: requestID,
                            text: cleanedOutput,
                            reasoningContent: parsedOutput.reasoning,
                            generationResult: inferenceSignal.result,
                            includeUsage: decoded.streamIncludeUsage
                        )
                    } else {
                        await writeStreamingHeaders(
                            connection,
                            dialect: dialect,
                            model: snapshot.0
                        )
                        switch dialect {
                        case .openAIChat:
                            try await send(
                                connection,
                                Data("data: ".utf8)
                                    + LocalAPIResponse.openAIToolCallChunk(
                                        id: requestID,
                                        model: snapshot.0,
                                        calls: calls
                                    )
                                    + Data("\n\n".utf8)
                            )
                            try await sendFinal(
                                connection,
                                LocalAPIResponse.openAIStreamTerminator(
                                    id: requestID,
                                    model: snapshot.0,
                                    finishReason: "tool_calls",
                                    usage: inferenceSignal.result.map(LocalAPIUsage.init),
                                    includeUsage: decoded.streamIncludeUsage
                                )
                            )
                        case .anthropic:
                            try await sendAnthropicToolCompletion(
                                connection,
                                calls: calls
                            )
                        case .openAIResponses:
                            try await sendOpenAIResponseToolCompletion(
                                connection,
                                requestID: requestID,
                                model: snapshot.0,
                                calls: calls
                            )
                        case .ollamaChat:
                            try await sendFinal(
                                connection,
                                LocalAPIResponse.ollamaToolCalls(
                                    model: snapshot.0,
                                    calls: calls,
                                    done: true
                                )
                                    + Data("\n".utf8)
                            )
                        case .ollamaGenerate:
                            break
                        }
                    }
                } catch {
                    await cancelInference()
                    connection.cancel()
                    return
                }
                connection.cancel()
                return
            }
            await writeStreamingHeaders(connection, dialect: dialect, model: snapshot.0)
            var streamedOutput = ""
            streamedOutput.reserveCapacity(min(maxTokens * 4, 256 * 1_024))
            var reasoningFilter = LocalAPIReasoningFilter(
                prefilledOpening: prefilledReasoningOpen
            )
            var startedAssistantRole = false
            var startedVisibleContent = false

            func emitOpenAICompatibleDelta(
                _ delta: LocalAPIReasoningDelta
            ) async throws {
                guard !delta.isEmpty else { return }
                streamedOutput += delta.content
                let reasoningContent = delta.reasoning.isEmpty
                    ? nil
                    : delta.reasoning
                let payload: Data
                switch dialect {
                case .openAIChat:
                    let role: String? = startedAssistantRole ? nil : "assistant"
                    startedAssistantRole = true
                    payload = Data("data: ".utf8) + LocalAPIResponse.openAIChunk(
                        id: requestID,
                        model: snapshot.0,
                        text: delta.content,
                        role: role,
                        reasoningContent: reasoningContent
                    ) + Data("\n\n".utf8)
                case .openAIResponses:
                    // Responses API streams visible text only; reasoning is
                    // attached on the completed response object.
                    guard !delta.content.isEmpty else { return }
                    startedVisibleContent = true
                    payload = LocalAPIResponse.openAIResponseEvent(
                        "response.output_text.delta",
                        fields: [
                            "response_id": requestID,
                            "item_id": "msg_\(requestID)",
                            "output_index": 0,
                            "content_index": 0,
                            "delta": delta.content
                        ]
                    )
                case .anthropic:
                    guard !delta.content.isEmpty else { return }
                    let start = !startedVisibleContent
                        ? LocalAPIResponse.anthropicEvent("content_block_start", object: [
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": ["type": "text", "text": ""]
                        ])
                        : Data()
                    startedVisibleContent = true
                    payload = start + LocalAPIResponse.anthropicEvent(
                        "content_block_delta",
                        object: [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": ["type": "text_delta", "text": delta.content]
                        ]
                    )
                case .ollamaChat:
                    startedAssistantRole = true
                    payload = LocalAPIResponse.ollamaChat(
                        model: snapshot.0,
                        text: delta.content,
                        done: false,
                        reasoningContent: reasoningContent ?? ""
                    ) + Data("\n".utf8)
                case .ollamaGenerate:
                    startedAssistantRole = true
                    payload = LocalAPIResponse.ollamaGenerate(
                        model: snapshot.0,
                        text: delta.content,
                        done: false,
                        reasoningContent: reasoningContent ?? ""
                    ) + Data("\n".utf8)
                }
                try await send(connection, payload)
            }

            // Prefill on a long prompt can run for many seconds before the
            // first token. Keep the connection warm with SSE comments, which
            // every OpenAI-compatible client ignores.
            let heartbeat = dialect.usesServerSentEvents
                ? Task { [weak self] in
                    while !Task.isCancelled {
                        try await Task.sleep(for: .seconds(15))
                        try await self?.send(connection, Data(": ping\n\n".utf8))
                    }
                }
                : nil
            defer { heartbeat?.cancel() }

            for await token in stream {
                heartbeat?.cancel()
                do {
                    try await emitOpenAICompatibleDelta(
                        reasoningFilter.consume(token)
                    )
                } catch {
                    await cancelInference()
                    connection.cancel()
                    return
                }
            }
            do {
                try await emitOpenAICompatibleDelta(reasoningFilter.finish())
            } catch {
                await cancelInference()
                connection.cancel()
                return
            }
            let timedOut = activeInference?.id == inferenceID
                && activeInference?.timedOut == true
            if timedOut {
                RuntimeLogCenter.emit(
                    "Generation deadline reached before a terminal stop · model=\(snapshot.0)",
                    level: .error,
                    subsystem: "api"
                )
                try? await sendStreamingGenerationTimeout(
                    connection,
                    dialect: dialect
                )
                connection.cancel()
                return
            }
            guard LocalAPIAssistantOutputValidator.hasAnswer(
                text: streamedOutput,
                toolCalls: []
            ) else {
                RuntimeLogCenter.emit(
                    "Model returned no assistant answer · model=\(snapshot.0)",
                    level: .error,
                    subsystem: "api"
                )
                try? await sendStreamingModelError(
                    connection,
                    dialect: dialect
                )
                connection.cancel()
                return
            }
            switch dialect {
            case .openAIChat:
                let generationResult = inferenceSignal.result
                let finishReason = generationResult?.stopReason == .length
                    ? "length"
                    : "stop"
                try? await sendFinal(
                    connection,
                    LocalAPIResponse.openAIStreamTerminator(
                        id: requestID,
                        model: snapshot.0,
                        finishReason: finishReason,
                        usage: generationResult.map(LocalAPIUsage.init),
                        includeUsage: decoded.streamIncludeUsage
                    )
                )
            case .openAIResponses:
                try? await send(connection, LocalAPIResponse.openAIResponseEvent("response.output_text.done", fields: [
                    "response_id": requestID,
                    "item_id": "msg_\(requestID)",
                    "output_index": 0,
                    "content_index": 0,
                    "text": streamedOutput
                ]))
                try? await sendFinal(connection, LocalAPIResponse.openAIResponseEvent("response.completed", fields: [
                    "response": (try? JSONSerialization.jsonObject(with: LocalAPIResponse.openAIResponse(
                        id: requestID,
                        model: snapshot.0,
                        text: streamedOutput,
                        reasoningContent: reasoningFilter.reasoningText
                    ))) ?? [:]
                ]))
            case .anthropic:
                try? await send(connection, LocalAPIResponse.anthropicEvent("content_block_stop", object: [
                    "type": "content_block_stop", "index": 0
                ]))
                try? await send(connection, LocalAPIResponse.anthropicEvent("message_delta", object: [
                    "type": "message_delta",
                    "delta": ["stop_reason": "end_turn", "stop_sequence": NSNull()],
                    "usage": ["output_tokens": 0]
                ]))
                try? await sendFinal(connection, LocalAPIResponse.anthropicEvent("message_stop", object: [
                    "type": "message_stop"
                ]))
            case .ollamaChat:
                try? await sendFinal(connection, LocalAPIResponse.ollamaChat(
                    model: snapshot.0,
                    text: "",
                    done: true,
                    reasoningContent: reasoningFilter.reasoningText
                ) + Data("\n".utf8))
            case .ollamaGenerate:
                try? await sendFinal(connection, LocalAPIResponse.ollamaGenerate(
                    model: snapshot.0,
                    text: "",
                    done: true,
                    reasoningContent: reasoningFilter.reasoningText
                ) + Data("\n".utf8))
            }
            connection.cancel()
        } else {
            var output = ""
            output.reserveCapacity(min(maxTokens * 4, 256 * 1_024))
            for await token in stream {
                output += token
            }
            let parsedOutput = LocalAPIReasoningFilter.parse(
                output,
                prefilledOpening: prefilledReasoningOpen
            )
            let cleanedOutput = LocalAPIToolCalling.textResponse(from: output)
                ?? parsedOutput.content
            let detectedCalls = toolCallingEnabled
                ? LocalAPIToolCalling.parse(
                    output,
                    tools: decoded.tools,
                    parallelToolCalls: parallelLimit > 1,
                    validateSchemas: snapshot.6,
                    parser: snapshot.7.toolParser,
                    maximumCalls: parallelLimit
                )
                : []
            let timedOut = activeInference?.id == inferenceID
                && activeInference?.timedOut == true
            if timedOut, !toolCallingEnabled {
                await error(
                    connection,
                    status: 504,
                    message: LocalAPIInferencePolicy.responseTimeoutMessage,
                    dialect: dialect
                )
                return
            }
            let invalidToolResponse = toolCallingEnabled
                && detectedCalls.isEmpty
                && (
                    decoded.toolChoice != .auto
                        || LocalAPIToolCalling.containsToolCallEnvelope(output)
                        || timedOut
                )
            if invalidToolResponse {
                await error(
                    connection,
                    status: 502,
                    message: timedOut
                        ? LocalAPIInferencePolicy.toolTimeoutMessage
                        : LocalAPIToolCalling.invalidToolCallMessage,
                    dialect: dialect
                )
                return
            }
            guard LocalAPIAssistantOutputValidator.hasAnswer(
                text: cleanedOutput,
                toolCalls: detectedCalls
            ) else {
                RuntimeLogCenter.emit(
                    "Model returned no assistant answer · model=\(snapshot.0)",
                    level: .error,
                    subsystem: "api"
                )
                await error(
                    connection,
                    status: 502,
                    message: LocalAPIAssistantOutputValidator.noAnswerMessage,
                    dialect: dialect
                )
                return
            }
            let payload: Data
            switch dialect {
            case .openAIChat:
                let generationResult = inferenceSignal.result
                payload = LocalAPIResponse.openAIChatCompletion(
                    id: "chatcmpl-\(UUID().uuidString)",
                    model: snapshot.0,
                    text: cleanedOutput,
                    toolCalls: detectedCalls,
                    reasoningContent: parsedOutput.reasoning,
                    finishReason: detectedCalls.isEmpty
                        ? (generationResult?.stopReason == .length ? "length" : "stop")
                        : "tool_calls",
                    usage: generationResult.map(LocalAPIUsage.init)
                )
            case .openAIResponses:
                payload = LocalAPIResponse.openAIResponse(
                    id: "resp_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                    model: snapshot.0,
                    text: cleanedOutput,
                    toolCalls: detectedCalls,
                    reasoningContent: parsedOutput.reasoning
                )
            case .anthropic:
                payload = detectedCalls.isEmpty
                    ? LocalAPIResponse.anthropicMessage(
                        id: "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                        model: snapshot.0,
                        text: cleanedOutput,
                        reasoningContent: parsedOutput.reasoning
                    )
                    : LocalAPIResponse.anthropicToolMessage(
                        id: "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))",
                        model: snapshot.0,
                        calls: detectedCalls
                    )
            case .ollamaChat:
                payload = detectedCalls.isEmpty
                    ? LocalAPIResponse.ollamaChat(
                        model: snapshot.0,
                        text: cleanedOutput,
                        done: true,
                        reasoningContent: parsedOutput.reasoning
                    )
                    : LocalAPIResponse.ollamaToolCalls(
                        model: snapshot.0,
                        calls: detectedCalls,
                        done: true
                    )
            case .ollamaGenerate:
                payload = LocalAPIResponse.ollamaGenerate(
                    model: snapshot.0,
                    text: cleanedOutput,
                    done: true,
                    reasoningContent: parsedOutput.reasoning
                )
            }
            await respond(connection, status: 200, contentType: "application/json", data: payload)
        }
    }

    private func inferenceStream(
        messages: [ChatMessage],
        maxTokens: Int?,
        temperature: Double?,
        topP: Double?,
        jsonMode: Bool,
        forceNoThinking: Bool,
        toolConstraint: MLXToolCallConstraintConfiguration?,
        nativeTools: [AssistantNativeToolDefinition],
        nativeToolFormat: AssistantNativeToolFormat?,
        signal: LocalAPIInferenceSignal
    ) async -> AsyncStream<String> {
        AsyncStream { continuation in
            signal.attach(continuation)
            Task { @MainActor in
                CodingAssistantService.shared.generate(
                    messages: messages,
                    maxTokensOverride: maxTokens,
                    temperatureOverride: temperature,
                    topPOverride: topP,
                    jsonMode: jsonMode,
                    toolConstraint: toolConstraint,
                    nativeTools: nativeTools,
                    nativeToolFormat: nativeToolFormat,
                    forceNoThinking: forceNoThinking,
                    onToken: { continuation.yield($0) },
                    onGenerationResult: { signal.record($0) },
                    onComplete: { _ in signal.finish() }
                )
            }
            continuation.onTermination = { termination in
                // `signal.finish()` is the normal end of a completed request.
                // Do not cancel the runtime for that case: the callback can be
                // delivered after the next Hermes request has already started,
                // and an unconditional stop would cancel the new generation.
                // Only a consumer-side cancellation means the HTTP client went
                // away before the model finished.
                guard case .cancelled = termination else { return }
                Task { @MainActor in CodingAssistantService.shared.stopGeneration() }
            }
        }
    }

    /// Give a request arriving immediately after another response a small
    /// hand-off window. The gate remains single-flight; this only removes the
    /// race between the native completion callback and the next HTTP request.
    private func acquireInferenceLease() async -> UUID? {
        for _ in 0..<60 {
            if let lease = RemoteInferenceGate.shared.acquire() {
                return lease
            }
            do {
                try await Task.sleep(for: .milliseconds(50))
            } catch {
                return nil
            }
        }
        return nil
    }

    private func cancelInference() async {
        await MainActor.run { CodingAssistantService.shared.stopGeneration() }
    }

    private func cancelInference(
        inferenceID: UUID,
        reason: String,
        timedOut: Bool = false
    ) async {
        guard activeInference?.id == inferenceID else { return }
        if timedOut {
            activeInference?.timedOut = true
        }
        print("[LocalAPIServer] Cancelling inference: \(reason)")
        // End the HTTP-facing stream immediately. Runtime cancellation is
        // cooperative and some backends do not invoke onComplete until native
        // decode has fully unwound, which previously left clients generating
        // indefinitely after their deadline.
        activeInference?.signal.finish()
        await cancelInference()
    }

    private func listOpenAIModels(_ connection: NWConnection) async {
        let modelSnapshot = await MainActor.run {
            (
                CodingAssistantService.shared.activeModel,
                AppSettings.shared.localAPIToolCallingEnabled,
                AppSettings.shared.localAPIParallelToolCallsEnabled,
                LocalAPIParallelToolCallsSetting(
                    rawValue: AppSettings.shared.localAPIParallelToolCallsLimit
                )?.maximumCalls ?? LocalAPIParallelToolCallsSetting.automaticTwo.maximumCalls,
                CodingAssistantService.shared.localAPIEffectiveMaximumOutputTokens
            )
        }
        let model = modelSnapshot.0
        let profile = ModelCapabilityProfile.resolve(for: model)
        var capabilities = ["chat", "streaming"]
        if modelSnapshot.1 {
            capabilities.append("tools")
            if modelSnapshot.2,
               profile.effectiveParallelLimit(
                   requestAllowsParallel: true,
                   globalEnabled: true,
                   configuredMaximumCalls: modelSnapshot.3
               ) > 1 {
                capabilities.append("parallel_tool_calls")
            }
        }
        let payload = LocalAPIResponse.json([
            "object": "list",
            "data": [[
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "on-device",
                "root": model.repoID,
                "context_length": profile.configuredContextLength,
                "context_window_tokens": profile.configuredContextLength,
                "model_context_length": profile.modelContextLength,
                "max_kv_cache_tokens": profile.maximumKVCacheTokens,
                "max_output_tokens": modelSnapshot.4,
                "tool_parser": profile.toolParser.rawValue,
                "reasoning_parser": profile.reasoningParser.rawValue,
                "parallel_tool_call_limit": profile.effectiveParallelLimit(
                    requestAllowsParallel: true,
                    globalEnabled: modelSnapshot.2,
                    configuredMaximumCalls: modelSnapshot.3
                ),
                "context_compaction": "automatic",
                "compaction_threshold": ConversationContextCompactor.triggerFraction,
                "supports_tools": modelSnapshot.1,
                "supports_tool_calls": modelSnapshot.1,
                "capabilities": capabilities
            ]]
        ])
        await respond(connection, status: 200, contentType: "application/json", data: payload)
    }

    private func describeAPI(_ connection: NWConnection) async {
        let payload = LocalAPIResponse.json([
            "name": "On Device LAS",
            "status": "ok",
            "authentication": ["bearer", "x-api-key"],
            "openai_base_url": "/v1",
            "ollama_base_url": "/api",
            "endpoints": [
                "/v1/models",
                "/v1/models/{model}",
                "/v1/chat/completions",
                "/v1/responses",
                "/v1/messages",
                "/api/tags",
                "/api/ps",
                "/api/version",
                "/api/show",
                "/api/chat",
                "/api/generate"
            ]
        ])
        await respond(
            connection,
            status: 200,
            contentType: "application/json",
            data: payload
        )
    }

    private func retrieveOpenAIModel(
        _ request: HTTPRequest,
        connection: NWConnection
    ) async {
        let rawID = String(request.path.dropFirst("/v1/models/".count))
        guard let requested = rawID.removingPercentEncoding, !requested.isEmpty else {
            await error(
                connection,
                status: 404,
                message: "Model not found",
                dialect: .openAIChat
            )
            return
        }

        let snapshot = await MainActor.run {
            (
                CodingAssistantService.shared.activeModel,
                CodingAssistantService.shared.localAPIEffectiveMaximumOutputTokens
            )
        }
        guard LocalAPIValidation.modelMatches(
            requested,
            id: snapshot.0.id,
            repoID: snapshot.0.repoID
        ) else {
            await error(
                connection,
                status: 404,
                message: "Model '\(requested)' not found",
                dialect: .openAIChat
            )
            return
        }

        let payload = LocalAPIResponse.json(
            openAIModelObject(snapshot.0, effectiveMaximumOutputTokens: snapshot.1)
        )
        await respond(
            connection,
            status: 200,
            contentType: "application/json",
            data: payload
        )
    }

    private func openAIModelObject(
        _ model: AssistantModel,
        effectiveMaximumOutputTokens: Int
    ) -> [String: Any] {
        let profile = ModelCapabilityProfile.resolve(for: model)
        return [
            "id": model.id,
            "object": "model",
            "created": 0,
            "owned_by": "on-device",
            "root": model.repoID,
            "context_length": profile.configuredContextLength,
            "max_output_tokens": effectiveMaximumOutputTokens
        ]
    }

    private func listOllamaModels(_ connection: NWConnection) async {
        let model = await MainActor.run { CodingAssistantService.shared.activeModel }
        let payload = LocalAPIResponse.json([
            "models": [[
                "name": model.id,
                "model": model.id,
                "modified_at": ISO8601DateFormatter().string(from: Date()),
                "size": model.downloadSizeBytes ?? 0,
                "digest": "",
                "details": [
                    "parent_model": "",
                    "family": model.repoID,
                    "families": [model.repoID],
                    "format": model.runtime == .llamaCpp ? "gguf" : "mlx"
                ]
            ]]
        ])
        await respond(connection, status: 200, contentType: "application/json", data: payload)
    }

    private func listRunningOllamaModels(_ connection: NWConnection) async {
        let snapshot = await MainActor.run {
            (
                CodingAssistantService.shared.activeModel,
                CodingAssistantService.shared.state,
                ProcessInfo.processInfo.physicalMemory
            )
        }
        let isResident: Bool
        switch snapshot.1 {
        case .ready, .generating:
            isResident = true
        case .unloaded, .loading, .failed:
            isResident = false
        }

        let models: [[String: Any]]
        if isResident {
            let profile = ModelCapabilityProfile.resolve(for: snapshot.0)
            models = [[
                "name": snapshot.0.id,
                "model": snapshot.0.id,
                "size": snapshot.0.downloadSizeBytes ?? 0,
                "digest": "",
                "details": [
                    "parent_model": "",
                    "format": snapshot.0.runtime == .llamaCpp ? "gguf" : "mlx",
                    "family": snapshot.0.repoID,
                    "families": [snapshot.0.repoID]
                ],
                "expires_at": ISO8601DateFormatter().string(
                    from: Date().addingTimeInterval(300)
                ),
                "size_vram": snapshot.2,
                "context_length": profile.configuredContextLength
            ]]
        } else {
            models = []
        }

        await respond(
            connection,
            status: 200,
            contentType: "application/json",
            data: LocalAPIResponse.json(["models": models])
        )
    }

    private func showOllamaVersion(_ connection: NWConnection) async {
        let version = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0"
        await respond(
            connection,
            status: 200,
            contentType: "application/json",
            data: LocalAPIResponse.json(["version": version])
        )
    }

    private func showOllamaModel(_ request: HTTPRequest, connection: NWConnection) async {
        guard request.headers["content-type"]?.lowercased().hasPrefix("application/json") == true,
              let body = request.body,
              let raw = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            await error(connection, status: 400, message: "JSON body required", dialect: .ollamaChat)
            return
        }
        let model = await MainActor.run { CodingAssistantService.shared.activeModel }
        let profile = ModelCapabilityProfile.resolve(for: model)
        var capabilities = ["completion"]
        if model.supportsTools {
            capabilities.append("tools")
        }
        if model.supportsThinking {
            capabilities.append("thinking")
        }
        let requested = (raw["model"] ?? raw["name"]) as? String
        guard requested == nil || requested?.isEmpty == true
                || LocalAPIValidation.modelMatches(requested!, id: model.id, repoID: model.repoID) else {
            await error(connection, status: 404, message: "Model '\(requested!)' is not loaded", dialect: .ollamaChat)
            return
        }
        let payload = LocalAPIResponse.json([
            "license": "",
            "modelfile": "",
            "parameters": "num_ctx \(profile.configuredContextLength)",
            "template": "",
            "capabilities": capabilities,
            "modified_at": ISO8601DateFormatter().string(from: Date()),
            "details": [
                "parent_model": "",
                "family": model.repoID,
                "families": [model.repoID],
                "format": model.runtime == .llamaCpp ? "gguf" : "mlx"
            ],
            "model_info": [
                "general.architecture": model.repoID,
                "\(model.repoID).context_length": profile.configuredContextLength
            ]
        ])
        await respond(connection, status: 200, contentType: "application/json", data: payload)
    }

    private func sendStreamingTextCompletion(
        _ connection: NWConnection,
        dialect: Dialect,
        model: String,
        requestID: String,
        text: String,
        reasoningContent: String = "",
        generationResult: AssistantGenerationResult? = nil,
        includeUsage: Bool = false
    ) async throws {
        switch dialect {
        case .openAIChat:
            var payload = Data("data: ".utf8)
            payload.append(LocalAPIResponse.openAIChunk(
                        id: requestID,
                        model: model,
                        text: text,
                        role: "assistant",
                        reasoningContent: reasoningContent.isEmpty ? nil : reasoningContent
                    ))
            payload.append(Data("\n\n".utf8))
            payload.append(LocalAPIResponse.openAIStreamTerminator(
                id: requestID,
                model: model,
                finishReason: generationResult?.stopReason == .length
                    ? "length"
                    : "stop",
                usage: generationResult.map(LocalAPIUsage.init),
                includeUsage: includeUsage
            ))
            try await sendFinal(connection, payload)
        case .openAIResponses:
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.output_text.delta",
                fields: [
                    "response_id": requestID,
                    "item_id": "msg_\(requestID)",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text
                ]
            ))
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.output_text.done",
                fields: [
                        "response_id": requestID,
                        "item_id": "msg_\(requestID)",
                        "output_index": 0,
                        "content_index": 0,
                        "text": text,
                        "reasoning_content": reasoningContent
                    ]
            ))
            try await sendFinal(connection, LocalAPIResponse.openAIResponseEvent(
                "response.completed",
                fields: [
                    "response": (try? JSONSerialization.jsonObject(with: LocalAPIResponse.openAIResponse(
                        id: requestID,
                        model: model,
                        text: text,
                        reasoningContent: reasoningContent
                    ))) ?? [:]
                ]
            ))
        case .anthropic:
            try await sendAnthropicTextCompletion(connection, text: text)
        case .ollamaChat:
            try await sendFinal(
                connection,
                LocalAPIResponse.ollamaChat(
                    model: model,
                    text: text,
                    done: true,
                    reasoningContent: reasoningContent
                )
                    + Data("\n".utf8)
            )
        case .ollamaGenerate:
            try await sendFinal(
                connection,
                LocalAPIResponse.ollamaGenerate(
                    model: model,
                    text: text,
                    done: true,
                    reasoningContent: reasoningContent
                )
                    + Data("\n".utf8)
            )
        }
    }

    private func sendStreamingModelError(
        _ connection: NWConnection,
        dialect: Dialect
    ) async throws {
        let message = LocalAPIAssistantOutputValidator.noAnswerMessage
        switch dialect {
        case .openAIChat:
            try await sendFinal(
                connection,
                Data("data: ".utf8)
                    + LocalAPIResponse.json([
                        "error": ["type": "model_no_answer", "message": message]
                    ])
                    + Data("\n\ndata: [DONE]\n\n".utf8)
            )
        case .openAIResponses:
            try await sendFinal(
                connection,
                LocalAPIResponse.openAIResponseEvent(
                    "error",
                    fields: [
                        "error": ["type": "model_no_answer", "message": message]
                    ]
                )
            )
        case .anthropic:
            try await sendFinal(
                connection,
                LocalAPIResponse.anthropicEvent(
                    "error",
                    object: [
                        "type": "error",
                        "error": ["type": "model_no_answer", "message": message]
                    ]
                )
            )
        case .ollamaChat, .ollamaGenerate:
            try await sendFinal(
                connection,
                LocalAPIResponse.json([
                    "error": message,
                    "error_type": "model_no_answer"
                ]) + Data("\n".utf8)
            )
        }
    }

    private func sendStreamingGenerationTimeout(
        _ connection: NWConnection,
        dialect: Dialect
    ) async throws {
        let message = LocalAPIInferencePolicy.responseTimeoutMessage
        switch dialect {
        case .openAIChat:
            try await sendFinal(
                connection,
                Data("data: ".utf8)
                    + LocalAPIResponse.json([
                        "error": [
                            "type": "generation_timeout",
                            "message": message
                        ]
                    ])
                    + Data("\n\ndata: [DONE]\n\n".utf8)
            )
        case .openAIResponses:
            try await sendFinal(
                connection,
                LocalAPIResponse.openAIResponseEvent(
                    "error",
                    fields: [
                        "error": [
                            "type": "generation_timeout",
                            "message": message
                        ]
                    ]
                )
            )
        case .anthropic:
            try await sendFinal(
                connection,
                LocalAPIResponse.anthropicEvent(
                    "error",
                    object: [
                        "type": "error",
                        "error": [
                            "type": "generation_timeout",
                            "message": message
                        ]
                    ]
                )
            )
        case .ollamaChat, .ollamaGenerate:
            try await sendFinal(
                connection,
                LocalAPIResponse.json([
                    "error": message,
                    "error_type": "generation_timeout"
                ]) + Data("\n".utf8)
            )
        }
    }

    private func sendOpenAIResponseToolCompletion(
        _ connection: NWConnection,
        requestID: String,
        model: String,
        calls: [LocalAPIToolCall]
    ) async throws {
        for (index, call) in calls.enumerated() {
            let item: [String: Any] = [
                "id": "fc_\(call.id)",
                "type": "function_call",
                "status": "in_progress",
                "call_id": call.id,
                "name": call.name,
                "arguments": ""
            ]
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.output_item.added",
                fields: [
                    "response_id": requestID,
                    "output_index": index,
                    "item": item
                ]
            ))
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.function_call_arguments.delta",
                fields: [
                    "response_id": requestID,
                    "item_id": "fc_\(call.id)",
                    "output_index": index,
                    "delta": call.argumentsJSON
                ]
            ))
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.function_call_arguments.done",
                fields: [
                    "response_id": requestID,
                    "item_id": "fc_\(call.id)",
                    "output_index": index,
                    "arguments": call.argumentsJSON
                ]
            ))
            var completedItem = item
            completedItem["status"] = "completed"
            completedItem["arguments"] = call.argumentsJSON
            try await send(connection, LocalAPIResponse.openAIResponseEvent(
                "response.output_item.done",
                fields: [
                    "response_id": requestID,
                    "output_index": index,
                    "item": completedItem
                ]
            ))
        }
        try await sendFinal(connection, LocalAPIResponse.openAIResponseEvent(
            "response.completed",
            fields: [
                "response": (try? JSONSerialization.jsonObject(with: LocalAPIResponse.openAIResponse(
                    id: requestID,
                    model: model,
                    text: "",
                    toolCalls: calls
                ))) ?? [:]
            ]
        ))
    }

    private func sendAnthropicTextCompletion(
        _ connection: NWConnection,
        text: String
    ) async throws {
        try await send(connection, LocalAPIResponse.anthropicEvent(
            "content_block_start",
            object: [
                "type": "content_block_start",
                "index": 0,
                "content_block": ["type": "text", "text": ""]
            ]
        ))
        try await send(connection, LocalAPIResponse.anthropicEvent(
            "content_block_delta",
            object: [
                "type": "content_block_delta",
                "index": 0,
                "delta": ["type": "text_delta", "text": text]
            ]
        ))
        try await send(connection, LocalAPIResponse.anthropicEvent(
            "content_block_stop",
            object: ["type": "content_block_stop", "index": 0]
        ))
        try await send(connection, LocalAPIResponse.anthropicEvent(
            "message_delta",
            object: [
                "type": "message_delta",
                "delta": [
                    "stop_reason": "end_turn",
                    "stop_sequence": NSNull()
                ],
                "usage": ["output_tokens": 0]
            ]
        ))
        try await sendFinal(connection, LocalAPIResponse.anthropicEvent(
            "message_stop",
            object: ["type": "message_stop"]
        ))
    }

    private func sendAnthropicToolCompletion(
        _ connection: NWConnection,
        calls: [LocalAPIToolCall]
    ) async throws {
        for (index, call) in calls.enumerated() {
            try await send(connection, LocalAPIResponse.anthropicEvent(
                "content_block_start",
                object: [
                    "type": "content_block_start",
                    "index": index,
                    "content_block": [
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": [:]
                    ]
                ]
            ))
            try await send(connection, LocalAPIResponse.anthropicEvent(
                "content_block_delta",
                object: [
                    "type": "content_block_delta",
                    "index": index,
                    "delta": [
                        "type": "input_json_delta",
                        "partial_json": call.argumentsJSON
                    ]
                ]
            ))
            try await send(connection, LocalAPIResponse.anthropicEvent(
                "content_block_stop",
                object: ["type": "content_block_stop", "index": index]
            ))
        }
        try await send(connection, LocalAPIResponse.anthropicEvent(
            "message_delta",
            object: [
                "type": "message_delta",
                "delta": [
                    "stop_reason": "tool_use",
                    "stop_sequence": NSNull()
                ],
                "usage": ["output_tokens": 0]
            ]
        ))
        try await sendFinal(connection, LocalAPIResponse.anthropicEvent(
            "message_stop",
            object: ["type": "message_stop"]
        ))
    }

    private func writeStreamingHeaders(
        _ connection: NWConnection,
        dialect: Dialect,
        model: String
    ) async {
        let contentType = (dialect == .openAIChat || dialect == .openAIResponses || dialect == .anthropic)
            ? "text/event-stream"
            : "application/x-ndjson"
        let header = """
        HTTP/1.1 200 OK\r
        Content-Type: \(contentType)\r
        Cache-Control: no-cache, no-transform\r
        X-Content-Type-Options: nosniff\r
        X-Accel-Buffering: no\r
        Access-Control-Allow-Origin: *\r
        Access-Control-Allow-Methods: GET, POST, OPTIONS\r
        Access-Control-Allow-Headers: Authorization, Content-Type, X-API-Key, Anthropic-Version, Anthropic-Beta\r
        Access-Control-Allow-Private-Network: true\r
        Connection: close\r
        \r

        """
        try? await send(connection, Data(header.utf8))
        if dialect == .anthropic {
            let messageID = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: ""))"
            try? await send(connection, LocalAPIResponse.anthropicEvent("message_start", object: [
                "type": "message_start",
                "message": [
                    "id": messageID, "type": "message", "role": "assistant",
                    "content": [], "model": model,
                    "stop_reason": NSNull(), "stop_sequence": NSNull(),
                    "usage": ["input_tokens": 0, "output_tokens": 0]
                ]
            ]))
        }
    }

    private func error(_ connection: NWConnection, status: Int, message: String, dialect: Dialect) async {
        let payload: Data
        if dialect == .anthropic {
            payload = LocalAPIResponse.json([
                "type": "error",
                "error": ["type": "invalid_request_error", "message": message]
            ])
        } else if dialect == .openAIChat || dialect == .openAIResponses {
            payload = LocalAPIResponse.json(["error": ["message": message, "type": "invalid_request_error"]])
        } else {
            payload = LocalAPIResponse.json(["error": message])
        }
        await respond(connection, status: status, contentType: "application/json", data: payload)
    }

    private func respond(_ connection: NWConnection, status: Int, contentType: String, data: Data) async {
        let reason: String
        switch status {
        case 200: reason = "OK"
        case 204: reason = "No Content"
        case 400: reason = "Bad Request"
        case 401: reason = "Unauthorized"
        case 404: reason = "Not Found"
        case 408: reason = "Request Timeout"
        case 415: reason = "Unsupported Media Type"
        case 413: reason = "Payload Too Large"
        case 422: reason = "Unprocessable Entity"
        case 502: reason = "Bad Gateway"
        case 503: reason = "Service Unavailable"
        case 504: reason = "Gateway Timeout"
        default: reason = "Internal Server Error"
        }
        var header = "HTTP/1.1 \(status) \(reason)\r\n"
        header += "Content-Type: \(contentType)\r\n"
        header += "Content-Length: \(data.count)\r\n"
        header += "Cache-Control: no-store\r\n"
        header += "X-Content-Type-Options: nosniff\r\n"
        if status == 401 {
            header += "WWW-Authenticate: Bearer\r\n"
        }
        header += "Access-Control-Allow-Origin: *\r\n"
        header += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        header += "Access-Control-Allow-Headers: Authorization, Content-Type, X-API-Key, Anthropic-Version, Anthropic-Beta\r\n"
        header += "Access-Control-Allow-Private-Network: true\r\n"
        header += "Connection: close\r\n\r\n"
        var payload = Data(header.utf8)
        payload.append(data)
        try? await sendFinal(connection, payload)
        RuntimeLogCenter.emit(
            "HTTP response \(status) (\(data.count) bytes)",
            subsystem: "api"
        )
        connection.cancel()
    }

    private func send(_ connection: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error { continuation.resume(throwing: error) }
                else { continuation.resume() }
            })
        }
    }

    /// Finish the HTTP response with a graceful write-side close. Calling
    /// `cancel()` after an ordinary send can turn the close into a TCP reset;
    /// command-line clients often tolerate that, while browser fetch/SSE
    /// stacks keep waiting for a clean end-of-stream.
    private func sendFinal(_ connection: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            connection.send(
                content: data,
                contentContext: .finalMessage,
                isComplete: true,
                completion: .contentProcessed { error in
                    if let error { continuation.resume(throwing: error) }
                    else { continuation.resume() }
                }
            )
        }
    }
}

// MARK: - Minimal HTTP/1.1 request parser

/// The server intentionally supports only the small HTTP/1.1 subset needed by
/// local SDK clients. Keep this parser in the local-API target so the dormant
/// Mac bridge server is not required just to decode a request.
struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data?

    init?(data: Data) {
        let separator = Data("\r\n\r\n".utf8)
        guard let separatorRange = data.range(of: separator),
              let headerSection = String(
                data: data[..<separatorRange.lowerBound],
                encoding: .utf8
              ) else {
            return nil
        }

        var lines = headerSection.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        lines.removeFirst()

        let parts = requestLine.split(
            separator: " ",
            maxSplits: 2,
            omittingEmptySubsequences: true
        )
        guard parts.count >= 2 else { return nil }
        method = String(parts[0]).uppercased()

        let rawPath = String(parts[1])
        path = rawPath.split(separator: "?", maxSplits: 1).first.map(String.init) ?? ""

        var parsedHeaders: [String: String] = [:]
        for line in lines {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            let value = line[line.index(after: colon)...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            parsedHeaders[key] = value
        }
        headers = parsedHeaders

        let headerByteCount = separatorRange.upperBound
        let declaredLength = Int(parsedHeaders["content-length"] ?? "0") ?? 0
        guard declaredLength >= 0 else { return nil }
        guard data.count >= headerByteCount + declaredLength else { return nil }
        body = declaredLength == 0
            ? nil
            : Data(data[headerByteCount..<(headerByteCount + declaredLength)])
    }
}

final class LocalAPIConnectionMonitor: @unchecked Sendable {
    private let lock = NSLock()
    private var disconnected = false
    private var stopped = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func markDisconnected() {
        lock.lock()
        guard !stopped, !disconnected else {
            lock.unlock()
            return
        }
        disconnected = true
        let pending = waiters
        waiters.removeAll(keepingCapacity: false)
        lock.unlock()
        pending.forEach { $0.resume() }
    }

    func stop() {
        lock.lock()
        guard !stopped else {
            lock.unlock()
            return
        }
        stopped = true
        let pending = waiters
        waiters.removeAll(keepingCapacity: false)
        lock.unlock()
        pending.forEach { $0.resume() }
    }

    func wait() async {
        await withTaskCancellationHandler {
            await withCheckedContinuation {
                (continuation: CheckedContinuation<Void, Never>) in
                lock.lock()
                if disconnected || stopped {
                    lock.unlock()
                    continuation.resume()
                } else {
                    waiters.append(continuation)
                    lock.unlock()
                }
            }
        } onCancel: {
            stop()
        }
    }
}

private final class LocalAPIResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Data, Error>?
    private var timeoutTask: Task<Void, Never>?

    init(_ continuation: CheckedContinuation<Data, Error>) {
        self.continuation = continuation
    }

    func setTimeoutTask(_ task: Task<Void, Never>) {
        lock.lock()
        guard continuation != nil else {
            lock.unlock()
            task.cancel()
            return
        }
        timeoutTask = task
        lock.unlock()
    }

    @discardableResult
    func resume(returning data: Data) -> Bool {
        lock.lock()
        let pending = continuation
        continuation = nil
        let timeout = timeoutTask
        timeoutTask = nil
        lock.unlock()
        timeout?.cancel()
        pending?.resume(returning: data)
        return pending != nil
    }

    @discardableResult
    func resume(throwing error: Error) -> Bool {
        lock.lock()
        let pending = continuation
        continuation = nil
        let timeout = timeoutTask
        timeoutTask = nil
        lock.unlock()
        timeout?.cancel()
        pending?.resume(throwing: error)
        return pending != nil
    }
}

final class LocalAPIInferenceSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: AsyncStream<String>.Continuation?
    private var finished = false
    private var storedResult: AssistantGenerationResult?

    func attach(_ continuation: AsyncStream<String>.Continuation) {
        lock.lock()
        if finished {
            lock.unlock()
            continuation.finish()
            return
        }
        self.continuation = continuation
        lock.unlock()
    }

    func finish() {
        lock.lock()
        finished = true
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.finish()
    }

    func record(_ result: AssistantGenerationResult) {
        lock.lock()
        storedResult = result
        lock.unlock()
    }

    var result: AssistantGenerationResult? {
        lock.lock()
        defer { lock.unlock() }
        return storedResult
    }
}

@MainActor
final class LocalAPIManager: ObservableObject {
    static let shared = LocalAPIManager()

    enum State: Equatable {
        case stopped
        case starting
        case running(port: UInt16)
        case failed(String)
    }

    @Published private(set) var state: State = .stopped
    @Published private(set) var addresses: [String] = []
    @Published private(set) var apiKey = LocalAPIKeyStore.key()

    private let server = LocalAPIServer()
    private let pathMonitor = NWPathMonitor()
    private let pathQueue = DispatchQueue(label: "com.mesutcydev.ioslocalllm.local-api-network")
    private var shouldRun = false
    private var recoveryAttempt = 0
    private var recoveryTask: Task<Void, Never>?

    private init() {
        pathMonitor.pathUpdateHandler = { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshAddresses()
            }
        }
        pathMonitor.start(queue: pathQueue)
    }

    func start() async {
        guard state == .stopped || isFailed else { return }
        shouldRun = true
        guard let port = LocalAPIValidation.validPort(AppSettings.shared.localAPIPort) else {
            state = .failed("Port must be between 1024 and 65535.")
            RuntimeLogCenter.emit(
                "Server refused to start: invalid port",
                level: .error,
                subsystem: "server"
            )
            return
        }
        updateIdleTimer(running: true)
        state = .starting
        RuntimeLogCenter.emit("Starting local API server on port \(port)", subsystem: "server")
        addresses = Self.localIPv4Addresses()
        await server.setEventHandler { [weak self] event in
            Task { @MainActor in
                guard let self else { return }
                switch event {
                case .ready:
                    self.recoveryAttempt = 0
                    self.state = .running(port: port)
                    RuntimeLogCenter.emit("Server listening on port \(port)", subsystem: "server")
                case .failed(let message):
                    self.state = .failed(message)
                    self.updateIdleTimer(running: false)
                    RuntimeLogCenter.emit(
                        "Server failed: \(message)",
                        level: .error,
                        subsystem: "server"
                    )
                    self.scheduleRecovery()
                }
            }
        }
        do {
            try await server.start(port: port)
        } catch {
            state = .failed(error.localizedDescription)
            updateIdleTimer(running: false)
            RuntimeLogCenter.emit(
                "Server start failed: \(error.localizedDescription)",
                level: .error,
                subsystem: "server"
            )
            scheduleRecovery()
        }
    }

    func stop() async {
        shouldRun = false
        recoveryTask?.cancel()
        recoveryTask = nil
        recoveryAttempt = 0
        await server.stop()
        state = .stopped
        updateIdleTimer(running: false)
        RuntimeLogCenter.emit("Server stopped", subsystem: "server")
    }

    func restart() async {
        await stop()
        await start()
    }

    func refreshIdleTimerPolicy() {
        switch state {
        case .starting, .running:
            updateIdleTimer(running: true)
        case .stopped, .failed:
            updateIdleTimer(running: false)
        }
    }

    func rotateKey() {
        apiKey = LocalAPIKeyStore.rotate()
    }

    func refreshAddresses() {
        addresses = Self.localIPv4Addresses()
    }

    var isFailed: Bool {
        if case .failed = state { return true }
        return false
    }

    static func localIPv4Addresses() -> [String] {
        var pointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&pointer) == 0, let first = pointer else { return [] }
        defer { freeifaddrs(pointer) }
        var results = Set<String>()
        var current: UnsafeMutablePointer<ifaddrs>? = first
        while let interface = current {
            defer { current = interface.pointee.ifa_next }
            let flags = Int32(interface.pointee.ifa_flags)
            let name = String(cString: interface.pointee.ifa_name)
            guard flags & IFF_UP != 0,
                  flags & IFF_LOOPBACK == 0,
                  LocalAPIValidation.isReachableLANInterface(name),
                  interface.pointee.ifa_addr.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let result = getnameinfo(
                interface.pointee.ifa_addr,
                socklen_t(interface.pointee.ifa_addr.pointee.sa_len),
                &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST
            )
            if result == 0 { results.insert(String(cString: host)) }
        }
        return results.sorted()
    }

    private func updateIdleTimer(running: Bool) {
        UIApplication.shared.isIdleTimerDisabled =
            running && AppSettings.shared.localAPIKeepScreenAwake
    }

    private func scheduleRecovery() {
        guard shouldRun,
              AppSettings.shared.localAPIEnabled,
              UIApplication.shared.applicationState == .active,
              recoveryTask == nil,
              recoveryAttempt < 5 else { return }

        recoveryAttempt += 1
        let attempt = recoveryAttempt
        let delay = min(pow(2.0, Double(attempt - 1)), 8.0)
        RuntimeLogCenter.emit(
            "Server recovery scheduled · attempt=\(attempt) · delay=\(Int(delay))s",
            level: .warning,
            subsystem: "server"
        )
        recoveryTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(delay))
            } catch {
                return
            }
            guard let self else { return }
            self.recoveryTask = nil
            guard self.shouldRun,
                  AppSettings.shared.localAPIEnabled,
                  UIApplication.shared.applicationState == .active else { return }
            await self.start()
        }
    }
}
