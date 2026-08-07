import Foundation
import MLX
import MLXLMCommon

// MARK: - MLX tool-decision grammar

/// The local API has to make a provider-level decision before it can send a
/// response: ordinary text or one (or more) declared tool calls. Prompt
/// instructions alone are not reliable enough for small local models, so the
/// MLX path uses this compact grammar as a logit-level constraint.
///
/// The grammar is deliberately an app-owned protocol rather than a model
/// chat-template protocol. That keeps Qwen, Ornith, Llama, and future MLX
/// models on the same API-facing wire shape. The server removes the private
/// envelope before returning the provider-native response.
struct MLXToolCallConstraintConfiguration: Equatable, Sendable {
    enum Decision: Equatable, Sendable {
        case automatic
        case required
    }

    let toolNames: [String]
    let decision: Decision
    let allowParallelCalls: Bool
    let allowReasoningPrefixes: Bool

    init(
        toolNames: [String],
        decision: Decision,
        allowParallelCalls: Bool,
        allowReasoningPrefixes: Bool
    ) {
        self.toolNames = Array(Set(toolNames)).sorted()
        self.decision = decision
        self.allowParallelCalls = allowParallelCalls
        self.allowReasoningPrefixes = allowReasoningPrefixes
    }

    /// Avoid turning a very large or pathological tool list into a large
    /// vocabulary scan. The protocol parser remains available as a safe
    /// fallback when a request exceeds this bounded native-constraint path.
    var isSuitableForNativeConstraint: Bool {
        !toolNames.isEmpty
            && toolNames.count <= 128
            && toolNames.allSatisfy { !$0.isEmpty && $0.count <= 128 }
    }
}

/// A character-level JSON-prefix parser. It accepts every prefix that can
/// still become one valid JSON value and reports the exact point at which the
/// root value is complete. It is used only for tool arguments; the outer
/// response envelope has a smaller dedicated state machine below.
private struct MLXJSONPrefixParser {
    enum Result {
        case accepted
        case completed
    }

    private enum ContainerKind {
        case object
        case array
    }

    private enum ContainerState {
        case objectKeyOrEnd
        case objectColon
        case objectValue
        case objectAfterValue
        case arrayValueOrEnd
        case arrayAfterValue
    }

    private struct Frame {
        let kind: ContainerKind
        var state: ContainerState
    }

    private enum Mode {
        case structural
        case stringKey(MLXJSONStringPrefix)
        case stringValue(MLXJSONStringPrefix)
        case number(String)
        case literal(expected: String, index: Int)
    }

    private var stack: [Frame] = []
    private var mode: Mode = .structural
    private var rootStarted = false
    private var rootCompleted = false
    private let rootMustBeObject: Bool

    init(rootMustBeObject: Bool) {
        self.rootMustBeObject = rootMustBeObject
    }

    var isComplete: Bool {
        rootCompleted && stack.isEmpty && isStructural
    }

    var allowsPlainJSONStringToken: Bool {
        switch mode {
        case .stringKey(let value), .stringValue(let value):
            value.allowsPlainText
        default:
            false
        }
    }

    /// A cheap first-character filter used before the exact prefix parser.
    /// `nil` means that the current state accepts arbitrary leading text (the
    /// string and reasoning cases); an empty set means no token can advance.
    var allowedLeadingCharacters: Set<Character>? {
        switch mode {
        case .stringKey, .stringValue:
            return nil
        case .number:
            return Set(["-", ".", "+", "e", "E", " ", "\n", "\r", "\t", ",", "}", "]"])
                .union(Set(Array("0123456789")))
        case .literal(let expected, let index):
            if index < expected.count {
                return [expected.character(at: index)]
            }
            return Set([" ", "\n", "\r", "\t", ",", "}", "]"])
        case .structural:
            if stack.isEmpty {
                if rootCompleted { return [] }
                if !rootStarted {
                    return rootMustBeObject
                        ? Set([" ", "\n", "\r", "\t", "{"])
                        : jsonValueLeadingCharacters
                }
                return []
            }

            switch stack[stack.count - 1].state {
            case .objectKeyOrEnd:
                return Set([" ", "\n", "\r", "\t", "\"", "}"])
            case .objectColon:
                return Set([" ", "\n", "\r", "\t", ":"])
            case .objectValue:
                return jsonValueLeadingCharacters
            case .objectAfterValue:
                return Set([" ", "\n", "\r", "\t", ",", "}"])
            case .arrayValueOrEnd:
                return Set([" ", "\n", "\r", "\t", "]"])
                    .union(jsonValueLeadingCharacters)
            case .arrayAfterValue:
                return Set([" ", "\n", "\r", "\t", ",", "]"])
            }
        }
    }

    mutating func consume(_ character: Character) -> Result? {
        let character = character

        while true {
            switch mode {
            case .stringKey(var value):
                guard value.consume(character) else { return nil }
                if value.isClosed {
                    guard let frame = stack.last, frame.kind == .object else {
                        return nil
                    }
                    mode = .structural
                    stack[stack.count - 1].state = .objectColon
                } else {
                    mode = .stringKey(value)
                }
                return .accepted

            case .stringValue(var value):
                guard value.consume(character) else { return nil }
                if value.isClosed {
                    mode = .structural
                    completeValue()
                    return rootCompleted ? .completed : .accepted
                }
                mode = .stringValue(value)
                return .accepted

            case .number(let number):
                if isNumberCharacter(character) {
                    let next = number + String(character)
                    guard isNumberPrefix(next) else { return nil }
                    mode = .number(next)
                    return .accepted
                }

                guard isJSONDelimiter(character), isCompleteNumber(number) else {
                    return nil
                }
                mode = .structural
                completeValue()
                if rootCompleted {
                    return .completed
                }
                // The delimiter belongs to the containing object/array, not
                // to the number. Re-run it through the structural state.
                continue

            case .literal(let expected, let index):
                if index < expected.count {
                    let expectedCharacter = expected.character(at: index)
                    guard character == expectedCharacter else { return nil }
                    mode = .literal(expected: expected, index: index + 1)
                    return .accepted
                }

                guard isJSONDelimiter(character) else { return nil }
                mode = .structural
                completeValue()
                if rootCompleted {
                    return .completed
                }
                continue

            case .structural:
                return consumeStructural(character)
            }
        }
    }

    private var isStructural: Bool {
        if case .structural = mode { return true }
        return false
    }

    private var jsonValueLeadingCharacters: Set<Character> {
        Set(["{", "[", "\"", "-", "t", "f", "n"])
            .union(Set(Array("0123456789")))
            .union(Set([" ", "\n", "\r", "\t"]))
    }

    private mutating func consumeStructural(_ character: Character) -> Result? {
        if stack.isEmpty {
            guard !rootCompleted else { return nil }
            if !rootStarted, isJSONWhitespace(character) {
                return .accepted
            }
            return startValue(character)
        }

        let index = stack.count - 1
        let frame = stack[index]
        switch frame.state {
        case .objectKeyOrEnd:
            if isJSONWhitespace(character) {
                return .accepted
            }
            if character == "}" {
                stack.removeLast()
                completeValue()
                return rootCompleted ? .completed : .accepted
            }
            guard character == "\"" else { return nil }
            mode = .stringKey(MLXJSONStringPrefix())
            return .accepted

        case .objectColon:
            if isJSONWhitespace(character) {
                return .accepted
            }
            guard character == ":" else { return nil }
            stack[index].state = .objectValue
            return .accepted

        case .objectValue:
            if isJSONWhitespace(character) {
                return .accepted
            }
            return startValue(character)

        case .objectAfterValue:
            if isJSONWhitespace(character) {
                return .accepted
            }
            if character == "," {
                stack[index].state = .objectKeyOrEnd
                return .accepted
            }
            if character == "}" {
                stack.removeLast()
                completeValue()
                return rootCompleted ? .completed : .accepted
            }
            return nil

        case .arrayValueOrEnd:
            if isJSONWhitespace(character) {
                return .accepted
            }
            if character == "]" {
                stack.removeLast()
                completeValue()
                return rootCompleted ? .completed : .accepted
            }
            return startValue(character)

        case .arrayAfterValue:
            if isJSONWhitespace(character) {
                return .accepted
            }
            if character == "," {
                stack[index].state = .arrayValueOrEnd
                return .accepted
            }
            if character == "]" {
                stack.removeLast()
                completeValue()
                return rootCompleted ? .completed : .accepted
            }
            return nil
        }
    }

    private mutating func startValue(_ character: Character) -> Result? {
        if stack.isEmpty {
            guard !rootStarted else { return nil }
            if rootMustBeObject, character != "{" {
                return nil
            }
            rootStarted = true
        } else {
            let state = stack[stack.count - 1].state
            switch state {
            case .objectValue, .arrayValueOrEnd:
                break
            default:
                return nil
            }
        }

        switch character {
        case "{":
            stack.append(Frame(kind: .object, state: .objectKeyOrEnd))
            return .accepted
        case "[":
            stack.append(Frame(kind: .array, state: .arrayValueOrEnd))
            return .accepted
        case "\"":
            mode = .stringValue(MLXJSONStringPrefix())
            return .accepted
        case "-":
            mode = .number(String(character))
            return .accepted
        case "t":
            mode = .literal(expected: "true", index: 1)
            return .accepted
        case "f":
            mode = .literal(expected: "false", index: 1)
            return .accepted
        case "n":
            mode = .literal(expected: "null", index: 1)
            return .accepted
        default:
            guard isASCIIJSONDigit(character) else { return nil }
            mode = .number(String(character))
            return .accepted
        }
    }

    private mutating func completeValue() {
        guard !stack.isEmpty else {
            rootCompleted = true
            return
        }
        let index = stack.count - 1
        switch stack[index].kind {
        case .object:
            stack[index].state = .objectAfterValue
        case .array:
            stack[index].state = .arrayAfterValue
        }
    }

    private func isNumberCharacter(_ character: Character) -> Bool {
        character == "-"
            || character == "."
            || character == "+"
            || character == "e"
            || character == "E"
            || isASCIIJSONDigit(character)
    }

    private func isCompleteNumber(_ value: String) -> Bool {
        let characters = Array(value)
        guard !characters.isEmpty else { return false }
        var index = 0
        if characters[index] == "-" {
            index += 1
            guard index < characters.count else { return false }
        }

        if characters[index] == "0" {
            index += 1
            if index < characters.count, isASCIIJSONDigit(characters[index]) {
                return false
            }
        } else {
            guard isASCIIJSONDigit(characters[index]), characters[index] != "0" else {
                return false
            }
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
        }

        if index < characters.count, characters[index] == "." {
            index += 1
            let start = index
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
            guard index > start else { return false }
        }

        if index < characters.count, characters[index] == "e" || characters[index] == "E" {
            index += 1
            if index < characters.count,
               characters[index] == "+" || characters[index] == "-" {
                index += 1
            }
            let start = index
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
            guard index > start else { return false }
        }

        return index == characters.count
    }

    private func isNumberPrefix(_ value: String) -> Bool {
        let characters = Array(value)
        guard !characters.isEmpty else { return false }
        var index = 0

        if characters[index] == "-" {
            index += 1
            if index == characters.count { return true }
        }

        guard index < characters.count, isASCIIJSONDigit(characters[index]) else {
            return false
        }
        if characters[index] == "0" {
            index += 1
            if index < characters.count, isASCIIJSONDigit(characters[index]) {
                return false
            }
        } else {
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
        }

        if index < characters.count, characters[index] == "." {
            index += 1
            let fractionStart = index
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
            if index == fractionStart {
                // A decimal point can be the last character of a valid
                // prefix, but it cannot be followed by an exponent or any
                // other delimiter until at least one fraction digit exists.
                return index == characters.count
            }
            if index == characters.count { return true }
        }

        if index < characters.count, characters[index] == "e" || characters[index] == "E" {
            index += 1
            if index < characters.count,
               characters[index] == "+" || characters[index] == "-" {
                index += 1
            }
            while index < characters.count, isASCIIJSONDigit(characters[index]) {
                index += 1
            }
        }

        return index == characters.count
    }
}

private struct MLXJSONStringPrefix {
    private enum Mode {
        case normal
        case escaped
        case unicode(remaining: Int)
    }

    private var mode: Mode = .normal
    private(set) var isClosed = false

    var allowsPlainText: Bool {
        if case .normal = mode, !isClosed { return true }
        return false
    }

    mutating func consume(_ character: Character) -> Bool {
        guard !isClosed else { return false }
        switch mode {
        case .normal:
            if character == "\"" {
                isClosed = true
                return true
            }
            if character == "\\" {
                mode = .escaped
                return true
            }
            guard !character.unicodeScalars.isEmpty,
                  character.unicodeScalars.allSatisfy({ $0.value >= 0x20 }) else {
                return false
            }
            return true

        case .escaped:
            if character == "u" {
                mode = .unicode(remaining: 4)
                return true
            }
            guard "\"\\/bfnrt".contains(character) else { return false }
            mode = .normal
            return true

        case .unicode(let remaining):
            guard isHex(character) else { return false }
            if remaining == 1 {
                mode = .normal
            } else {
                mode = .unicode(remaining: remaining - 1)
            }
            return true
        }
    }

    private func isHex(_ character: Character) -> Bool {
        isASCIIJSONDigit(character)
            || ("a"..."f").contains(character)
            || ("A"..."F").contains(character)
    }
}

private enum MLXToolGrammarFixedText {
    case jsonPrefix
    case textPrefix
    case toolPrefix
    case textSuffix
    case toolNameSuffix
    case moreToolPrefix

    var text: String {
        switch self {
        case .jsonPrefix:
            return "{\"response_type\":\""
        case .textPrefix:
            return "\",\"content\":\""
        case .toolPrefix:
            return "\",\"tool_calls\":[{\"name\":\""
        case .textSuffix:
            return "}"
        case .toolNameSuffix:
            return "\",\"arguments\":"
        case .moreToolPrefix:
            return "},{\"name\":\""
        }
    }
}

/// Grammar for:
///
/// - automatic choice: `text` or `tool_calls`
/// - required choice: `tool_calls` only
/// - parallel calls: one or more declared calls in the same envelope
/// - optional `<think>...</think>` / `Thinking Process: ... </think>` prefix
///
/// Tool arguments remain generic JSON and are schema-validated by the API
/// layer after decoding. This keeps the native constraint fast while still
/// guaranteeing that the model cannot select an undeclared function or emit
/// malformed JSON.
struct MLXToolResponseGrammar {
    private enum Phase {
        case initial(current: String)
        case reasoning(tail: String)
        case reasoningWhitespace
        case fixed(MLXToolGrammarFixedText, index: Int)
        case branch(current: String)
        case textString(MLXJSONStringPrefix)
        case textSuffix(index: Int)
        case toolName(current: String)
        case toolNameSuffix(index: Int)
        case arguments(MLXJSONPrefixParser)
        case afterArguments(current: String)
        case done
        case invalid
    }

    private enum ChoiceAdvance {
        case invalid
        case partial(String)
        case complete(String)
    }

    private static let reasoningMarker = Array("</think>")
    private static let reasoningOpen = "<think>"
    private static let ornithReasoningOpen = "Thinking Process:"

    private let configuration: MLXToolCallConstraintConfiguration
    private let escapedToolNames: [String]
    private var phase: Phase

    init(configuration: MLXToolCallConstraintConfiguration) {
        self.configuration = configuration
        self.escapedToolNames = configuration.toolNames.map(Self.jsonEscapedString).sorted()

        var options = [MLXToolGrammarFixedText.jsonPrefix.text]
        if configuration.allowReasoningPrefixes {
            options.append(Self.reasoningOpen)
            options.append(Self.ornithReasoningOpen)
        }
        self.phase = .initial(current: "")
        self.initialOptions = options
    }

    private let initialOptions: [String]

    var isComplete: Bool {
        if case .done = phase { return true }
        return false
    }

    var isValidPrefix: Bool {
        if case .invalid = phase { return false }
        return true
    }

    /// Used by the vocabulary index to avoid re-running the parser for the
    /// many ordinary text tokens that are always legal inside a JSON string.
    var allowsPlainJSONStringToken: Bool {
        switch phase {
        case .textString(let value):
            return value.allowsPlainText
        case .arguments(let parser):
            return parser.allowsPlainJSONStringToken
        default:
            return false
        }
    }

    var allowsArbitraryReasoningToken: Bool {
        if case .reasoning = phase { return true }
        return false
    }

    /// A first-character approximation for the current grammar state. Exact
    /// validation still happens for every candidate token, including tokens
    /// that cross multiple fixed-text boundaries.
    var allowedLeadingCharacters: Set<Character>? {
        switch phase {
        case .initial:
            return Set(initialOptions.compactMap(\.first))
        case .reasoning:
            return nil
        case .reasoningWhitespace:
            return Set([" ", "\n", "\r", "\t", "{"])
        case .fixed(let kind, let index):
            let expected = Array(kind.text)
            return index < expected.count ? [expected[index]] : []
        case .branch(let current):
            let options = configuration.decision == .required
                ? ["tool_calls"]
                : ["text", "tool_calls"]
            return Set(options.compactMap { option in
                guard current.count < option.count else { return nil }
                return option.character(at: current.count)
            })
        case .textString:
            return nil
        case .textSuffix:
            return ["}"]
        case .toolName(let current):
            var result: Set<Character> = Set(escapedToolNames.compactMap { name in
                guard name.hasPrefix(current), current.count < name.count else {
                    return nil
                }
                return name.character(at: current.count)
            })
            if escapedToolNames.contains(where: { $0 == current }) {
                // The closing quote is also a legal next character when a
                // shorter declared name is a prefix of another name.
                result.insert("\"")
            }
            return result
        case .toolNameSuffix:
            return []
        case .arguments(let parser):
            return parser.allowedLeadingCharacters
        case .afterArguments(let current):
            let options = configuration.allowParallelCalls
                ? ["}]}", "},{\"name\":\""]
                : ["}]}" ]
            return Set(options.compactMap { option in
                guard current.count < option.count else { return nil }
                return option.character(at: current.count)
            })
        case .done, .invalid:
            return []
        }
    }

    func canAccept(_ text: String) -> Bool {
        var copy = self
        return copy.acceptInPlace(text)
    }

    @discardableResult
    mutating func accept(_ text: String) -> Bool {
        let old = self
        guard acceptInPlace(text) else {
            self = old
            return false
        }
        return true
    }

    private mutating func acceptInPlace(_ text: String) -> Bool {
        for character in text {
            guard consume(character) else {
                phase = .invalid
                return false
            }
        }
        return true
    }

    private mutating func consume(_ character: Character) -> Bool {
        switch phase {
        case .initial(let current):
            switch advanceChoice(
                current: current,
                character: character,
                options: initialOptions
            ) {
            case .invalid:
                return false
            case .partial(let next):
                phase = .initial(current: next)
            case .complete(let option):
                if option == MLXToolGrammarFixedText.jsonPrefix.text {
                    phase = .branch(current: "")
                } else {
                    phase = .reasoning(tail: "")
                }
            }
            return true

        case .reasoning(let tail):
            let combined = Array(tail) + [character]
            if combined.count >= Self.reasoningMarker.count,
               Array(combined.suffix(Self.reasoningMarker.count)) == Self.reasoningMarker {
                phase = .reasoningWhitespace
                return true
            }
            let keepCount = min(
                combined.count,
                max(0, Self.reasoningMarker.count - 1)
            )
            phase = .reasoning(
                tail: keepCount == 0
                    ? ""
                    : String(combined.suffix(keepCount))
            )
            return true

        case .reasoningWhitespace:
            if isJSONWhitespace(character) {
                return true
            }
            phase = .fixed(.jsonPrefix, index: 0)
            return consume(character)

        case .fixed(let kind, let index):
            let expected = Array(kind.text)
            guard index < expected.count, expected[index] == character else {
                return false
            }
            let nextIndex = index + 1
            if nextIndex < expected.count {
                phase = .fixed(kind, index: nextIndex)
                return true
            }
            switch kind {
            case .jsonPrefix:
                phase = .branch(current: "")
            case .textPrefix:
                phase = .textString(MLXJSONStringPrefix())
            case .toolPrefix, .moreToolPrefix:
                phase = .toolName(current: "")
            case .textSuffix:
                phase = .done
            case .toolNameSuffix:
                phase = .arguments(MLXJSONPrefixParser(rootMustBeObject: true))
            }
            return true

        case .branch(let current):
            let options = configuration.decision == .required
                ? ["tool_calls"]
                : ["text", "tool_calls"]
            switch advanceChoice(
                current: current,
                character: character,
                options: options
            ) {
            case .invalid:
                return false
            case .partial(let next):
                phase = .branch(current: next)
            case .complete(let option):
                if option == "text" {
                    phase = .fixed(.textPrefix, index: 0)
                } else {
                    phase = .fixed(.toolPrefix, index: 0)
                }
            }
            return true

        case .textString(var value):
            guard value.consume(character) else { return false }
            if value.isClosed {
                phase = .textSuffix(index: 0)
            } else {
                phase = .textString(value)
            }
            return true

        case .textSuffix(let index):
            let expected: [Character] = ["}"]
            guard index < expected.count, expected[index] == character else {
                return false
            }
            phase = .done
            return true

        case .toolName(let current):
            guard !escapedToolNames.isEmpty else { return false }
            let next = current + String(character)
            if escapedToolNames.contains(where: { $0.hasPrefix(next) }) {
                let exact = escapedToolNames.first(where: { $0 == next })
                let hasLongerChoice = escapedToolNames.contains {
                    $0.count > next.count && $0.hasPrefix(next)
                }
                if exact != nil, !hasLongerChoice {
                    phase = .fixed(.toolNameSuffix, index: 0)
                } else {
                    phase = .toolName(current: next)
                }
                return true
            }

            // If one declared name is a prefix of another, the closing quote
            // disambiguates the shorter name. Re-process this character in
            // the fixed suffix state so both `read` and `read_file` remain
            // strictly constrained.
            if escapedToolNames.contains(where: { $0 == current }) {
                phase = .fixed(.toolNameSuffix, index: 0)
                return consume(character)
            }
            return false

        case .toolNameSuffix:
            return false

        case .arguments(var parser):
            guard let result = parser.consume(character) else { return false }
            switch result {
            case .accepted:
                phase = .arguments(parser)
            case .completed:
                phase = .afterArguments(current: "")
            }
            return true

        case .afterArguments(let current):
            let options = configuration.allowParallelCalls
                ? ["}]}", "},{\"name\":\""]
                : ["}]}" ]
            switch advanceChoice(
                current: current,
                character: character,
                options: options
            ) {
            case .invalid:
                return false
            case .partial(let next):
                phase = .afterArguments(current: next)
            case .complete(let option):
                if option == "}]}" {
                    phase = .done
                } else {
                    phase = .toolName(current: "")
                }
            }
            return true

        case .done, .invalid:
            return false
        }
    }

    private func advanceChoice(
        current: String,
        character: Character,
        options: [String]
    ) -> ChoiceAdvance {
        let next = current + String(character)
        guard options.contains(where: { $0.hasPrefix(next) }) else {
            return .invalid
        }

        let exact = options.first(where: { $0 == next })
        let hasLongerChoice = options.contains {
            $0.count > next.count && $0.hasPrefix(next)
        }
        if let exact, !hasLongerChoice {
            return .complete(exact)
        } else {
            return .partial(next)
        }
    }

    private static func jsonEscapedString(_ value: String) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: [value],
            options: []
        ), let encoded = String(data: data, encoding: .utf8) else {
            return value
        }
        // The encoded value is a one-element JSON array. Remove the array
        // brackets and retain only the escaped string contents; the grammar
        // already supplies the surrounding quotes.
        let prefix = encoded.dropFirst(2)
        return String(prefix.dropLast(2))
    }
}

private extension String {
    func character(at offset: Int) -> Character {
        self[index(startIndex, offsetBy: offset)]
    }
}

private func isJSONWhitespace(_ character: Character) -> Bool {
    character == " " || character == "\n" || character == "\r" || character == "\t"
}

private func isJSONDelimiter(_ character: Character) -> Bool {
    isJSONWhitespace(character) || character == "," || character == "}" || character == "]"
}

private func isASCIIJSONDigit(_ character: Character) -> Bool {
    ("0"..."9").contains(character)
}

// MARK: - Vocabulary masking

private struct MLXToolTokenCandidate {
    let id: Int
    let text: String
    let isPlainJSONStringText: Bool
}

private struct MLXToolTokenVocabulary {
    let candidates: [MLXToolTokenCandidate]
    let plainJSONStringTokenIDs: [Int]
    let nonPlainJSONStringCandidates: [MLXToolTokenCandidate]
    let candidatesByLeadingCharacter: [Character: [MLXToolTokenCandidate]]

    init(tokenizer: any Tokenizer, vocabularySize: Int, excludedIDs: Set<Int>) {
        var candidates: [MLXToolTokenCandidate] = []
        candidates.reserveCapacity(vocabularySize)
        var plainIDs: [Int] = []
        plainIDs.reserveCapacity(vocabularySize / 2)
        var nonPlainCandidates: [MLXToolTokenCandidate] = []
        var byLeadingCharacter: [Character: [MLXToolTokenCandidate]] = [:]

        for id in 0..<vocabularySize {
            guard !excludedIDs.contains(id),
                  let text = tokenizer.decode(
                    tokenIds: [id],
                    skipSpecialTokens: false
                  ).nilIfEmpty else {
                continue
            }

            let isPlain = text.allSatisfy { character in
                return character != "\""
                    && character != "\\"
                    && !character.unicodeScalars.isEmpty
                    && character.unicodeScalars.allSatisfy({ $0.value >= 0x20 })
            }
            let candidate = MLXToolTokenCandidate(
                id: id,
                text: text,
                isPlainJSONStringText: isPlain
            )
            candidates.append(candidate)
            if let first = text.first {
                byLeadingCharacter[first, default: []].append(candidate)
            }
            if isPlain {
                plainIDs.append(id)
            } else {
                nonPlainCandidates.append(candidate)
            }
        }

        self.candidates = candidates
        self.plainJSONStringTokenIDs = plainIDs
        self.nonPlainJSONStringCandidates = nonPlainCandidates
        self.candidatesByLeadingCharacter = byLeadingCharacter
    }

    func allowedIDs(
        for grammar: MLXToolResponseGrammar,
        vocabularySize: Int
    ) -> [Int] {
        let candidatePool: [MLXToolTokenCandidate]
        if let leadingCharacters = grammar.allowedLeadingCharacters {
            candidatePool = leadingCharacters.flatMap {
                candidatesByLeadingCharacter[$0] ?? []
            }
        } else {
            candidatePool = candidates
        }

        if grammar.allowsArbitraryReasoningToken {
            // Reasoning is outside the constrained JSON envelope. Only tokens
            // which actually contain the closing marker need parser work. All
            // other non-special tokens remain legal here. Testing the marker
            // boundary matters for byte-pair tokens such as `</think>{"`.
            return candidates.compactMap { candidate in
                candidate.text.contains("</think>")
                    ? (grammar.canAccept(candidate.text) ? candidate.id : nil)
                    : candidate.id
            }
        }

        if grammar.allowsPlainJSONStringToken {
            var ids = plainJSONStringTokenIDs
            for candidate in nonPlainJSONStringCandidates {
                if grammar.canAccept(candidate.text) {
                    ids.append(candidate.id)
                }
            }
            return ids
        }

        var ids: [Int] = []
        ids.reserveCapacity(min(candidates.count, 64))
        for candidate in candidatePool where grammar.canAccept(candidate.text) {
            ids.append(candidate.id)
        }
        // `vocabularySize` is kept in the signature to make it explicit that
        // the mask must always match the model's current logits dimension.
        _ = vocabularySize
        return ids
    }
}

/// Vocabulary decoding is the expensive part of a constrained request. Keep
/// one model's index alive across API calls; the next model selection replaces
/// it, so this does not accumulate one full tokenizer vocabulary per model.
private final class MLXToolVocabularyCache: @unchecked Sendable {
    static let shared = MLXToolVocabularyCache()

    private let lock = NSLock()
    private var key = ""
    private var value: MLXToolTokenVocabulary?

    func vocabulary(
        modelKey: String,
        tokenizer: any Tokenizer,
        vocabularySize: Int,
        excludedIDs: Set<Int>
    ) -> MLXToolTokenVocabulary {
        let cacheKey = "\(modelKey)|\(vocabularySize)|\(excludedIDs.sorted())"
        lock.lock()
        defer { lock.unlock() }
        if key == cacheKey, let value {
            return value
        }

        let value = MLXToolTokenVocabulary(
            tokenizer: tokenizer,
            vocabularySize: vocabularySize,
            excludedIDs: excludedIDs
        )
        key = cacheKey
        self.value = value
        return value
    }
}

/// Stateful sampler wrapper. MLX's `LogitSampler` is the most compatible
/// insertion point here: it masks the logits before the existing temperature /
/// top-p / top-k sampler and records the sampled token for the next grammar
/// state. The enclosing iterator still applies the app's normal penalty
/// processor and KV-cache quantization.
private final class MLXToolConstrainedSampler: LogitSampler, @unchecked Sendable {
    private let base: any LogitSampler
    private let tokenizer: any Tokenizer
    private let modelKey: String
    private let configuration: MLXToolCallConstraintConfiguration
    private let excludedIDs: Set<Int>
    private let stopTokenIDs: Set<Int>
    private var vocabulary: MLXToolTokenVocabulary?
    private var grammar: MLXToolResponseGrammar
    private var relaxed = false

    init(
        base: any LogitSampler,
        tokenizer: any Tokenizer,
        modelKey: String,
        configuration: MLXToolCallConstraintConfiguration,
        stopTokenIDs: Set<Int>
    ) {
        self.base = base
        self.tokenizer = tokenizer
        self.modelKey = modelKey
        self.configuration = configuration
        self.excludedIDs = stopTokenIDs
            .union(tokenizer.unknownTokenId.map { [$0] } ?? [])
        self.stopTokenIDs = stopTokenIDs
        self.grammar = MLXToolResponseGrammar(configuration: configuration)
    }

    func sample(logits: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        if vocabulary == nil {
            vocabulary = MLXToolVocabularyCache.shared.vocabulary(
                modelKey: modelKey,
                tokenizer: tokenizer,
                vocabularySize: vocabularySize,
                excludedIDs: excludedIDs
            )
        }

        var maskedLogits = logits
        if grammar.isComplete {
            let stopIDs = stopTokenIDs.filter { $0 >= 0 && $0 < vocabularySize }
            if !stopIDs.isEmpty {
                maskedLogits = mask(logits, allowedIDs: Array(stopIDs))
            }
        } else if !relaxed, let vocabulary {
            let allowed = vocabulary.allowedIDs(
                for: grammar,
                vocabularySize: vocabularySize
            )
            if allowed.isEmpty {
                // Tokenizers occasionally contain a model-specific special
                // token whose decoded form cannot be inspected in isolation.
                // Do not deadlock generation in that case. Relax only this
                // request and let the API validator/repair path reject any
                // malformed result; ordinary requests remain unaffected.
                relaxed = true
            } else {
                maskedLogits = mask(logits, allowedIDs: allowed)
            }
        }

        let token = base.sample(logits: maskedLogits)
        let tokenID = token.item(Int.self)
        if !relaxed, let tokenText = tokenizer.decode(
            tokenIds: [tokenID],
            skipSpecialTokens: false
        ).nilIfEmpty {
            _ = grammar.accept(tokenText)
        }
        return token
    }

    private func mask(_ logits: MLXArray, allowedIDs: [Int]) -> MLXArray {
        var allowed = [Bool](repeating: false, count: logits.dim(-1))
        for id in allowedIDs where allowed.indices.contains(id) {
            allowed[id] = true
        }
        return MLX.where(
            MLXArray(allowed),
            logits,
            MLXArray(-Float.infinity)
        )
    }
}

// MARK: - KV-cache-preserving iterator

/// `TokenIterator`'s public custom-processor initializer disables dynamic KV
/// quantization. This small app-owned equivalent keeps the package's normal
/// parameters while swapping only the sampler, so constrained tool decisions
/// do not create a hidden memory regression on 9B/27B models.
struct MLXToolConstrainedTokenIterator: TokenIteratorProtocol {
    private let model: any LanguageModel
    private var y: LMInput.Text
    private var cache: [KVCache]
    private var state: LMOutput.State?
    private var processor: LogitProcessor?
    private let sampler: any LogitSampler
    let maxTokens: Int?
    private let kvBits: Int?
    private let kvGroupSize: Int
    private let quantizedKVStart: Int
    private let kvScheme: String?
    private(set) var tokenCount = 0
    private(set) var promptPrefillTime: TimeInterval

    init(
        input: LMInput,
        model: any LanguageModel,
        cache: [KVCache],
        parameters: GenerateParameters,
        sampler: any LogitSampler
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache
        self.processor = parameters.processor()
        self.sampler = sampler
        self.maxTokens = parameters.maxTokens
        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.kvScheme = parameters.kvScheme
        self.promptPrefillTime = 0

        let start = Date.timeIntervalSinceReferenceDate
        try self.prepare(input: input, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
    }

    private mutating func prepare(input: LMInput, windowSize: Int?) throws {
        processor?.prompt(input.text.tokens)
        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)
        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)
        }
    }

    private mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        return token
    }

    private mutating func step(previous: LMInput.Text) -> MLXArray {
        let result = withPreparedCache(cache, lengths: previous.sequenceLengths) {
            model(
                previous[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: state
            )
        }
        state = result.state
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            kvScheme: kvScheme
        )
        return convertToToken(logits: result.logits)
    }

    mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        let previousY = y
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)
        tokenCount += 1
        return previousY.tokens.item(Int.self)
    }
}

extension String {
    fileprivate var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

/// Internal entry point used by `CodingAssistantService` so the regular MLX
/// generation path remains unchanged for ordinary chat.
enum MLXToolCallConstraintRuntime {
    static func makeIterator(
        input: LMInput,
        context: ModelContext,
        cache: [KVCache],
        parameters: GenerateParameters,
        configuration: MLXToolCallConstraintConfiguration,
        modelKey: String
    ) throws -> MLXToolConstrainedTokenIterator {
        let stopTokenIDs = context.configuration.eosTokenIds
            .union(context.configuration.extraEOSTokens.compactMap {
                context.tokenizer.convertTokenToId($0)
            })
            .union(context.tokenizer.eosTokenId.map { [$0] } ?? [])
        let sampler = MLXToolConstrainedSampler(
            base: parameters.sampler(),
            tokenizer: context.tokenizer,
            modelKey: modelKey,
            configuration: configuration,
            stopTokenIDs: stopTokenIDs
        )
        return try MLXToolConstrainedTokenIterator(
            input: input,
            model: context.model,
            cache: cache,
            parameters: parameters,
            sampler: sampler
        )
    }
}
