import XCTest
@testable import OnDeviceLAS

final class LocalAPIServerTests: XCTestCase {
    func testPortValidation() {
        XCTAssertNil(LocalAPIValidation.validPort(1023))
        XCTAssertEqual(LocalAPIValidation.validPort(11434), 11434)
        XCTAssertEqual(LocalAPIValidation.validPort(65535), 65535)
        XCTAssertNil(LocalAPIValidation.validPort(65536))
    }

    func testModelAliases() {
        XCTAssertTrue(LocalAPIValidation.modelMatches("qwen", id: "qwen", repoID: "org/qwen"))
        XCTAssertTrue(LocalAPIValidation.modelMatches("org/qwen", id: "qwen", repoID: "org/qwen"))
        XCTAssertFalse(LocalAPIValidation.modelMatches("other", id: "qwen", repoID: "org/qwen"))
    }

    func testAuthorizationAcceptsBearerCaseVariantsAndAPIKeyHeader() {
        XCTAssertTrue(
            LocalAPIValidation.isAuthorized(
                headers: ["authorization": "bearer secret"],
                key: "secret"
            )
        )
        XCTAssertTrue(
            LocalAPIValidation.isAuthorized(
                headers: ["x-api-key": "secret"],
                key: "secret"
            )
        )
        XCTAssertTrue(
            LocalAPIValidation.isAuthorized(
                headers: ["Authorization": "BeArEr secret"],
                key: "secret"
            )
        )
        XCTAssertTrue(
            LocalAPIValidation.isAuthorized(
                headers: ["X-API-Key": "secret"],
                key: "secret"
            )
        )
        XCTAssertFalse(
            LocalAPIValidation.isAuthorized(
                headers: ["authorization": "Bearer secret-extra"],
                key: "secret"
            )
        )
    }

    func testTunnelInterfacesAreExcludedFromLANAddresses() {
        XCTAssertTrue(LocalAPIValidation.isReachableLANInterface("en0"))
        XCTAssertTrue(LocalAPIValidation.isReachableLANInterface("bridge100"))
        XCTAssertFalse(LocalAPIValidation.isReachableLANInterface("utun4"))
        XCTAssertFalse(LocalAPIValidation.isReachableLANInterface("ipsec0"))
        XCTAssertFalse(LocalAPIValidation.isReachableLANInterface("pdp_ip0"))
    }

    func testRemoteInferenceLimitsBoundToolSelection() {
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 65_536,
                toolCallingEnabled: true
            ),
            256
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 128,
                toolCallingEnabled: true
            ),
            128
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: nil,
                toolCallingEnabled: true
            ),
            256
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 65_536,
                toolCallingEnabled: false
            ),
            4_096
        )
        XCTAssertNil(
            LocalAPIInferencePolicy.maxTokens(
                requested: nil,
                toolCallingEnabled: false
            )
        )
    }

    func testRemoteInferenceDeadlinesCoverToolsAndPlainResponses() {
        XCTAssertEqual(
            LocalAPIInferencePolicy.deadline(toolCallingEnabled: true),
            .seconds(30)
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.deadline(toolCallingEnabled: false),
            .seconds(90)
        )
    }

    func testToolDecisionBuffersOnlyPotentialToolSyntax() {
        XCTAssertTrue(LocalAPIToolCalling.shouldBufferForToolDecision(""))
        XCTAssertTrue(LocalAPIToolCalling.shouldBufferForToolDecision("  {"))
        XCTAssertTrue(LocalAPIToolCalling.shouldBufferForToolDecision("<tool"))
        XCTAssertTrue(LocalAPIToolCalling.shouldBufferForToolDecision("```json\n{"))
        XCTAssertFalse(
            LocalAPIToolCalling.shouldBufferForToolDecision(
                "The weather is sunny."
            )
        )
        XCTAssertFalse(
            LocalAPIToolCalling.shouldBufferForToolDecision(
                #"{"answer":"This is ordinary JSON output, not a tool call, and it is long enough to make that decision without buffering the whole generation."}"#
            )
        )
        XCTAssertTrue(
            LocalAPIToolCalling.shouldBufferForToolDecision(
                #"{"tool_calls":[{"name":"get_weather","arguments":{"city":"Istanbul","units":"metric","include_forecast":true}}]}"#
            )
        )
    }

    func testOpenAIRequestDecodesTextMessagesAndOverrides() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[
            {"role":"system","content":"Be concise."},
            {"role":"user","content":"Hello"}
          ],
          "stream":true,
          "max_tokens":128,
          "temperature":0.2,
          "top_p":0.8
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(request.model, "qwen")
        XCTAssertEqual(request.messages.count, 2)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.maxTokens, 128)
        XCTAssertEqual(request.temperature, 0.2)
        XCTAssertEqual(request.topP, 0.8)
    }

    func testOpenAIAcceptsEmptyTools() throws {
        let data = Data("""
        {"model":"qwen","messages":[{"role":"user","content":"Hi"}],"tools":[]}
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(request.messages.map(\.content), ["Hi"])
    }

    func testOpenAIDecodesFunctionTools() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[{"role":"user","content":"What is the weather in Cairo?"}],
          "tools":[{
            "type":"function",
            "function":{
              "name":"get_weather",
              "description":"Get weather",
              "parameters":{
                "type":"object",
                "properties":{"city":{"type":"string"}},
                "required":["city"]
              }
            }
          }],
          "tool_choice":"required",
          "parallel_tool_calls":false
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(request.tools.map(\.name), ["get_weather"])
        XCTAssertEqual(request.tools.first?.description, "Get weather")
        XCTAssertEqual(request.toolChoice, .required)
        XCTAssertFalse(request.parallelToolCalls)
    }

    func testOpenAIAcceptsHermesAgentCustomProviderOptions() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[
            {"role":"system","content":"You are Hermes Agent."},
            {"role":"user","content":"Inspect the workspace."}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"terminal",
              "description":"Run a shell command",
              "parameters":{
                "type":"object",
                "properties":{"command":{"type":"string"}},
                "required":["command"]
              }
            }
          }],
          "stream":true,
          "stream_options":{"include_usage":true},
          "max_tokens":65536,
          "reasoning_effort":"none",
          "think":false,
          "options":{"num_ctx":64000}
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.maxTokens, 65_536)
        XCTAssertEqual(request.tools.map(\.name), ["terminal"])
        XCTAssertEqual(request.toolChoice, .auto)

        let inferenceMessages = LocalAPIToolCalling.messages(
            from: request.messages,
            tools: request.tools,
            choice: request.toolChoice,
            parallelToolCalls: request.parallelToolCalls
        )
        XCTAssertFalse(inferenceMessages.contains { $0.role == .system })
        XCTAssertTrue(inferenceMessages.last?.content.contains("You are Hermes Agent.") == true)
        XCTAssertTrue(inferenceMessages.last?.content.contains("Inspect the workspace.") == true)
    }

    func testOpenAIDecodesHermesTextPartArraysAndToolResult() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[
            {"role":"system","content":"You are in agent mode."},
            {"role":"user","content":[{"type":"text","text":"show me the codebase"}]},
            {
              "role":"assistant",
              "content":[{"type":"text","text":"I will inspect the root."}],
              "tool_calls":[{
                "id":"tool_1",
                "type":"function",
                "function":{"name":"terminal","arguments":"{\\"command\\":\\"find . -maxdepth 2\\"}"}
              }]
            },
            {
              "role":"tool",
              "tool_call_id":"tool_1",
              "content":[{"type":"text","text":"README.md\\nsrc/main.py"}]
            },
            {"role":"user","content":[{"type":"input_text","text":"continue"}]}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"terminal",
              "description":"Run a command",
              "parameters":{"type":"object"}
            }
          }],
          "stream":true,
          "messagesOptions":{"precompleted":true}
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(
            request.messages.map(\.role),
            [.system, .user, .assistant, .user, .user]
        )
        XCTAssertEqual(request.messages[1].content, "show me the codebase")
        XCTAssertTrue(request.messages[2].content.contains("I will inspect the root."))
        XCTAssertTrue(request.messages[2].content.contains("terminal"))
        XCTAssertTrue(request.messages[3].content.contains("README.md\nsrc/main.py"))
        XCTAssertEqual(request.messages[4].content, "continue")
    }

    func testOpenAIRejectsUnknownNamedToolChoice() {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[{"role":"user","content":"Hi"}],
          "tools":[{
            "type":"function",
            "function":{"name":"known","parameters":{"type":"object"}}
          }],
          "tool_choice":{"type":"function","function":{"name":"unknown"}}
        }
        """.utf8)
        XCTAssertThrowsError(try LocalAPIChatRequest.decodeOpenAI(data))
    }

    func testOpenAIDecodesToolCallHistoryAndResult() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[
            {"role":"user","content":"Weather?"},
            {
              "role":"assistant",
              "content":null,
              "tool_calls":[{
                "id":"call_1",
                "type":"function",
                "function":{"name":"get_weather","arguments":"{\\"city\\":\\"Cairo\\"}"}
              }]
            },
            {"role":"tool","tool_call_id":"call_1","content":"31 C, sunny"}
          ]
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(request.messages.map(\.role), [.user, .assistant, .user])
        XCTAssertTrue(request.messages[1].content.contains("get_weather"))
        XCTAssertTrue(request.messages[2].content.contains("31 C, sunny"))
    }

    func testOpenAIPreservesAssistantTextAlongsideToolCallHistory() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[{
            "role":"assistant",
            "content":"I will inspect that.",
            "tool_calls":[{
              "id":"call_1",
              "type":"function",
              "function":{"name":"terminal","arguments":"{\\"command\\":\\"pwd\\"}"}
            }]
          }]
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertTrue(request.messages[0].content.contains("I will inspect that."))
        XCTAssertTrue(request.messages[0].content.contains("terminal"))
        XCTAssertTrue(request.messages[0].content.contains("call_1"))
    }

    func testToolCallingPromptAndParser() {
        let tool = LocalAPIToolDefinition(
            name: "get_weather",
            description: "Get weather",
            parametersJSON: #"{"properties":{"city":{"type":"string"}},"type":"object"}"#
        )
        let messages = LocalAPIToolCalling.messages(
            from: [ChatMessage(role: .user, content: "Weather in Cairo?")],
            tools: [tool],
            choice: .auto,
            parallelToolCalls: true
        )
        XCTAssertEqual(messages.first?.role, .user)
        XCTAssertTrue(messages.first?.content.contains("get_weather") == true)
        XCTAssertTrue(messages.first?.content.contains("Weather in Cairo?") == true)

        let calls = LocalAPIToolCalling.parse(
            """
            <tool_call>
            {"tool_calls":[{"name":"get_weather","arguments":{"city":"Cairo"}}]}
            </tool_call>
            """,
            tools: [tool],
            parallelToolCalls: true
        )
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"city":"Cairo"}"#)
    }

    func testToolCallingParserCollectsParallelHermesCalls() {
        let tools = [
            LocalAPIToolDefinition(
                name: "read_file",
                description: nil,
                parametersJSON: #"{"type":"object"}"#
            ),
            LocalAPIToolDefinition(
                name: "terminal",
                description: nil,
                parametersJSON: #"{"type":"object"}"#
            )
        ]
        let calls = LocalAPIToolCalling.parse(
            """
            <tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>
            <tool_call>{"name":"terminal","parameters":{"command":"pwd"}}</tool_call>
            """,
            tools: tools,
            parallelToolCalls: true
        )
        XCTAssertEqual(calls.map(\.name), ["read_file", "terminal"])
        XCTAssertEqual(calls[1].argumentsJSON, #"{"command":"pwd"}"#)
    }

    func testToolCallingParserAcceptsAppStyleArgs() {
        let tool = LocalAPIToolDefinition(
            name: "get_weather",
            description: nil,
            parametersJSON: #"{"type":"object"}"#
        )
        let calls = LocalAPIToolCalling.parse(
            #"{"name":"get_weather","args":{"city":"Asyut"}}"#,
            tools: [tool],
            parallelToolCalls: false
        )
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"city":"Asyut"}"#)
    }

    func testToolCallingParserPreservesIDAndValidatesArguments() {
        let tool = LocalAPIToolDefinition(
            name: "terminal",
            description: "Run a command",
            parametersJSON: #"{"additionalProperties":false,"properties":{"command":{"type":"string"}},"required":["command"],"type":"object"}"#
        )
        let valid = LocalAPIToolCalling.parse(
            #"{"tool_calls":[{"id":"call_123","name":"terminal","arguments":{"command":"pwd"}}]}"#,
            tools: [tool],
            parallelToolCalls: true
        )
        XCTAssertEqual(valid.map(\.id), ["call_123"])
        XCTAssertEqual(valid.first?.argumentsJSON, #"{"command":"pwd"}"#)

        let invalid = LocalAPIToolCalling.parse(
            #"{"tool_calls":[{"id":"call_bad","name":"terminal","arguments":{"command":12}}]}"#,
            tools: [tool],
            parallelToolCalls: true
        )
        XCTAssertTrue(invalid.isEmpty)
    }

    func testToolCallingParserIgnoresOrnithReasoningBeforeToolCall() {
        let tool = LocalAPIToolDefinition(
            name: "terminal",
            description: "Run a command",
            parametersJSON: #"{"properties":{"command":{"type":"string"}},"required":["command"],"type":"object"}"#
        )
        let calls = LocalAPIToolCalling.parse(
            "Thinking Process:\nprivate reasoning\n</think>\n<tool_call>{\"name\":\"terminal\",\"arguments\":{\"command\":\"pwd\"}}</tool_call>",
            tools: [tool],
            parallelToolCalls: false
        )
        XCTAssertEqual(calls.map(\.name), ["terminal"])
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"command":"pwd"}"#)
    }

    func testNativeMLXToolGrammarAcceptsRequiredToolEnvelope() {
        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: ["terminal"],
            decision: .required,
            allowParallelCalls: false,
            allowReasoningPrefixes: false
        )
        var grammar = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertTrue(
            grammar.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"terminal","arguments":{"command":"pwd"}}]}"#
            )
        )
        XCTAssertTrue(grammar.isComplete)
        XCTAssertFalse(grammar.accept("x"))
    }

    func testNativeMLXToolGrammarSupportsAutomaticTextAndOrnithReasoning() {
        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: ["terminal"],
            decision: .automatic,
            allowParallelCalls: false,
            allowReasoningPrefixes: true
        )
        var grammar = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertTrue(
            grammar.accept(
                #"Thinking Process: answer without a tool </think> {"response_type":"text","content":"The path is local."}"#
            )
        )
        XCTAssertTrue(grammar.isComplete)
        XCTAssertEqual(
            LocalAPIToolCalling.textResponse(
                from: #"Thinking Process: done </think> {"response_type":"text","content":"The path is local."}"#
            ),
            "The path is local."
        )
    }

    func testNativeMLXToolGrammarRejectsUndeclaredToolAndMalformedArguments() {
        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: ["terminal"],
            decision: .required,
            allowParallelCalls: false,
            allowReasoningPrefixes: false
        )
        var undeclared = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertFalse(
            undeclared.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"browser","arguments":{}}]}"#
            )
        )

        var malformed = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertFalse(
            malformed.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"terminal","arguments":{"command":}}]}"#
            )
        )
    }

    func testNativeMLXToolGrammarSupportsParallelCalls() {
        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: ["terminal", "file_read"],
            decision: .required,
            allowParallelCalls: true,
            allowReasoningPrefixes: false
        )
        var grammar = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertTrue(
            grammar.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"terminal","arguments":{}},{"name":"file_read","arguments":{"path":"/tmp/a"}}]}"#
            )
        )
        XCTAssertTrue(grammar.isComplete)
    }

    func testNativeMLXToolGrammarDisambiguatesSharedToolNamePrefixes() {
        let configuration = MLXToolCallConstraintConfiguration(
            toolNames: ["read", "read_file"],
            decision: .required,
            allowParallelCalls: false,
            allowReasoningPrefixes: false
        )
        var shorterName = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertTrue(
            shorterName.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"read","arguments":{}}]}"#
            )
        )

        var invalidNumber = MLXToolResponseGrammar(configuration: configuration)
        XCTAssertFalse(
            invalidNumber.accept(
                #"{"response_type":"tool_calls","tool_calls":[{"name":"read_file","arguments":{"value":1.e}}]}"#
            )
        )
    }

    func testReasoningFilterRemovesSplitThinkTags() {
        var filter = LocalAPIReasoningFilter()
        let deltas = ["<thi", "nk>private", " reasoning</thi", "nk>Final answer"]
            .map { filter.consume($0) }
        let trailing = filter.finish()
        let visible = deltas.map(\.content).joined() + trailing.content
        let reasoning = deltas.map(\.reasoning).joined() + trailing.reasoning
        XCTAssertEqual(visible, "Final answer")
        XCTAssertEqual(
            reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            "private reasoning"
        )
    }

    func testReasoningParserSeparatesOrnithThinkingProcess() {
        let parsed = LocalAPIReasoningFilter.parse(
            "Thinking Process:\nprivate reasoning\n</think>\nThe answer."
        )
        XCTAssertEqual(parsed.reasoning, "private reasoning")
        XCTAssertEqual(parsed.content, "The answer.")
    }

    func testReasoningParserStreamsOrnithThinkingProcessAcrossChunks() {
        var filter = LocalAPIReasoningFilter()
        let deltas = [
            "Thinking Pro",
            "cess:\nprivate reasoning</thi",
            "nk>The answer."
        ].map { filter.consume($0) }
        let trailing = filter.finish()

        XCTAssertEqual(deltas.map(\.content).joined() + trailing.content, "The answer.")
        XCTAssertEqual(filter.reasoningText, "private reasoning")
    }

    func testReasoningFilterStreamsQwenPrefilledClosingTagOnly() {
        var filter = LocalAPIReasoningFilter(prefilledOpening: true)
        let chunks = [
            "Let me analyze the request carefully. ",
            "The user wants STREAM_OK.</thi",
            "nk>\nSTREAM_OK"
        ]
        var reasoningParts: [String] = []
        var contentParts: [String] = []
        for chunk in chunks {
            let delta = filter.consume(chunk)
            if !delta.reasoning.isEmpty { reasoningParts.append(delta.reasoning) }
            if !delta.content.isEmpty { contentParts.append(delta.content) }
        }
        let trailing = filter.finish()
        if !trailing.reasoning.isEmpty { reasoningParts.append(trailing.reasoning) }
        if !trailing.content.isEmpty { contentParts.append(trailing.content) }

        let reasoning = reasoningParts.joined()
        let content = contentParts.joined()
        XCTAssertTrue(reasoning.contains("Let me analyze"))
        XCTAssertFalse(reasoning.contains("<think>"))
        XCTAssertFalse(reasoning.contains("</think>"))
        XCTAssertEqual(content.trimmingCharacters(in: .whitespacesAndNewlines), "STREAM_OK")
        XCTAssertFalse(content.contains("think"))
        XCTAssertFalse(content.contains("Let me analyze"))
    }

    func testOpenAIResponsesDecodesTextInputBlocks() throws {
        let data = Data("""
        {
          "model":"qwen",
          "instructions":"Be concise.",
          "input":[{
            "role":"user",
            "content":[{"type":"input_text","text":"Hi"}]
          }],
          "tools":[],
          "max_output_tokens":64
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAIResponses(data)
        XCTAssertEqual(request.messages.map(\.role), [.system, .user])
        XCTAssertEqual(request.messages.map(\.content), ["Be concise.", "Hi"])
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertFalse(request.stream)
    }

    func testOpenAIResponsesAcceptsStringInput() throws {
        let data = Data("""
        {"model":"qwen","input":"Hi","stream":true}
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAIResponses(data)
        XCTAssertEqual(request.messages.map(\.content), ["Hi"])
        XCTAssertTrue(request.stream)
    }

    func testOpenAIResponsesDecodesFunctionToolHistory() throws {
        let data = Data("""
        {
          "model":"qwen",
          "input":[
            {"role":"user","content":"Run pwd"},
            {"type":"function_call","call_id":"call_123","name":"terminal","arguments":"{\\"command\\":\\"pwd\\"}"},
            {"type":"function_call_output","call_id":"call_123","output":"/tmp"}
          ],
          "tools":[{
            "type":"function",
            "name":"terminal",
            "description":"Run a command",
            "parameters":{"type":"object"}
          }],
          "tool_choice":{"type":"function","name":"terminal"},
          "thinking":{"type":"disabled"},
          "output_config":{"format":"text"}
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOpenAIResponses(data)
        XCTAssertEqual(request.tools.map(\.name), ["terminal"])
        XCTAssertEqual(request.toolChoice, .function("terminal"))
        XCTAssertTrue(request.messages[1].content.contains("call_123"))
        XCTAssertTrue(request.messages[2].content.contains("/tmp"))
    }

    func testOllamaChatDefaultsToStreaming() throws {
        let data = Data("""
        {"model":"qwen","messages":[{"role":"user","content":"Hi"}]}
        """.utf8)
        XCTAssertTrue(try LocalAPIChatRequest.decodeOllamaChat(data).stream)
    }

    func testOllamaChatCanUseCurrentModelWhenOmitted() throws {
        let data = Data("""
        {"messages":[{"role":"user","content":"Hi"}],"stream":false}
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOllamaChat(data)
        XCTAssertEqual(request.model, "")
        XCTAssertFalse(request.stream)
    }

    func testOllamaChatDecodesToolsAndObjectArgumentsHistory() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[
            {"role":"user","content":"Weather?"},
            {
              "role":"assistant",
              "content":"",
              "tool_calls":[{
                "function":{
                  "name":"get_weather",
                  "arguments":{"city":"Cairo"}
                }
              }]
            },
            {"role":"tool","content":"31 C, sunny"}
          ],
          "tools":[{
            "type":"function",
            "function":{
              "name":"get_weather",
              "description":"Get weather",
              "parameters":{"type":"object"}
            }
          }]
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOllamaChat(data)
        XCTAssertEqual(request.tools.map(\.name), ["get_weather"])
        XCTAssertEqual(request.toolChoice, .auto)
        XCTAssertTrue(request.messages[1].content.contains("get_weather"))
        XCTAssertTrue(request.messages[2].content.contains("31 C, sunny"))
    }

    func testOllamaGenerateBuildsSystemAndUserMessages() throws {
        let data = Data("""
        {"model":"qwen","system":"Be concise.","prompt":"Hi","stream":false}
        """.utf8)
        let request = try LocalAPIChatRequest.decodeOllamaGenerate(data)
        XCTAssertEqual(request.messages.map(\.role), [.system, .user])
        XCTAssertFalse(request.stream)
    }

    func testAnthropicRequestDecodesTextBlocksAndSystem() throws {
        let data = Data("""
        {
          "model":"qwen",
          "system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"Hello"}]}],
          "max_tokens":128,
          "stream":true
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeAnthropic(data)
        XCTAssertEqual(request.messages.map(\.role), [.system, .user])
        XCTAssertEqual(request.messages.map(\.content), ["Be concise.", "Hello"])
        XCTAssertEqual(request.maxTokens, 128)
        XCTAssertTrue(request.stream)
    }

    func testAnthropicDecodesToolsAndToolResultHistory() throws {
        let data = Data("""
        {
          "model":"qwen",
          "max_tokens":64,
          "messages":[
            {"role":"user","content":"Weather?"},
            {
              "role":"assistant",
              "content":[{
                "type":"tool_use",
                "id":"toolu_1",
                "name":"get_weather",
                "input":{"city":"Cairo"}
              }]
            },
            {
              "role":"user",
              "content":[{
                "type":"tool_result",
                "tool_use_id":"toolu_1",
                "content":"31 C, sunny"
              }]
            }
          ],
          "tools":[{
            "name":"get_weather",
            "description":"Get weather",
            "input_schema":{"type":"object"}
          }],
          "tool_choice":{"type":"auto"}
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeAnthropic(data)
        XCTAssertEqual(request.tools.map(\.name), ["get_weather"])
        XCTAssertEqual(request.toolChoice, .auto)
        XCTAssertTrue(request.messages[1].content.contains("toolu_1"))
        XCTAssertTrue(request.messages[2].content.contains("31 C, sunny"))
    }

    func testAnthropicIgnoresUnsupportedReasoningOptions() throws {
        let data = Data("""
        {
          "model":"qwen",
          "messages":[{"role":"user","content":"Hello"}],
          "max_tokens":64,
          "thinking":{"type":"disabled"},
          "output_config":{"format":"text"},
          "metadata":{"user_id":"hermes"},
          "service_tier":"auto"
        }
        """.utf8)
        let request = try LocalAPIChatRequest.decodeAnthropic(data)
        XCTAssertEqual(request.messages.last?.content, "Hello")
        XCTAssertTrue(request.parallelToolCalls)
    }

    func testAnthropicResponseHasCompatibilityShape() throws {
        let data = LocalAPIResponse.anthropicMessage(id: "msg_test", model: "qwen", text: "hello")
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["type"] as? String, "message")
        XCTAssertEqual(object["role"] as? String, "assistant")
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content.first?["text"] as? String, "hello")
    }

    func testOpenAIChunkHasCompatibilityShape() throws {
        let data = LocalAPIResponse.openAIChunk(
            id: "chatcmpl-test",
            model: "qwen",
            text: "hello",
            role: "assistant"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["object"] as? String, "chat.completion.chunk")
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let delta = try XCTUnwrap(choices.first?["delta"] as? [String: Any])
        XCTAssertEqual(delta["content"] as? String, "hello")
        XCTAssertEqual(delta["role"] as? String, "assistant")
    }

    func testOpenAIResponseSeparatesReasoningContent() throws {
        let data = LocalAPIResponse.openAIChatCompletion(
            id: "chatcmpl-test",
            model: "ornith-1.0-9b-4bit",
            text: "The answer.",
            toolCalls: [],
            reasoningContent: "private reasoning"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        XCTAssertEqual(message["content"] as? String, "The answer.")
        XCTAssertEqual(message["reasoning_content"] as? String, "private reasoning")
    }

    func testOpenAIToolCallResponseHasCompatibilityShape() throws {
        let data = LocalAPIResponse.openAIChatCompletion(
            id: "chatcmpl-test",
            model: "qwen",
            text: "",
            toolCalls: [
                LocalAPIToolCall(
                    id: "call_test",
                    name: "get_weather",
                    argumentsJSON: #"{"city":"Cairo"}"#
                )
            ]
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        XCTAssertEqual(choices.first?["finish_reason"] as? String, "tool_calls")
        let message = try XCTUnwrap(choices.first?["message"] as? [String: Any])
        XCTAssertTrue(message["content"] is NSNull)
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(function["arguments"] as? String, #"{"city":"Cairo"}"#)
    }

    func testAnthropicToolResponseHasCompatibilityShape() throws {
        let data = LocalAPIResponse.anthropicToolMessage(
            id: "msg_test",
            model: "qwen",
            calls: [
                LocalAPIToolCall(
                    id: "toolu_test",
                    name: "get_weather",
                    argumentsJSON: #"{"city":"Cairo"}"#
                )
            ]
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["stop_reason"] as? String, "tool_use")
        let content = try XCTUnwrap(object["content"] as? [[String: Any]])
        XCTAssertEqual(content.first?["type"] as? String, "tool_use")
        XCTAssertEqual(content.first?["id"] as? String, "toolu_test")
    }

    func testOllamaToolResponseHasCompatibilityShape() throws {
        let data = LocalAPIResponse.ollamaToolCalls(
            model: "qwen",
            calls: [
                LocalAPIToolCall(
                    id: "ignored",
                    name: "get_weather",
                    argumentsJSON: #"{"city":"Cairo"}"#
                )
            ],
            done: true
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let message = try XCTUnwrap(object["message"] as? [String: Any])
        let calls = try XCTUnwrap(message["tool_calls"] as? [[String: Any]])
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "get_weather")
        XCTAssertEqual(
            (function["arguments"] as? [String: Any])?["city"] as? String,
            "Cairo"
        )
    }

    func testOpenAIResponseHasCompatibilityShape() throws {
        let data = LocalAPIResponse.openAIResponse(
            id: "resp_test", model: "qwen", text: "hello"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        XCTAssertEqual(object["object"] as? String, "response")
        XCTAssertEqual(object["status"] as? String, "completed")
        XCTAssertEqual(object["output_text"] as? String, "hello")
        let output = try XCTUnwrap(object["output"] as? [[String: Any]])
        XCTAssertEqual(output.first?["role"] as? String, "assistant")
    }

    func testOpenAIResponseToolHasCompatibilityShape() throws {
        let data = LocalAPIResponse.openAIResponse(
            id: "resp_test",
            model: "qwen",
            text: "",
            toolCalls: [
                LocalAPIToolCall(
                    id: "call_123",
                    name: "terminal",
                    argumentsJSON: #"{"command":"pwd"}"#
                )
            ]
        )
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let output = try XCTUnwrap(object["output"] as? [[String: Any]])
        XCTAssertEqual(output.first?["type"] as? String, "function_call")
        XCTAssertEqual(output.first?["call_id"] as? String, "call_123")
        XCTAssertEqual(output.first?["name"] as? String, "terminal")
        XCTAssertEqual(output.first?["arguments"] as? String, #"{"command":"pwd"}"#)
        XCTAssertEqual(object["output_text"] as? String, "")
    }

    func testHTTPRequestParsesQueryAndBearer() {
        let request = HTTPRequest(data: Data("""
        POST /v1/chat/completions?trace=1 HTTP/1.1\r
        Authorization: Bearer abc\r
        Content-Type: application/json\r
        Content-Length: 2\r
        \r
        {}
        """.utf8))
        XCTAssertEqual(request?.path, "/v1/chat/completions")
        XCTAssertEqual(request?.headers["authorization"], "Bearer abc")
        XCTAssertEqual(request?.body, Data("{}".utf8))
    }

    func testHTTPRequestAcceptsOptionalHeaderWhitespace() {
        let request = HTTPRequest(data: Data("""
        post /v1/models HTTP/1.1\r
        Authorization:\tbearer abc\r
        Content-Type:application/json\r
        Content-Length: 0\r
        \r

        """.utf8))
        XCTAssertEqual(request?.method, "POST")
        XCTAssertEqual(request?.path, "/v1/models")
        XCTAssertEqual(request?.headers["authorization"], "bearer abc")
        XCTAssertNil(request?.body)
    }
}
