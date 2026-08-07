import XCTest
@testable import OnDeviceLAS

final class LocalAPIServerTests: XCTestCase {
    private let onePixelPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlK7nAAAAAASUVORK5CYII="

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

    func testRemoteInferenceRespectsRequestedAndRuntimeTokenLimits() {
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 128,
                runtimeMaximum: 4_096
            ),
            128
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 1_024,
                runtimeMaximum: 4_096
            ),
            1_024
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 4_096,
                runtimeMaximum: 4_096
            ),
            4_096
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: 65_536,
                runtimeMaximum: 4_096
            ),
            4_096
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.maxTokens(
                requested: nil,
                runtimeMaximum: 2_048
            ),
            2_048
        )
    }

    func testRemoteInferenceDeadlineScalesForLongCompletions() {
        XCTAssertEqual(
            LocalAPIInferencePolicy.deadline(
                maxTokens: 128,
                toolCallingEnabled: true
            ),
            .seconds(90)
        )
        XCTAssertEqual(
            LocalAPIInferencePolicy.deadline(
                maxTokens: 4_096,
                toolCallingEnabled: false
            ),
            .seconds(900)
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

    func testOpenAIRequestPrefersMaxCompletionTokens() throws {
        let request = try LocalAPIChatRequest.decodeOpenAI(Data("""
        {
          "model":"qwen",
          "messages":[{"role":"user","content":"Hello"}],
          "max_tokens":128,
          "max_completion_tokens":4096
        }
        """.utf8))
        XCTAssertEqual(request.maxTokens, 4_096)
    }

    func testOpenAINormalizesNoThinkingOptions() throws {
        let variants = [
            #"{"reasoning_effort":"none"}"#,
            #"{"think":false}"#,
            #"{"enable_thinking":false}"#,
            #"{"chat_template_kwargs":{"enable_thinking":false}}"#
        ]
        for variant in variants {
            var options = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Data(variant.utf8))
                    as? [String: Any]
            )
            options["model"] = "qwen"
            options["messages"] = [["role": "user", "content": "Write code"]]
            let data = try JSONSerialization.data(withJSONObject: options)
            let request = try LocalAPIChatRequest.decodeOpenAI(data)
            XCTAssertEqual(request.reasoningPreference, .disabled)
            XCTAssertTrue(
                request.reasoningPreference.forceNoThinking(globalEnabled: true)
            )
        }
    }

    func testReasoningPreferenceOverridesAndFallsBackToGlobalSetting() {
        XCTAssertTrue(
            LocalAPIReasoningPreference.disabled.forceNoThinking(
                globalEnabled: true
            )
        )
        XCTAssertFalse(
            LocalAPIReasoningPreference.enabled.forceNoThinking(
                globalEnabled: false
            )
        )
        XCTAssertTrue(
            LocalAPIReasoningPreference.automatic.forceNoThinking(
                globalEnabled: false
            )
        )
        XCTAssertFalse(
            LocalAPIReasoningPreference.automatic.forceNoThinking(
                globalEnabled: true
            )
        )
    }

    func testOpenAIStreamUsageOptionIsDecoded() throws {
        let request = try LocalAPIChatRequest.decodeOpenAI(Data("""
        {
          "model":"qwen",
          "messages":[{"role":"user","content":"Hello"}],
          "stream":true,
          "stream_options":{"include_usage":true}
        }
        """.utf8))
        XCTAssertTrue(request.streamIncludeUsage)
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
            [.system, .user, .assistant, .tool, .user]
        )
        XCTAssertEqual(request.messages[1].content, "show me the codebase")
        XCTAssertEqual(request.messages[2].content, "I will inspect the root.")
        XCTAssertEqual(request.messages[2].toolCalls?.first?.name, "terminal")
        XCTAssertEqual(request.messages[2].toolCalls?.first?.id, "tool_1")
        XCTAssertEqual(request.messages[3].content, "README.md\nsrc/main.py")
        XCTAssertEqual(request.messages[3].toolCallID, "tool_1")
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
        XCTAssertEqual(request.messages.map(\.role), [.user, .assistant, .tool])
        XCTAssertEqual(request.messages[1].content, "")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.name, "get_weather")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.id, "call_1")
        XCTAssertEqual(request.messages[2].content, "31 C, sunny")
        XCTAssertEqual(request.messages[2].toolCallID, "call_1")
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
        XCTAssertEqual(request.messages[0].content, "I will inspect that.")
        XCTAssertEqual(request.messages[0].toolCalls?.first?.name, "terminal")
        XCTAssertEqual(request.messages[0].toolCalls?.first?.id, "call_1")
    }

    func testOpenAIDecodesEmbeddedVisionAlongsideTools() throws {
        let data = Data("""
        {
          "model":"qwopus",
          "messages":[{
            "role":"user",
            "content":[
              {"type":"text","text":"Read the screenshot and use a tool if needed."},
              {"type":"image_url","image_url":{"url":"data:image/png;base64,\(onePixelPNG)"}}
            ]
          }],
          "tools":[{
            "type":"function",
            "function":{"name":"terminal","parameters":{"type":"object"}}
          }]
        }
        """.utf8)

        let request = try LocalAPIChatRequest.decodeOpenAI(data)
        XCTAssertEqual(request.messages[0].imageThumbnails.count, 1)
        XCTAssertEqual(request.messages[0].content, "Read the screenshot and use a tool if needed.")

        let prepared = LocalAPIToolCalling.messages(
            from: request.messages,
            tools: request.tools,
            choice: request.toolChoice,
            parallelToolCalls: false
        )
        XCTAssertEqual(prepared.last?.imageThumbnails.count, 1)
    }

    func testOpenAIRejectsSilentRemoteImageFetch() {
        let data = Data("""
        {
          "model":"qwopus",
          "messages":[{
            "role":"user",
            "content":[{
              "type":"image_url",
              "image_url":{"url":"https://example.com/private.png"}
            }]
          }]
        }
        """.utf8)

        XCTAssertThrowsError(try LocalAPIChatRequest.decodeOpenAI(data)) { error in
            XCTAssertTrue(String(describing: error).contains("Remote image URLs"))
        }
    }

    func testResponsesDecodesInputImageDataURL() throws {
        let data = Data("""
        {
          "model":"qwopus",
          "input":[{
            "type":"message",
            "role":"user",
            "content":[
              {"type":"input_text","text":"What is shown?"},
              {"type":"input_image","image_url":"data:image/png;base64,\(onePixelPNG)"}
            ]
          }]
        }
        """.utf8)

        let request = try LocalAPIChatRequest.decodeOpenAIResponses(data)
        XCTAssertEqual(request.messages[0].content, "What is shown?")
        XCTAssertEqual(request.messages[0].imageThumbnails.count, 1)
    }

    func testAnthropicDecodesBase64ImageWithToolDefinition() throws {
        let data = Data("""
        {
          "model":"qwopus",
          "max_tokens":128,
          "messages":[{
            "role":"user",
            "content":[
              {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(onePixelPNG)"}},
              {"type":"text","text":"Inspect this."}
            ]
          }],
          "tools":[{"name":"terminal","input_schema":{"type":"object"}}]
        }
        """.utf8)

        let request = try LocalAPIChatRequest.decodeAnthropic(data)
        XCTAssertEqual(request.messages[0].content, "Inspect this.")
        XCTAssertEqual(request.messages[0].imageThumbnails.count, 1)
        XCTAssertEqual(request.tools.first?.name, "terminal")
    }

    func testOllamaDecodesBase64Images() throws {
        let data = Data("""
        {
          "model":"qwopus",
          "stream":false,
          "messages":[{"role":"user","content":"Describe it","images":["\(onePixelPNG)"]}]
        }
        """.utf8)

        let request = try LocalAPIChatRequest.decodeOllamaChat(data)
        XCTAssertEqual(request.messages[0].imageThumbnails.count, 1)
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

    func testNativeToolTemplateDoesNotDuplicateSchemasOrWireSyntax() {
        let tool = LocalAPIToolDefinition(
            name: "terminal",
            description: "Run a command",
            parametersJSON: #"{"type":"object","properties":{"command":{"type":"string"}}}"#
        )
        let messages = LocalAPIToolCalling.messages(
            from: [ChatMessage(role: .user, content: "Run pwd")],
            tools: [tool],
            choice: .required,
            parallelToolCalls: false,
            parser: .hermes,
            nativeTemplate: true
        )

        XCTAssertEqual(messages.first?.role, .system)
        XCTAssertEqual(messages.last?.content, "Run pwd")
        XCTAssertTrue(messages.first?.content.contains("native tool interface") == true)
        XCTAssertFalse(messages.first?.content.contains("<tool_call>") == true)
        XCTAssertFalse(messages.first?.content.contains("properties") == true)
        // The instructions were interpolated as literal "(choiceInstruction)"
        // for long enough that tool_choice:"required" never reached the model.
        XCTAssertTrue(
            messages.first?.content.contains("You must call at least one available tool") == true,
            "choice instruction missing: \(messages.first?.content ?? "")"
        )
        XCTAssertTrue(
            messages.first?.content.contains("Return exactly one tool call at a time") == true
        )
        XCTAssertFalse(messages.first?.content.contains("(choiceInstruction)") == true)
        XCTAssertFalse(messages.first?.content.contains("(countInstruction)") == true)
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

        // A schema mismatch used to delete the call, which surfaced to the
        // client as a 502 blaming tool count and prompt length. The call is a
        // real call to a real tool, so it is forwarded for the client to judge.
        let mismatched = LocalAPIToolCalling.parse(
            #"{"tool_calls":[{"id":"call_bad","name":"terminal","arguments":{"command":12}}]}"#,
            tools: [tool],
            parallelToolCalls: true
        )
        XCTAssertEqual(mismatched.map(\.name), ["terminal"])
        XCTAssertEqual(mismatched.map(\.id), ["call_bad"])

        // A schema-valid call still wins over a mismatched one.
        let mixed = LocalAPIToolCalling.parse(
            #"{"tool_calls":[{"name":"terminal","arguments":{"command":12}},{"name":"terminal","arguments":{"command":"pwd"}}]}"#,
            tools: [tool],
            parallelToolCalls: false
        )
        XCTAssertEqual(mixed.map(\.argumentsJSON), [#"{"command":"pwd"}"#])

        // An invented tool name is still a hard reject.
        XCTAssertTrue(LocalAPIToolCalling.parse(
            #"{"tool_calls":[{"name":"not_a_tool","arguments":{"command":"pwd"}}]}"#,
            tools: [tool],
            parallelToolCalls: true
        ).isEmpty)
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

    func testToolCallingParserAcceptsNativeQwen3XMLAndCapsParallelCalls() {
        let tools = ["first", "second", "third"].map {
            LocalAPIToolDefinition(
                name: $0,
                description: nil,
                parametersJSON: #"{"properties":{"value":{"type":"integer"}},"required":["value"],"type":"object"}"#
            )
        }
        let calls = LocalAPIToolCalling.parse(
            """
            <tool_call><function=first><parameter=value>1</parameter></function></tool_call>
            <tool_call><function=second><parameter=value>2</parameter></function></tool_call>
            <tool_call><function=third><parameter=value>3</parameter></function></tool_call>
            """,
            tools: tools,
            parallelToolCalls: true,
            // The registered model may still advertise Hermes; the payload's
            // first non-whitespace character after <tool_call> is authoritative.
            parser: .hermes,
            maximumCalls: 2
        )
        XCTAssertEqual(calls.map(\.name), ["first", "second"])
        XCTAssertEqual(calls.first?.argumentsJSON, #"{"value":1}"#)
    }

    func testToolCallingParserAcceptsXMLTagVariantsAndSchemaCoercion() throws {
        let tool = LocalAPIToolDefinition(
            name: "get_weather",
            description: nil,
            parametersJSON: #"{"properties":{"city":{"type":"string"},"count":{"type":"integer"},"enabled":{"type":"boolean"},"options":{"type":"object"},"tags":{"type":"array"},"temperature":{"type":"number"}},"required":["city"],"type":"object"}"#
        )
        let calls = LocalAPIToolCalling.parse(
            """
            <tool_call>
              <function name="get_weather">
                <parameter="city"> Paris </parameter>
                <parameter name="count"> 5 </parameter>
                <parameter=temperature> 21.5 </parameter>
                <parameter=enabled> TRUE </parameter>
                <parameter=options> {"unit":"C"} </parameter>
                <parameter=tags> ["today", "tomorrow"] </parameter>
              </function>
              <function=get_weather>
                <parameter=city> New York </parameter>
              </function>
            </tool_call>
            """,
            tools: [tool],
            parallelToolCalls: true,
            parser: .hermes,
            maximumCalls: 2
        )

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(
            calls[0].argumentsJSON,
            #"{"city":"Paris","count":5,"enabled":true,"options":{"unit":"C"},"tags":["today","tomorrow"],"temperature":21.5}"#
        )
        XCTAssertTrue(calls[1].argumentsJSON.contains(#""city":"New York""#))
    }

    func testToolCallingParserSeparatesUnavailableToolFromMalformedDialect() {
        let tool = LocalAPIToolDefinition(
            name: "get_weather",
            description: nil,
            parametersJSON: #"{"properties":{"city":{"type":"string"}},"type":"object"}"#
        )
        let unavailable = LocalAPIToolCalling.parseResult(
            "<tool_call><function=delete_everything><parameter=city>x</parameter></function></tool_call>",
            tools: [tool],
            parallelToolCalls: false,
            parser: .hermes
        )
        XCTAssertEqual(unavailable.failure, .unavailableTool("delete_everything"))
        XCTAssertTrue(
            LocalAPIToolCalling.toolCallErrorMessage(for: unavailable, tools: [tool])
                .contains("unavailable tool")
        )

        let malformed = LocalAPIToolCalling.parseResult(
            "<tool_call><function=get_weather><parameter=city> Paris",
            tools: [tool],
            parallelToolCalls: false,
            parser: .hermes
        )
        guard case .malformed(let dialect, let rawPreview) = malformed.failure else {
            return XCTFail("Expected malformed XML result")
        }
        XCTAssertEqual(dialect, "xml")
        XCTAssertTrue(rawPreview.contains("<tool_call>"))
        let message = LocalAPIToolCalling.toolCallErrorMessage(for: malformed, tools: [tool])
        XCTAssertTrue(message.contains("server-side parser error"))
        XCTAssertTrue(message.contains("<tool_call>"))
    }

    func testToolCallingParserRepairsTrailingCommaOnce() {
        let tool = LocalAPIToolDefinition(
            name: "terminal",
            description: nil,
            parametersJSON: #"{"properties":{"command":{"type":"string"}},"required":["command"],"type":"object"}"#
        )
        let calls = LocalAPIToolCalling.parse(
            #"{"name":"terminal","arguments":{"command":"pwd",},}"#,
            tools: [tool],
            parallelToolCalls: false,
            parser: .hermes
        )
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
        // Template already opened <think>; the model emits CoT + </think> + answer.
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

    func testReasoningFilterParseAndStreamAgreeForQwenCloseOnly() {
        let raw = "chain of thought about the task</think>\nSTREAM_OK"
        let parsed = LocalAPIReasoningFilter.parse(raw, prefilledOpening: true)

        var filter = LocalAPIReasoningFilter(prefilledOpening: true)
        var reasoning = ""
        var content = ""
        for piece in [String(raw.prefix(12)), String(raw.dropFirst(12).prefix(20)), String(raw.dropFirst(32))] {
            let delta = filter.consume(piece)
            reasoning += delta.reasoning
            content += delta.content
        }
        let trailing = filter.finish()
        reasoning += trailing.reasoning
        content += trailing.content

        XCTAssertEqual(
            reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            parsed.reasoning
        )
        XCTAssertEqual(
            content.trimmingCharacters(in: .whitespacesAndNewlines),
            parsed.content
        )
        XCTAssertEqual(parsed.content, "STREAM_OK")
    }

    func testReasoningFilterStreamsOrphanedCloseWithoutPrefill() {
        // Sync parse() salvages close-only CoT. Streaming must not flush CoT
        // into delta.content before </think> arrives.
        var filter = LocalAPIReasoningFilter(prefilledOpening: false)
        let chunks = [
            "Let me analyze the request carefully. ",
            "The user wants STREAM_OK.</thi",
            "nk>\nSTREAM_OK"
        ]
        var reasoning = ""
        var content = ""
        for chunk in chunks {
            let delta = filter.consume(chunk)
            reasoning += delta.reasoning
            content += delta.content
            if !chunk.contains("nk>") {
                XCTAssertEqual(content, "", "content leaked before closing tag: \(content)")
            }
        }
        let trailing = filter.finish()
        reasoning += trailing.reasoning
        content += trailing.content

        XCTAssertTrue(reasoning.contains("Let me analyze"))
        XCTAssertFalse(content.contains("Let me analyze"))
        XCTAssertFalse(content.contains("think"))
        XCTAssertFalse(reasoning.contains("</think>"))
        XCTAssertEqual(content.trimmingCharacters(in: .whitespacesAndNewlines), "STREAM_OK")

        let parsed = LocalAPIReasoningFilter.parse(chunks.joined())
        XCTAssertEqual(
            reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            parsed.reasoning
        )
        XCTAssertEqual(
            content.trimmingCharacters(in: .whitespacesAndNewlines),
            parsed.content
        )
    }

    func testReasoningFilterParsePrefillWithoutClosePromotesToContent() {
        // parse() holds the whole completion, so it can still decide a no-tag
        // completion was never CoT. Streaming cannot — see the test below.
        let parsed = LocalAPIReasoningFilter.parse(
            "Just a normal answer with no tags.",
            prefilledOpening: true
        )
        XCTAssertEqual(parsed.reasoning, "")
        XCTAssertEqual(parsed.content, "Just a normal answer with no tags.")
    }

    func testReasoningFilterStreamsPrefilledReasoningWithoutWaitingForClose() {
        // Withholding CoT until </think> meant a client saw nothing until
        // generation finished. Reasoning now leaves as it arrives.
        var filter = LocalAPIReasoningFilter(prefilledOpening: true)
        let first = filter.consume("Let me think ")
        XCTAssertEqual(first.reasoning, "Let me think ")
        let second = filter.consume("about it.</think>DONE")
        XCTAssertEqual(second.reasoning, "about it.")
        XCTAssertEqual(second.content, "DONE")
    }

    func testReasoningFilterStreamsTagFreeAnswerProgressively() {
        // The regression this guards: a plain answer (thinking disabled, so no
        // tags at all) used to be held whole and flushed in one delta at the
        // end of generation.
        let answer = String(repeating: "The sea is wide and deep. ", count: 75)
        var filter = LocalAPIReasoningFilter(prefilledOpening: false)
        var deltas: [LocalAPIReasoningDelta] = []
        for token in stride(from: 0, to: answer.count, by: 4).map({ offset -> String in
            let start = answer.index(answer.startIndex, offsetBy: offset)
            let end = answer.index(start, offsetBy: 4, limitedBy: answer.endIndex)
                ?? answer.endIndex
            return String(answer[start..<end])
        }) {
            let delta = filter.consume(token)
            if !delta.isEmpty { deltas.append(delta) }
        }
        let trailing = filter.finish()
        if !trailing.isEmpty { deltas.append(trailing) }

        XCTAssertEqual(deltas.map(\.content).joined(), answer)
        XCTAssertGreaterThan(deltas.count, 100, "answer arrived in \(deltas.count) deltas")
        // Only the opening speculation window may batch; the rest is per-token.
        XCTAssertLessThanOrEqual(deltas.dropFirst().map(\.content.count).max() ?? 0, 8)
    }

    func testReasoningFilterStreamsReasoningMarkupFamily() {
        var filter = LocalAPIReasoningFilter()
        let deltas = [
            "<reas",
            "oning>chain of thought</reas",
            "oning>\nFINAL"
        ].map { filter.consume($0) }
        let trailing = filter.finish()
        let reasoning = deltas.map(\.reasoning).joined() + trailing.reasoning
        let content = deltas.map(\.content).joined() + trailing.content

        XCTAssertEqual(
            reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
            "chain of thought"
        )
        XCTAssertEqual(content.trimmingCharacters(in: .whitespacesAndNewlines), "FINAL")
        XCTAssertFalse(content.contains("reasoning"))
        XCTAssertFalse(reasoning.contains("<"))
    }

    func testReasoningFilterCloseWinsOverLaterOpeningInsideBlock() {
        // A closing marker must beat a later echoed open by position, or
        // streaming would swallow `</think>` into reasoning_content.
        var filter = LocalAPIReasoningFilter()
        let delta = filter.consume("<think>a</think><think>b</think>c")
        let trailing = filter.finish()

        XCTAssertEqual(delta.reasoning + trailing.reasoning, "ab")
        XCTAssertEqual(delta.content + trailing.content, "c")
    }

    func testReasoningFilterPrefillAndParseAgreeAcrossChunkBoundaries() {
        let raw = "Let me analyze carefully.</think>\nSTREAM_OK"
        let parsed = LocalAPIReasoningFilter.parse(raw, prefilledOpening: true)
        let chars = Array(raw)
        for split in [1, 8, 17, chars.count / 2] {
            var filter = LocalAPIReasoningFilter(prefilledOpening: true)
            var reasoning = ""
            var content = ""
            let first = String(chars.prefix(split))
            let second = String(chars.suffix(chars.count - split))
            for piece in [first, second] {
                let delta = filter.consume(piece)
                reasoning += delta.reasoning
                content += delta.content
            }
            let trailing = filter.finish()
            reasoning += trailing.reasoning
            content += trailing.content
            XCTAssertEqual(
                reasoning.trimmingCharacters(in: .whitespacesAndNewlines),
                parsed.reasoning,
                "split=\(split)"
            )
            XCTAssertEqual(
                content.trimmingCharacters(in: .whitespacesAndNewlines),
                parsed.content,
                "split=\(split)"
            )
        }
        XCTAssertEqual(parsed.content, "STREAM_OK")
        XCTAssertFalse(parsed.content.contains("think"))
        XCTAssertFalse(parsed.reasoning.contains("</"))
    }

    func testOpenAIChunkEmitsReasoningWithoutContentKey() throws {
        let data = LocalAPIResponse.openAIChunk(
            id: "chatcmpl-test",
            model: "qwen",
            text: "",
            role: "assistant",
            reasoningContent: "private reasoning"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let choices = try XCTUnwrap(object["choices"] as? [[String: Any]])
        let delta = try XCTUnwrap(choices.first?["delta"] as? [String: Any])
        XCTAssertEqual(delta["reasoning_content"] as? String, "private reasoning")
        XCTAssertNil(delta["content"])
        XCTAssertEqual(delta["role"] as? String, "assistant")
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
        XCTAssertEqual(request.messages[1].toolCalls?.first?.id, "call_123")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.name, "terminal")
        XCTAssertEqual(request.messages[2].role, .tool)
        XCTAssertEqual(request.messages[2].toolCallID, "call_123")
        XCTAssertEqual(request.messages[2].content, "/tmp")
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
        XCTAssertEqual(request.messages[1].content, "")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.name, "get_weather")
        XCTAssertEqual(request.messages[2].role, .tool)
        XCTAssertEqual(request.messages[2].content, "31 C, sunny")
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
        XCTAssertEqual(request.messages[1].content, "")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.id, "toolu_1")
        XCTAssertEqual(request.messages[1].toolCalls?.first?.name, "get_weather")
        XCTAssertEqual(request.messages[2].role, .tool)
        XCTAssertEqual(request.messages[2].toolCallID, "toolu_1")
        XCTAssertEqual(request.messages[2].content, "31 C, sunny")
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

    func testAssistantOutputValidatorRejectsEmptyAndReasoningOnlyAnswers() {
        XCTAssertFalse(
            LocalAPIAssistantOutputValidator.hasAnswer(text: "", toolCalls: [])
        )
        XCTAssertFalse(
            LocalAPIAssistantOutputValidator.hasAnswer(
                text: "  \n\t",
                toolCalls: []
            )
        )
    }

    func testAssistantOutputValidatorAcceptsTextOrToolCalls() {
        XCTAssertTrue(
            LocalAPIAssistantOutputValidator.hasAnswer(text: "OK", toolCalls: [])
        )
        XCTAssertTrue(
            LocalAPIAssistantOutputValidator.hasAnswer(
                text: "",
                toolCalls: [
                    LocalAPIToolCall(
                        id: "call_test",
                        name: "terminal",
                        argumentsJSON: #"{"command":"pwd"}"#
                    )
                ]
            )
        )
    }

    func testRemoteInferenceGateReleaseIsImmediateAndOwnerScoped() throws {
        let gate = RemoteInferenceGate()
        let first = try XCTUnwrap(gate.acquire())
        XCTAssertNil(gate.acquire())

        gate.release(UUID())
        XCTAssertNil(gate.acquire(), "A different request must not release the busy owner")

        gate.release(first)
        let second = try XCTUnwrap(gate.acquire())
        gate.release(second)
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

    func testOpenAIResponseReportsNormalStopAndExactUsage() throws {
        let usage = LocalAPIUsage(AssistantGenerationResult(
            promptTokenCount: 37,
            completionTokenCount: 128,
            stopReason: .stop
        ))
        let data = LocalAPIResponse.openAIChatCompletion(
            id: "chatcmpl-stop",
            model: "ornith-1.0-9b-4bit",
            text: "Done.",
            toolCalls: [],
            finishReason: "stop",
            usage: usage
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "stop")
        let encodedUsage = try XCTUnwrap(object["usage"] as? [String: Int])
        XCTAssertEqual(encodedUsage["prompt_tokens"], 37)
        XCTAssertEqual(encodedUsage["completion_tokens"], 128)
        XCTAssertEqual(encodedUsage["total_tokens"], 165)
    }

    func testOpenAIResponseReportsLengthStop() throws {
        let data = LocalAPIResponse.openAIChatCompletion(
            id: "chatcmpl-length",
            model: "ornith-1.0-9b-4bit",
            text: "partial",
            toolCalls: [],
            finishReason: "length",
            usage: LocalAPIUsage(AssistantGenerationResult(
                promptTokenCount: 12,
                completionTokenCount: 4_096,
                stopReason: .length
            ))
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let choice = try XCTUnwrap((object["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "length")
        let usage = try XCTUnwrap(object["usage"] as? [String: Int])
        XCTAssertEqual(usage["completion_tokens"], 4_096)
    }

    func testOpenAIStreamTerminatorIncludesFinishUsageAndDone() throws {
        let data = LocalAPIResponse.openAIStreamTerminator(
            id: "chatcmpl-stream",
            model: "ornith-1.0-9b-4bit",
            finishReason: "length",
            usage: LocalAPIUsage(AssistantGenerationResult(
                promptTokenCount: 21,
                completionTokenCount: 1_024,
                stopReason: .length
            )),
            includeUsage: true
        )
        let text = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertTrue(text.hasSuffix("data: [DONE]\n\n"))

        let payloads = text
            .components(separatedBy: "\n\n")
            .filter { $0.hasPrefix("data: {") }
            .map { String($0.dropFirst("data: ".count)) }
        XCTAssertEqual(payloads.count, 2)

        let finish = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(payloads[0].utf8))
                as? [String: Any]
        )
        let choice = try XCTUnwrap((finish["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual(choice["finish_reason"] as? String, "length")

        let usageChunk = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(payloads[1].utf8))
                as? [String: Any]
        )
        XCTAssertTrue((usageChunk["choices"] as? [Any])?.isEmpty == true)
        let usage = try XCTUnwrap(usageChunk["usage"] as? [String: Int])
        XCTAssertEqual(usage["prompt_tokens"], 21)
        XCTAssertEqual(usage["completion_tokens"], 1_024)
        XCTAssertEqual(usage["total_tokens"], 1_045)
    }

    func testInferenceSignalPreservesTerminalGenerationMetadata() {
        let signal = LocalAPIInferenceSignal()
        let expected = AssistantGenerationResult(
            promptTokenCount: 11,
            completionTokenCount: 2_048,
            stopReason: .length
        )
        signal.record(expected)
        XCTAssertEqual(signal.result, expected)
    }

    func testClientDisconnectReleasesConnectionWaiter() async {
        let monitor = LocalAPIConnectionMonitor()
        let released = expectation(description: "disconnect waiter released")
        Task {
            await monitor.wait()
            released.fulfill()
        }
        await Task.yield()
        monitor.markDisconnected()
        await fulfillment(of: [released], timeout: 1)
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

    func testOpenAIToolCallStreamingShapeForOpenCode() throws {
        let call = LocalAPIToolCall(
            id: "call_stream",
            name: "terminal",
            argumentsJSON: #"{"command":"pwd"}"#
        )
        let callData = LocalAPIResponse.openAIToolCallChunk(
            id: "chatcmpl-stream",
            model: "qwen",
            calls: [call]
        )
        let callObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: callData) as? [String: Any]
        )
        let callChoices = try XCTUnwrap(callObject["choices"] as? [[String: Any]])
        let delta = try XCTUnwrap(callChoices.first?["delta"] as? [String: Any])
        XCTAssertEqual(delta["role"] as? String, "assistant")
        let calls = try XCTUnwrap(delta["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls.first?["index"] as? Int, 0)
        XCTAssertEqual(calls.first?["id"] as? String, "call_stream")
        let function = try XCTUnwrap(calls.first?["function"] as? [String: Any])
        XCTAssertEqual(function["name"] as? String, "terminal")
        XCTAssertEqual(function["arguments"] as? String, #"{"command":"pwd"}"#)

        let finishData = LocalAPIResponse.openAIToolCallChunk(
            id: "chatcmpl-stream",
            model: "qwen",
            calls: [],
            finishReason: "tool_calls"
        )
        let finishObject = try XCTUnwrap(
            JSONSerialization.jsonObject(with: finishData) as? [String: Any]
        )
        let finishChoices = try XCTUnwrap(finishObject["choices"] as? [[String: Any]])
        XCTAssertEqual(finishChoices.first?["finish_reason"] as? String, "tool_calls")
        XCTAssertTrue((finishChoices.first?["delta"] as? [String: Any])?.isEmpty == true)
    }

    func testOpenAIToolCallStreamingFragmentsKeepStableIndex() throws {
        let data = LocalAPIResponse.openAIToolCallFragmentChunk(
            id: "chatcmpl-stream",
            model: "qwen",
            index: 1,
            callID: nil,
            name: nil,
            arguments: #"{"city":"Paris"}"#,
            includeRole: false
        )
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let delta = try XCTUnwrap(
            (object["choices"] as? [[String: Any]])?.first?["delta"] as? [String: Any]
        )
        let calls = try XCTUnwrap(delta["tool_calls"] as? [[String: Any]])
        XCTAssertEqual(calls.first?["index"] as? Int, 1)
        XCTAssertNil(calls.first?["id"])
        XCTAssertEqual(
            (calls.first?["function"] as? [String: Any])?["arguments"] as? String,
            #"{"city":"Paris"}"#
        )
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
        XCTAssertTrue(request?.wantsKeepAlive == true)
    }

    func testHTTPRequestAcceptsOptionalHeaderWhitespace() {
        let request = HTTPRequest(data: Data("""
        post /v1/models HTTP/1.1\r
        Authorization:\tbearer abc\r
        Content-Type:application/json\r
        Connection: close\r
        Content-Length: 0\r
        \r

        """.utf8))
        XCTAssertEqual(request?.method, "POST")
        XCTAssertEqual(request?.path, "/v1/models")
        XCTAssertEqual(request?.headers["authorization"], "bearer abc")
        XCTAssertNil(request?.body)
        XCTAssertFalse(request?.wantsKeepAlive == true)
    }
}
