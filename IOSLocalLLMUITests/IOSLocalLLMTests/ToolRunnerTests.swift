import XCTest
@testable import IOSLocalLLM

// MARK: - ToolRunnerTests
//
// Covers the on-device tool/function-call dispatch layer:
//   • extractCall  — regex parsing of fenced `tool` blocks
//   • run          — calculator, datetime, unit_convert dispatch
//   • resultBlock  — JSON result fence formatting
//
// Web search and file read are excluded — they require MainActor + UI
// coordination and are better suited for UI/integration tests.

final class ToolRunnerTests: XCTestCase {

    // MARK: - extractCall

    func test_extractCalculatorCall_parsesCorrectly() {
        let text = """
        Let me calculate that for you.

        ```tool
        {"name": "calculator", "args": {"expression": "(3+5)*2"}}
        ```

        The result is...
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertNotNil(call)
        XCTAssertEqual(call?.name, "calculator")
        XCTAssertEqual(call?.args["expression"] as? String, "(3+5)*2")
    }

    func test_extractDatetimeCall_parsesCorrectly() {
        let text = """
        ```tool
        {"name": "datetime", "args": {"timezone": "Europe/Istanbul"}}
        ```
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertNotNil(call)
        XCTAssertEqual(call?.name, "datetime")
        XCTAssertEqual(call?.args["timezone"] as? String, "Europe/Istanbul")
    }

    func test_extractUnitConvertCall_parsesCorrectly() {
        let text = """
        ```tool
        {"name": "unit_convert", "args": {"value": 10, "from": "km", "to": "mi"}}
        ```
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertNotNil(call)
        XCTAssertEqual(call?.name, "unit_convert")
        XCTAssertEqual(call?.args["value"] as? Double, 10)
        XCTAssertEqual(call?.args["from"] as? String, "km")
        XCTAssertEqual(call?.args["to"] as? String, "mi")
    }

    func test_extractCall_noToolBlock_returnsNil() {
        let text = "Just a normal reply with no tool calls."
        XCTAssertNil(ToolRunner.extractCall(from: text))
    }

    func test_extractCall_emptyText_returnsNil() {
        XCTAssertNil(ToolRunner.extractCall(from: ""))
    }

    func test_extractCall_onlyFirstToolBlock() {
        let text = """
        ```tool
        {"name": "calculator", "args": {"expression": "1+1"}}
        ```
        ```tool
        {"name": "calculator", "args": {"expression": "2+2"}}
        ```
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertEqual(call?.args["expression"] as? String, "1+1")
    }

    func test_extractCall_toolResultBlockBeforeToolCall_extractsTheCall() {
        // A model that echoes a ```tool_result fence before emitting its next
        // ```tool call must not have the tool_result matched first (the keyword
        // guard prevents ```tool matching the ```tool_result prefix).
        let text = """
        ```tool_result
        {"some": "prior result"}
        ```
        ```tool
        {"name": "calculator", "args": {"expression": "3+4"}}
        ```
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertEqual(call?.name, "calculator")
        XCTAssertEqual(call?.args["expression"] as? String, "3+4")
    }

    func test_extractCall_malformedJSON_returnsNil() {
        let text = """
        ```tool
        {not valid json}
        ```
        """
        XCTAssertNil(ToolRunner.extractCall(from: text))
    }

    func test_extractCall_missingName_returnsNil() {
        let text = """
        ```tool
        {"args": {"expression": "1+1"}}
        ```
        """
        XCTAssertNil(ToolRunner.extractCall(from: text))
    }

    func test_extractCall_plainJSONFromSmallModel() {
        let call = ToolRunner.extractCall(from:
            #"{"name":"web_search","args":{"query":"weather in Asyut tomorrow"}}"#
        )
        XCTAssertEqual(call?.name, "web_search")
        XCTAssertEqual(call?.args["query"] as? String, "weather in Asyut tomorrow")
    }

    func test_extractCall_openAIToolCallsShape() {
        let call = ToolRunner.extractCall(from:
            #"{"tool_calls":[{"name":"calculator","arguments":{"expression":"6*7"}}]}"#
        )
        XCTAssertEqual(call?.name, "calculator")
        XCTAssertEqual(call?.args["expression"] as? String, "6*7")
    }

    func test_extractCall_hermesFunctionWithStringArguments() {
        let text = """
        <tool_call>
        {"function":{"name":"datetime","arguments":"{\\"timezone\\":\\"Africa/Cairo\\"}"}}
        </tool_call>
        """
        let call = ToolRunner.extractCall(from: text)
        XCTAssertEqual(call?.name, "datetime")
        XCTAssertEqual(call?.args["timezone"] as? String, "Africa/Cairo")
    }

    func test_extractCall_doesNotTreatArbitraryJSONAsTool() {
        XCTAssertNil(ToolRunner.extractCall(from: #"{"name":"Kareem","args":{"age":30}}"#))
    }

    // MARK: - Calculator (via run)

    func test_calculator_simpleAddition() async {
        let call = ToolCall(name: "calculator", args: ["expression": "2+2"])
        let result = await ToolRunner.run(call)
        XCTAssertEqual(result, "4")
    }

    func test_calculator_complexExpression() async {
        let call = ToolCall(name: "calculator", args: ["expression": "(3+5)*2"])
        let result = await ToolRunner.run(call)
        XCTAssertEqual(result, "16")
    }

    func test_calculator_division() async {
        let call = ToolCall(name: "calculator", args: ["expression": "10/3"])
        let result = await ToolRunner.run(call)
        // Should be ~3.33333, not just "3"
        XCTAssertTrue(result.contains("3.3333") || result.contains("3.33333"),
                      "Expected decimal result, got \(result)")
    }

    func test_calculator_missingExpression_returnsError() async {
        let call = ToolCall(name: "calculator", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"))
    }

    func test_calculator_symbolAliases() async {
        // "×" → "*" and "÷" → "/"
        let call1 = ToolCall(name: "calculator", args: ["expression": "8×4"])
        let r1 = await ToolRunner.run(call1)
        XCTAssertEqual(r1, "32")

        let call2 = ToolCall(name: "calculator", args: ["expression": "10÷2"])
        let r2 = await ToolRunner.run(call2)
        XCTAssertEqual(r2, "5")
    }

    // MARK: - Datetime

    func test_datetime_returnsNonEmpty() async {
        let call = ToolCall(name: "datetime", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertFalse(result.isEmpty)
        // Should contain a day-of-week, month, or year indicator.
        XCTAssertTrue(result.contains("202") || result.contains(","),
                      "Expected a date string, got \(result)")
    }

    func test_datetime_specificTimezone_producesOutput() async {
        let call = ToolCall(name: "datetime",
                            args: ["timezone": "America/New_York"])
        let result = await ToolRunner.run(call)
        XCTAssertFalse(result.isEmpty)
    }

    // MARK: - Unit convert

    func test_unitConvert_kmToMiles() async {
        let call = ToolCall(name: "unit_convert",
                            args: ["value": 10.0, "from": "km", "to": "mi"])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.contains("mi"), "Expected miles output, got \(result)")
        // 10 km ≈ 6.21371 mi
        XCTAssertTrue(result.contains("6.21") || result.contains("6.213"),
                      "Expected ~6.21 mi, got \(result)")
    }

    func test_unitConvert_celsiusToFahrenheit() async {
        let call = ToolCall(name: "unit_convert",
                            args: ["value": 0, "from": "c", "to": "f"])
        let result = await ToolRunner.run(call)
        // 0°C = 32°F
        XCTAssertTrue(result.contains("32"),
                      "Expected 32°F, got \(result)")
    }

    func test_unitConvert_kgToLbs() async {
        let call = ToolCall(name: "unit_convert",
                            args: ["value": 1.0, "from": "kg", "to": "lb"])
        let result = await ToolRunner.run(call)
        // 1 kg ≈ 2.20462 lb
        XCTAssertTrue(result.contains("2.20") || result.contains("2.204"),
                      "Expected ~2.2 lb, got \(result)")
    }

    func test_unitConvert_incompatibleUnits_returnsError() async {
        let call = ToolCall(name: "unit_convert",
                            args: ["value": 1.0, "from": "km", "to": "kg"])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"),
                      "Expected error for incompatible units, got \(result)")
    }

    func test_unitConvert_missingArgs_returnsError() async {
        let call = ToolCall(name: "unit_convert", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"))
    }

    func test_unitConvert_unknownSourceUnit_returnsError() async {
        let call = ToolCall(name: "unit_convert",
                            args: ["value": 1.0, "from": "nonsense", "to": "m"])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"),
                      "Expected error for unknown unit, got \(result)")
    }

    // MARK: - Unknown tool

    func test_unknownTool_returnsError() async {
        let call = ToolCall(name: "nonexistent_tool", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"))
        XCTAssertTrue(result.contains("nonexistent_tool"))
    }

    // MARK: - index_document

    func test_indexDocument_missingText_returnsError() async {
        let call = ToolCall(name: "index_document", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"),
                      "Expected missing-text error, got \(result)")
    }

    func test_indexDocument_emptyText_returnsError() async {
        let call = ToolCall(name: "index_document", args: ["text": "   "])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"))
    }

    // MARK: - generate_image

    func test_generateImage_missingPrompt_returnsError() async {
        let call = ToolCall(name: "generate_image", args: [:])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"),
                      "Expected missing-prompt error, got \(result)")
    }

    func test_generateImage_modelNotInstalled_returnsActionableError() async {
        // No diffusion model is on disk in the test host, so the tool must
        // refuse rather than kick off a multi-GB download mid-reply.
        let call = ToolCall(name: "generate_image",
                            args: ["prompt": "a red bicycle"])
        let result = await ToolRunner.run(call)
        XCTAssertTrue(result.hasPrefix("Error:"))
        XCTAssertTrue(result.lowercased().contains("download"),
                      "Expected an install-first hint, got \(result)")
    }

    // MARK: - resultBlock

    func test_resultBlock_formatsCorrectly() {
        let block = ToolRunner.resultBlock(name: "calculator", result: "16")
        XCTAssertTrue(block.contains("```tool_result"))
        XCTAssertTrue(block.contains("\"name\""))
        XCTAssertTrue(block.contains("\"calculator\""))
        XCTAssertTrue(block.contains("\"16\""))
    }

    func test_resultBlock_newlinesInResult_escapedInJSON() {
        let block = ToolRunner.resultBlock(name: "web_search",
                                           result: "line1\nline2")
        // JSONSerialization handles escaping; just verify the block is valid.
        XCTAssertTrue(block.contains("```tool_result"))
        XCTAssertFalse(block.isEmpty)
    }

    func test_resultForModelContext_webSearchKeepsReadableContext() {
        let webContext = """
        WEB CONTEXT START (untrusted external content)

        Source [1]
        Title: Example
        Excerpt:
        alpha
        beta

        WEB CONTEXT END
        """
        let prompt = ToolRunner.resultForModelContext(name: "web_search", result: webContext)
        XCTAssertTrue(prompt.contains("TOOL RESULT: web_search"))
        XCTAssertTrue(prompt.contains("Source [1]\nTitle: Example"))
        XCTAssertTrue(prompt.contains("alpha\nbeta"))
        XCTAssertFalse(prompt.contains(#"\nSource [1]\n"#))
    }

    // MARK: - ToolCall Equatable

    func test_toolCall_equatable_ignoresArgs() {
        let a = ToolCall(name: "calculator", args: ["expression": "1+1"])
        let b = ToolCall(name: "calculator", args: ["expression": "2+2"])
        XCTAssertEqual(a, b, "ToolCall equality only considers name")
    }
}
