import XCTest
@testable import IOSLocalLLM

// MARK: - BM25ScorerTests
//
// The deterministic lexical fallback ranker (also the floor under the semantic
// reranker). Ranking must put query-matching passages first.

final class BM25ScorerTests: XCTestCase {

    private func passage(_ i: Int, _ text: String) -> Passage {
        Passage(pageIndex: 0, chunkIndex: i, text: text, approxTokens: max(1, text.count / 4))
    }

    func test_matchingPassageScoresHigher() {
        let scorer = BM25Scorer()
        let passages = [
            passage(0, "the quick brown fox jumps over the lazy dog"),
            passage(1, "lorem ipsum dolor sit amet consectetur"),
        ]
        let scores = scorer.score(query: "fox", passages: passages)
        XCTAssertEqual(scores.count, 2)
        XCTAssertGreaterThan(scores[0], scores[1])
    }

    func test_emptyQueryScoresZero() {
        let scorer = BM25Scorer()
        let scores = scorer.score(query: "   ", passages: [passage(0, "anything here")])
        XCTAssertEqual(scores, [0.0])
    }

    func test_emptyPassagesReturnsEmpty() {
        XCTAssertTrue(BM25Scorer().score(query: "fox", passages: []).isEmpty)
    }
}

// MARK: - ToolRunnerParsingTests
//
// Tool-call extraction (incl. the new knowledge_base tool) and a couple of the
// pure, on-device tool implementations.

final class ToolRunnerParsingTests: XCTestCase {

    func test_extractKnowledgeBaseCall() {
        let reply = """
        Let me check your files.
        ```tool
        {"name": "knowledge_base", "args": {"query": "auth flow"}}
        ```
        """
        let call = ToolRunner.extractCall(from: reply)
        XCTAssertEqual(call?.name, "knowledge_base")
        XCTAssertEqual(call?.args["query"] as? String, "auth flow")
    }

    func test_extractCalculatorCall() {
        let reply = "```tool\n{\"name\": \"calculator\", \"args\": {\"expression\": \"(3+5)*2\"}}\n```"
        XCTAssertEqual(ToolRunner.extractCall(from: reply)?.name, "calculator")
    }

    func test_noToolBlockReturnsNil() {
        XCTAssertNil(ToolRunner.extractCall(from: "just a normal answer with no tool block"))
    }

    func test_runCalculator() async {
        let result = await ToolRunner.run(ToolCall(name: "calculator", args: ["expression": "2+3"]))
        XCTAssertEqual(result, "5")
    }

    func test_unknownToolReturnsError() async {
        let result = await ToolRunner.run(ToolCall(name: "definitely_not_a_tool", args: [:]))
        XCTAssertTrue(result.lowercased().contains("unknown"))
    }

    func test_resultBlockIsParseableJSON() {
        let block = ToolRunner.resultBlock(name: "calculator", result: "multi\nline\nresult")
        XCTAssertTrue(block.contains("```tool_result"))
        XCTAssertTrue(block.contains("calculator"))
    }
}
