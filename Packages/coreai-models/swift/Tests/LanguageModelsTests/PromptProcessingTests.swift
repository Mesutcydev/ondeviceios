// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("PromptProcessing")
struct PromptProcessingTests {
    // MARK: - RawTokensInput
    @Test("RawTokensInput decodes from JSON")
    func rawTokensDecodesFromJSON() throws {
        let json = """
            {"tokens": [100, 200, 300]}
            """
        let data = json.data(using: .utf8)!

        let input = try JSONDecoder().decode(RawTokensInput.self, from: data)

        #expect(input.tokens == [100, 200, 300])
    }

    // MARK: - PromptInput

    @Test("fromTextFile loads valid file")
    func textFileLoads() throws {
        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).txt")
        try "Test prompt".write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let input = try PromptInput.fromTextFile(path: tempFile.path)

        if case .text(let content) = input {
            #expect(content == "Test prompt")
        } else {
            Issue.record("Expected .text case")
        }
    }

    @Test("fromRawTokensFile loads valid JSON")
    func rawTokensFileLoads() throws {
        let tempFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_\(UUID().uuidString).json")
        try """
        {"tokens": [1, 2, 3]}
        """.write(to: tempFile, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tempFile) }

        let input = try PromptInput.fromRawTokensFile(path: tempFile.path)

        if case .rawTokens(let container) = input {
            #expect(container.tokens.count == 3)
        } else {
            Issue.record("Expected .rawTokens case")
        }
    }

    // MARK: - PromptInputResolver

    @Test("Resolver returns default when nothing specified")
    func resolverDefault() throws {
        let input = try PromptInputResolver.resolve(
            prompt: nil, promptFile: nil, rawTokens: nil,
            default: "Test default"
        )
        if case .text(let content) = input {
            #expect(content == "Test default")
        } else {
            Issue.record("Expected .text case")
        }
    }

    @Test("Resolver throws for multiple sources")
    func resolverMultipleSources() {
        #expect(throws: PromptInputError.self) {
            _ = try PromptInputResolver.resolve(
                prompt: "A", promptFile: "B", rawTokens: nil,
                default: "unused"
            )
        }
    }

    // MARK: - Preview

    @Test("Preview summary format")
    func previewSummary() {
        let truncated = RawTokensInput.Preview(tokenCount: 100, decodedText: "Hello", isTruncated: true)
        let full = RawTokensInput.Preview(tokenCount: 5, decodedText: "Hi", isTruncated: false)

        #expect(truncated.summary == "(100 tokens) Hello...")
        #expect(full.summary == "(5 tokens) Hi")
    }
}
