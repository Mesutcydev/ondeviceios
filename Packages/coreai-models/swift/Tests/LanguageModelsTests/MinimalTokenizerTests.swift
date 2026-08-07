// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// MARK: - ByteLevel Encoding Regression Tests
//
// Tests for Issue #139: swift-transformers ByteLevel pre-tokenizer handling.
// Uses a minimal tokenizer JSON with Sequence pre_tokenizer (nested ByteLevel)
// matching the structure used by Qwen and other modern tokenizers.

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Test Resource Helper

/// Helper to locate test resources bundled with the test target
enum TestResources {
    /// Get URL for a test resource directory
    static func url(for directory: String) -> URL? {
        // Bundle.module works in CI and proper SPM builds
        if let bundleURL = Bundle.module.url(forResource: directory, withExtension: nil) {
            return bundleURL
        }

        // Fallback to #filePath for local development
        let sourceFile = URL(fileURLWithPath: #filePath)
        let resourceURL = sourceFile.deletingLastPathComponent().appendingPathComponent("Resources/\(directory)")
        if FileManager.default.fileExists(atPath: resourceURL.path) {
            return resourceURL
        }

        return nil
    }
}

// MARK: - ByteLevel Helper Function Tests

@Suite("ByteLevelEncoding")
struct ByteLevelEncodingTests {
    @Test("byteleveled() converts control characters correctly")
    func byteLeveledFunction() {
        #expect("hello\nworld".byteleveled() == "helloĊworld")
        #expect("hello\tworld".byteleveled() == "helloĉworld")
        #expect("hello world".byteleveled() == "helloĠworld")
    }
}

// MARK: - Issue #139 Regression Tests
//
// These tests verify that swift-transformers correctly handles:
// 1. Sequence pre_tokenizer (nested structure like Qwen uses)
// 2. ByteLevel encoding within the Sequence
//
// The test tokenizer JSON uses the same structure as Qwen:
// "pre_tokenizer": { "type": "Sequence", "pretokenizers": [Split, ByteLevel] }

@Suite("ByteLevel Issue139 Regression")
struct ByteLevelIssue139RegressionTests {
    @Test("AutoTokenizer loads JSON with Sequence pre_tokenizer")
    func autoTokenizerLoadsCorrectly() async throws {
        guard let tokenizerURL = TestResources.url(for: "MinimalTokenizer") else {
            Issue.record("MinimalTokenizer resource directory not found")
            return
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let tokens = tokenizer.encode(text: "A")
        #expect(!tokens.isEmpty, "Tokenizer should encode text")
    }

    @Test("REGRESSION #139: Newlines encode to Ċ token")
    func newlinesEncodeToByteLevelToken() async throws {
        guard let tokenizerURL = TestResources.url(for: "MinimalTokenizer") else {
            Issue.record("MinimalTokenizer resource directory not found")
            return
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let tokens = tokenizer.encode(text: "?\nAnswer:")

        // Issue #139: Without ByteLevel, \n would become <unk> (1)
        // With ByteLevel: \n → Ċ → token 121
        #expect(tokens.contains(121), "REGRESSION #139: Should have ByteLevel Ċ token (121) for newline")
        #expect(!tokens.contains(1), "Should NOT have <unk> token for newline")
    }

    @Test("REGRESSION #139: Spaces encode to Ġ token")
    func spacesEncodeToByteLevelToken() async throws {
        guard let tokenizerURL = TestResources.url(for: "MinimalTokenizer") else {
            Issue.record("MinimalTokenizer resource directory not found")
            return
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        let tokens = tokenizer.encode(text: "Answer: B")

        // Issue #139: Without ByteLevel, space would become <unk> (1)
        // With ByteLevel: space → Ġ → token 122
        #expect(tokens.contains(122), "REGRESSION #139: Should have ByteLevel Ġ token (122) for space")
        #expect(!tokens.contains(1), "Should NOT have <unk> token for space")
    }

    @Test("REGRESSION #139: Continuation extraction works with ByteLevel")
    func continuationExtractionWithByteLevel() async throws {
        guard let tokenizerURL = TestResources.url(for: "MinimalTokenizer") else {
            Issue.record("MinimalTokenizer resource directory not found")
            return
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)

        // MMLU evaluation pattern: context + answer → continuation extraction
        let context = "Answer:"
        let full = "Answer: B"

        let contextTokens = tokenizer.encode(text: context)
        let fullTokens = tokenizer.encode(text: full)

        // Critical: full tokens must start with context tokens
        let fullPrefix = Array(fullTokens.prefix(contextTokens.count))
        #expect(fullPrefix == contextTokens, "Full should start with context tokens")

        // Continuation should contain space (Ġ) and B
        let continuation = Array(fullTokens.dropFirst(contextTokens.count))
        #expect(continuation.contains(122), "Continuation should have Ġ (122) for space")
        #expect(continuation.contains(5), "Continuation should have B (5)")
    }
}
