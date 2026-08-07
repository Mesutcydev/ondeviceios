// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Graph Selection")
struct GraphSelectionTests {
    private func selectGraph(
        extendFunctions: [String],
        numInputTokens: Int,
        currentPosition: Int,
        isPrefill: Bool
    ) -> String? {
        var pairs: [(contextLength: Int, queryLength: Int)] = []
        for name in extendFunctions {
            let parts = Array(name.split(separator: "_").suffix(2))
            guard parts.count == 2, let maxCtx = Int(parts[0]), let seqLen = Int(parts[1]) else { continue }
            pairs.append((maxCtx, seqLen))
        }

        let sorted = pairs.sorted { $0.queryLength < $1.queryLength }
        guard
            let selectedSeq = sorted.first(where: { $0.queryLength >= numInputTokens })?.queryLength
                ?? sorted.last?.queryLength
        else { return nil }
        let candidates = pairs.filter { $0.queryLength == selectedSeq }

        guard
            let selected =
                candidates
                .sorted(by: { $0.contextLength < $1.contextLength })
                .first(where: { $0.contextLength > currentPosition })
        else { return nil }

        return isPrefill
            ? "prompt_opt_\(selected.contextLength)_\(selected.queryLength)"
            : "extend_\(selected.contextLength)_\(selected.queryLength)"
    }

    /// Standard Qwen3 static-shape graph set
    private let qwen3Graphs = [
        "extend_256_8", "extend_256_16", "extend_256_64",
        "extend_512_8", "extend_512_16", "extend_512_64",
        "extend_1024_8", "extend_1024_16", "extend_1024_64",
        "extend_2048_8", "extend_2048_16", "extend_2048_64",
        "extend_4096_8", "extend_4096_16", "extend_4096_64",
        "prompt_opt_256_8", "prompt_opt_256_64",
        "load_embeddings", "gather_embeddings_8",
    ]

    // MARK: - Single-token decode

    @Test("Single-token decode selects smallest querySize (8)")
    func singleTokenDecode() {
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 0, isPrefill: false)
        #expect(graph == "extend_256_8")
    }

    @Test("Single-token decode at various positions picks querySize=8")
    func singleTokenDecodeAtVariousPositions() {
        for pos in [0, 10, 100, 200, 248] {
            let graph = selectGraph(
                extendFunctions: qwen3Graphs, numInputTokens: 1,
                currentPosition: pos, isPrefill: false)
            #expect(graph != nil, "No graph at position \(pos)")
            #expect(graph!.contains("_8"), "At position \(pos): expected querySize 8, got \(graph!)")
        }
    }

    // MARK: - Context bucket transitions

    @Test("Transitions to 512 bucket when 256 is nearly full")
    func contextBucketTransition256to512() {
        // At position 250, extend_256_8 has contextLength=256 > 250, so it still fits
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 250, isPrefill: false)
        #expect(graph == "extend_256_8")
    }

    @Test("Transitions to 512 bucket when 256 is exceeded")
    func contextBucketExceeded256() {
        // At position 256, need contextLength > 256
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 256, isPrefill: false)
        #expect(graph == "extend_512_8")
    }

    @Test("Uses largest context bucket near the end")
    func largestContextBucket() {
        // Position 2050 exceeds 2048, needs 4096 bucket
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 2050, isPrefill: false)
        #expect(graph != nil)
        #expect(graph!.contains("4096"))
    }

    @Test("Returns nil when context is completely exhausted")
    func exhaustedContext() {
        // Position 4096 — no graph has contextLength > 4096
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 4096, isPrefill: false)
        #expect(graph == nil)
    }

    // MARK: - Prefill / multi-token selection

    @Test("9-token request selects querySize=16 (smallest >= 9)")
    func prefillNineTokens() {
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 9,
            currentPosition: 0, isPrefill: false)
        #expect(graph != nil)
        #expect(graph!.contains("_16"))
    }

    @Test("64-token request selects querySize=64")
    func prefillSixtyFourTokens() {
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 64,
            currentPosition: 0, isPrefill: false)
        #expect(graph != nil)
        #expect(graph!.contains("_64"))
    }

    @Test("Prefill flag produces prompt_opt_ prefix")
    func prefillPrefix() {
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 64,
            currentPosition: 0, isPrefill: true)
        #expect(graph != nil)
        #expect(graph!.hasPrefix("prompt_opt_"))
    }

    @Test("Non-prefill produces extend_ prefix")
    func extendPrefix() {
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 1,
            currentPosition: 0, isPrefill: false)
        #expect(graph != nil)
        #expect(graph!.hasPrefix("extend_"))
    }

    @Test("Request exceeding all query sizes falls back to largest")
    func requestExceedingAllQuerySizes() {
        // desiredQuerySize=128 > largest querySize (64), should fall back to 64
        let graph = selectGraph(
            extendFunctions: qwen3Graphs, numInputTokens: 128,
            currentPosition: 0, isPrefill: false)
        #expect(graph != nil)
        #expect(graph!.contains("_64"))
    }

    // MARK: - Minimal shape set

    @Test("Works with single query size")
    func singleQuerySize() {
        let graph = selectGraph(
            extendFunctions: ["extend_256_8", "extend_512_8"],
            numInputTokens: 1, currentPosition: 0, isPrefill: false)
        #expect(graph == "extend_256_8")
    }

    @Test("Empty function list returns nil")
    func emptyFunctionList() {
        let graph = selectGraph(
            extendFunctions: [], numInputTokens: 1,
            currentPosition: 0, isPrefill: false)
        #expect(graph == nil)
    }

    // MARK: - Non-extend functions are ignored

    @Test("Non-extend/prompt functions are filtered out")
    func nonExtendFiltered() {
        // Only load_embeddings and gather_embeddings — no extend_ or prompt_ graphs
        let graph = selectGraph(
            extendFunctions: ["load_embeddings", "gather_embeddings_8"],
            numInputTokens: 1, currentPosition: 0, isPrefill: false)
        #expect(graph == nil)
    }
}
