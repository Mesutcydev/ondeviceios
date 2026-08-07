// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("StopSequences")
struct StopSequencesTests {
    @Test("Empty sequences have zero count and maxLength")
    func emptySequences() {
        let stopSequences = StopSequences(sequences: [])

        #expect(stopSequences.isEmpty)
        #expect(stopSequences.count == 0)
        #expect(stopSequences.maxLength == 0)
        #expect(!stopSequences.matches(recentTokens: [1, 2, 3]))
    }

    @Test("maxLength equals longest sequence")
    func maxLengthCalculation() {
        let sequences: [[Int32]] = [[1], [2, 3], [4, 5, 6, 7, 8]]
        let stopSequences = StopSequences(sequences: sequences)

        #expect(stopSequences.count == 3)
        #expect(stopSequences.maxLength == 5)
    }

    @Test("Single-token sequence matching")
    func singleTokenMatching() {
        let stopSequences = StopSequences(sequences: [[123]])

        #expect(stopSequences.matches(recentTokens: [123]))
        #expect(stopSequences.matches(recentTokens: [1, 2, 123]))
        #expect(!stopSequences.matches(recentTokens: [124]))
        #expect(!stopSequences.matches(recentTokens: []))
    }

    @Test("Multi-token sequence requires exact suffix match")
    func multiTokenMatching() {
        let stopSequences = StopSequences(sequences: [[100, 200, 300]])

        #expect(stopSequences.matches(recentTokens: [100, 200, 300]))
        #expect(stopSequences.matches(recentTokens: [50, 100, 200, 300]))
        #expect(!stopSequences.matches(recentTokens: [100, 200]))  // incomplete
        #expect(!stopSequences.matches(recentTokens: [200, 300]))  // missing prefix
    }

    @Test("Multiple sequences - any match succeeds")
    func multipleSequencesMatching() {
        let stopSequences = StopSequences(sequences: [[10], [20, 30], [40, 50, 60]])

        #expect(stopSequences.matches(recentTokens: [1, 2, 10]))
        #expect(stopSequences.matches(recentTokens: [1, 20, 30]))
        #expect(stopSequences.matches(recentTokens: [40, 50, 60]))
        #expect(!stopSequences.matches(recentTokens: [1, 2, 3]))
    }

    @Test("Sliding window usage pattern finds stop sequence")
    func slidingWindowPattern() {
        let sequences = StopSequences(sequences: [[456, 789]])
        var recentTokens: [Int32] = []
        let generatedTokens: [Int32] = [100, 200, 300, 456, 789]

        for (index, token) in generatedTokens.enumerated() {
            recentTokens.append(token)
            if recentTokens.count > sequences.maxLength {
                recentTokens.removeFirst()
            }
            if sequences.matches(recentTokens: recentTokens) {
                #expect(index == 4)
                return
            }
        }
        Issue.record("Stop sequence should have been detected")
    }
}
