// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

// MARK: - Test Constants

enum TestConstants {
    static let openBraceToken: Int32 = 0
    static let closeBraceToken: Int32 = 1
    static let quoteToken: Int32 = 6
    static let maxGenerationSteps = 200
    static let toyVocabSize = 39
}

// MARK: - Shared Test Data

let sharedToyVocab = [
    // 0-6: Structural tokens
    "{", "}", "[", "]", ":", ",", "\"",
    // 7-9: Whitespace
    " ", "\n", "\t",
    // 10-20: Words
    "name", "age", "email", "city", "true", "false", "null",
    "John", "Jane", "Doe", "Smith",
    // 21-23: Roles
    "user", "admin", "guest",
    // 24-25: Status words
    "status", "active",
    // 26-35: Digits
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    // 36-38: Multi-digit numbers
    "10", "20", "25",
]

let sharedPersonSchema = """
    {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
      },
      "required": ["name", "age"],
      "additionalProperties": false
    }
    """

// MARK: - Test Helpers

func createTestSession(
    schema: String = sharedPersonSchema,
    vocabulary: [String] = sharedToyVocab,
    vocabType: VocabularyType = .raw
) throws -> ConstrainedGenerationSession {
    try ConstrainedGenerationSession(
        jsonSchema: schema,
        vocabulary: vocabulary,
        vocabType: vocabType
    )
}

/// Decode the bitmask from a session into a set of allowed token IDs.
func allowedTokenIDs(session: inout ConstrainedGenerationSession) -> Set<Int32> {
    guard let bitmask = session.nextTokenBitmask() else { return Set() }
    var allowed = Set<Int32>()
    for tokenId in 0..<session.vocabularySize {
        let (wordIndex, bitIndex) = tokenId.quotientAndRemainder(dividingBy: 32)
        if wordIndex < bitmask.count && (bitmask[wordIndex] & (1 << bitIndex)) != 0 {
            allowed.insert(Int32(tokenId))
        }
    }
    return allowed
}

/// Drive a session to completion by picking tokens greedily with structural priority.
@discardableResult
func driveToCompletion(
    session: inout ConstrainedGenerationSession, maxSteps: Int = TestConstants.maxGenerationSteps
) -> String {
    let structuralChars: Set<String> = ["}", "]", "\"", ":", ",", "{", "["]
    let digitChars: Set<String> = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    var text = ""

    for _ in 0..<maxSteps {
        if session.isTerminated { break }

        let allowed = allowedTokenIDs(session: &session)
        guard !allowed.isEmpty else { break }

        // Pick token by priority
        var selectedToken: Int32 = allowed.first!
        var bestPriority = Int.max

        for tokenId in allowed {
            let tokenStr = sharedToyVocab[Int(tokenId)]
            let priority: Int
            if tokenStr == "\"" {
                // Quote gets highest priority: closes open strings before any content character
                priority = 0
            } else if tokenStr == "}" || tokenStr == "]" {
                // Closing braces terminate objects/arrays
                priority = 1
            } else if structuralChars.contains(tokenStr) {
                priority = 2
            } else if digitChars.contains(tokenStr) {
                priority = 3
            } else if tokenStr.count <= 2 {
                priority = 4
            } else {
                priority = 4
            }

            if priority < bestPriority {
                bestPriority = priority
                selectedToken = tokenId
            }
        }

        if !session.acceptToken(selectedToken) { break }
        text += sharedToyVocab[Int(selectedToken)]
    }

    return text
}

/// Validate that a string is a valid JSON schema and return the parsed dictionary.
func validateJSONSchema(_ schema: String, file: String = #file, line: Int = #line) throws -> [String: Any] {
    guard let data = schema.data(using: .utf8),
        let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
        Issue.record("Invalid JSON schema: \(schema)")
        return [:]
    }
    return obj
}
