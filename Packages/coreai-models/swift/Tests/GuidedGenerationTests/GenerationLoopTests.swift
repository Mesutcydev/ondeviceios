// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

@Suite("GenerationLoop")
struct GenerationLoopTests {
    @Test func tokenizerBridgeExtractTokenizerInfo() throws {
        let info = TokenizerInfo(
            vocabulary: sharedToyVocab,
            vocabType: .raw
        )
        // Should be usable to create a session
        var session = try ConstrainedGenerationSession(
            jsonSchema: sharedPersonSchema,
            tokenizerInfo: info
        )
        let isTerminated = session.isTerminated
        #expect(!isTerminated)
        let allowed = allowedTokenIDs(session: &session)
        #expect(allowed.contains(TestConstants.openBraceToken))
    }
}
