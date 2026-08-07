// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

/// Tests for EngineOptions.
@Suite("EngineOptions")
struct EngineOptionsTests {
    @Test("EngineOptions initialization")
    func engineOptionsInit() {
        // Defaults
        let defaults = EngineOptions()
        #expect(defaults.variant == nil)
        #expect(defaults.kvCacheStrategy == .auto)

        // Custom variant
        let custom = EngineOptions(variant: "pipelined")
        #expect(custom.variant == "pipelined")

        // Custom KV cache strategy
        let withStrategy = EngineOptions(variant: "sequential", kvCacheStrategy: .growing)
        #expect(withStrategy.variant == "sequential")
        #expect(withStrategy.kvCacheStrategy == .growing)
    }
}
