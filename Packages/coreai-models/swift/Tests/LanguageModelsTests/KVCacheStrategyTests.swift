// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

@Suite("KVCacheStrategy Resolution", .serialized)
struct KVCacheStrategyResolutionTests {
    // MARK: - Default Size Resolution Tests

    @Test("fixedSize strategy defaults to maxContextLength")
    func fixedSizeDefaultSize() {
        let size = KVCacheStrategy.fixedSize.defaultSize(maxContextLength: 8192)
        #expect(size == 8192)
    }

    @Test("growing strategy defaults to 256")
    func growingDefaultSize() {
        let size = KVCacheStrategy.growing.defaultSize(maxContextLength: 8192)
        #expect(size == 256)
    }

    @Test("auto strategy returns nil (resolved at factory level)")
    func autoDefaultSize() {
        let size = KVCacheStrategy.auto.defaultSize(maxContextLength: 8192)
        #expect(size == nil)
    }

    // MARK: - EngineOptions Size Resolution Tests

    @Test("Explicit size override takes precedence over strategy default")
    func explicitSizeOverride() {
        let options = EngineOptions(kvCacheStrategy: .growing, kvCacheSize: 1024)
        let size = options.resolvedKVCacheSize(maxContextLength: 8192)
        #expect(size == 1024)
    }

    @Test("Auto strategy without override returns nil")
    func autoWithoutOverride() {
        let options = EngineOptions(kvCacheStrategy: .auto, kvCacheSize: nil)
        let size = options.resolvedKVCacheSize(maxContextLength: 8192)
        #expect(size == nil)
    }

    @Test("Auto strategy with explicit size returns that size")
    func autoWithExplicitSize() {
        let options = EngineOptions(kvCacheStrategy: .auto, kvCacheSize: 512)
        let size = options.resolvedKVCacheSize(maxContextLength: 8192)
        #expect(size == 512)
    }
}
