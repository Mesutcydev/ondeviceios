// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

#if (arch(arm64) || arch(arm64e)) && canImport(CoreAI)

/// Bare-essentials coverage for `ThinkTagParser`. Pins the four behaviors
/// most likely to regress: full passthrough on non-reasoning models, a
/// complete in-one-chunk block, marker straddling two consumes (the
/// streaming buffer), and the end-of-stream flush path.
@Suite("ThinkTagParser")
struct ThinkTagParserTests {
    @Test("No markers in stream — all input is emitted as .text")
    func passthroughWhenNoMarkers() {
        var parser = ThinkTagParser()
        let events = parser.consume("Hello, world!") + parser.flush()
        #expect(eventStrings(events, kind: .text) == ["Hello, world!"])
        #expect(eventStrings(events, kind: .reasoning).isEmpty)
    }

    @Test("Full block in one consume — text/reasoning/text split correctly")
    func fullBlockInOneConsume() {
        var parser = ThinkTagParser()
        let events = parser.consume("before<think>thoughts</think>after") + parser.flush()
        #expect(eventStrings(events, kind: .text) == ["before", "after"])
        #expect(eventStrings(events, kind: .reasoning) == ["thoughts"])
    }

    @Test("Marker split across consumes — buffer holds back partial match")
    func markerStraddlesTwoConsumes() {
        var parser = ThinkTagParser()
        // First chunk ends in "<thi" — a prefix of the open marker. Parser must
        // hold it (no .text("<thi") leaks) until the next chunk disambiguates.
        var events = parser.consume("before<thi")
        #expect(eventStrings(events, kind: .text) == ["before"])
        events += parser.consume("nk>thoughts</think>after")
        events += parser.flush()
        #expect(eventStrings(events, kind: .text) == ["before", "after"])
        #expect(eventStrings(events, kind: .reasoning) == ["thoughts"])
    }

    @Test("Unclosed <think> at EOS — flush drains held buffer as .reasoning")
    func unclosedThinkAtEndOfStream() {
        var parser = ThinkTagParser()
        let events = parser.consume("<think>unterminated thoughts") + parser.flush()
        #expect(eventStrings(events, kind: .text).isEmpty)
        #expect(eventStrings(events, kind: .reasoning) == ["unterminated thoughts"])
    }

    // MARK: - Helpers

    private enum EventKind { case text, reasoning }

    private func eventStrings(_ events: [ThinkTagParser.Event], kind: EventKind) -> [String] {
        events.compactMap { event in
            switch (event, kind) {
            case (.text(let s), .text): return s
            case (.reasoning(let s), .reasoning): return s
            default: return nil
            }
        }
    }
}

#endif
