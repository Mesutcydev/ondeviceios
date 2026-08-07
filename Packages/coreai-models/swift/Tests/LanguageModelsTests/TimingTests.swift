// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Timing Utilities")
struct TimingTests {
    @Test("Duration.inSeconds and inMilliseconds convert correctly")
    func durationConversion() {
        let duration: Duration = .seconds(1) + .milliseconds(500)
        #expect(duration.inSeconds == 1.5)
        #expect(duration.inMilliseconds == 1500.0)
    }
}
