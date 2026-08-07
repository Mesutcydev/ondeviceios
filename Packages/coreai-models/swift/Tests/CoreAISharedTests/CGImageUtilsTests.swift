// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import TestUtilities
import Testing

@testable import CoreAIShared

@Suite("CGImageUtils")
struct CGImageUtilsTests {
    // MARK: - toNormalizedPlanarRGB

    @Test("White image normalizes to 1.0")
    func rgbFloatsWhite() throws {
        let image = try #require(makeSolidCGImage(r: 255, g: 255, b: 255, side: 4))
        let floats = try CGImageUtils.toNormalizedPlanarRGB(image)
        #expect(floats.count == 3 * 4 * 4)
        #expect(floats.allSatisfy { abs($0 - 1.0) < 1e-3 })
    }

    @Test("Black image normalizes to -1.0")
    func rgbFloatsBlack() throws {
        let image = try #require(makeSolidCGImage(r: 0, g: 0, b: 0, side: 4))
        let floats = try CGImageUtils.toNormalizedPlanarRGB(image)
        #expect(floats.allSatisfy { abs($0 + 1.0) < 1e-3 })
    }

    @Test("Red image has correct planar NCHW layout")
    func rgbFloatsPlanarLayout() throws {
        let side = 4
        let image = try #require(makeSolidCGImage(r: 255, g: 0, b: 0, side: side))
        let floats = try CGImageUtils.toNormalizedPlanarRGB(image)
        let pixelCount = side * side
        let rChannel = floats[0..<pixelCount]
        let gChannel = floats[pixelCount..<(2 * pixelCount)]
        let bChannel = floats[(2 * pixelCount)..<(3 * pixelCount)]
        #expect(rChannel.allSatisfy { $0 > 0.9 })
        #expect(gChannel.allSatisfy { $0 < -0.9 })
        #expect(bChannel.allSatisfy { $0 < -0.9 })
    }

    // MARK: - resize

    @Test("resize: output dimensions match requested side")
    func resizeDimensions() {
        let image = makeSolidCGImage(r: 128, g: 64, b: 32, side: 16)!
        let resized = CGImageUtils.resize(image, to: 8)!
        #expect(resized.width == 8)
        #expect(resized.height == 8)
    }
}
