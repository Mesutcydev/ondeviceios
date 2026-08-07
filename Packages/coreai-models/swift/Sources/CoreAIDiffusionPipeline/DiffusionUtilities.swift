// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreGraphics

/// Shared utilities for diffusion pipeline image processing.
public enum DiffusionUtilities {
    /// Convert CHW float pixel data (range [-1, 1]) to a CGImage.
    ///
    /// Expects pixels laid out as `[R_plane, G_plane, B_plane]` where each
    /// plane has `height * width` float values in [-1, 1].
    public static func pixelsToCGImage(_ pixels: [Float], height: Int, width: Int) throws -> CGImage {
        guard height > 0, width > 0 else {
            throw CoreAIComponentError.invalidShape("Image dimensions must be positive (got \(width)x\(height))")
        }
        guard pixels.count == 3 * height * width else {
            throw CoreAIComponentError.invalidShape("Expected \(3 * height * width) pixels, got \(pixels.count)")
        }

        typealias PlanarF = vImage.PixelBuffer<vImage.PlanarF>
        typealias InterleavedFx3 = vImage.PixelBuffer<vImage.InterleavedFx3>
        typealias Interleaved8x3 = vImage.PixelBuffer<vImage.Interleaved8x3>

        let spatialCount = height * width
        let floatChannels: [PlanarF] = pixels.withUnsafeBufferPointer { pixelsBuf in
            (0..<3).map { c in
                let cOut = PlanarF(width: width, height: height)
                let channelOffset = c * spatialCount
                let cIn = PlanarF(
                    data: .init(mutating: pixelsBuf.baseAddress! + channelOffset),
                    width: width, height: height,
                    byteCountPerRow: width * MemoryLayout<Float>.size)
                cIn.multiply(by: 0.5, preBias: 1.0, postBias: 0.0, destination: cOut)
                return cOut
            }
        }

        let floatImage = InterleavedFx3(planarBuffers: floatChannels)
        let uint8Image = Interleaved8x3(width: width, height: height)
        floatImage.convert(to: uint8Image)

        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        guard
            let format = vImage_CGImageFormat(
                bitsPerComponent: 8, bitsPerPixel: 3 * 8,
                colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: bitmapInfo),
            let cgImage = uint8Image.makeCGImage(cgImageFormat: format)
        else {
            throw CoreAIComponentError.imageConversionFailed
        }
        return cgImage
    }
}
