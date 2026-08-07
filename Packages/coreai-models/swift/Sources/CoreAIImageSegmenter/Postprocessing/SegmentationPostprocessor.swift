// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// Decodes raw `SegmentationOutput` into `Segment` values.
///
/// Scoring matches SAM3's test_sam3.py:
/// ```
/// combined_score = sigmoid(pred_logit) * sigmoid(presence_logit)
/// ```
/// If the output has no presence logits, presence score is treated as 1.0.
public enum SegmentationPostprocessor {
    /// Decode segmentation outputs into a `SegmentationResponse`.
    ///
    /// - Parameters:
    ///   - output: Raw engine outputs (flat Float arrays).
    ///   - inputSize: Original input image size in pixels (used to scale boxes and upsample masks).
    ///   - parameters: Decoding parameters.
    /// - Returns: A `SegmentationResponse` containing up to `parameters.maxSegments` segments
    ///   sorted by score descending, and a `SemanticSegmentationMap` if the model produced one.
    public static func decode(
        output: SegmentationOutput,
        inputSize: CGSize,
        parameters: SegmentationParameters = .default
    ) -> SegmentationResponse {
        let outputHeight = Int(inputSize.height)
        let outputWidth = Int(inputSize.width)

        let shape = output.masksShape
        guard shape.count >= 4 else {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }

        let batchIndex = 0
        let queryCount = shape[1]
        guard queryCount > 0 else {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        let maskHeight = shape[2]
        let maskWidth = shape[3]

        // Defensive bounds check: a malformed engine output (count smaller than the masks
        // shape implies) would crash the indexing below. Bail out with an empty response.
        let useDirectScores = !output.predictedScores.isEmpty
        let pixelsPerQuery = maskHeight * maskWidth
        let querySlotCount = (batchIndex + 1) * queryCount
        guard output.predictedMasks.count >= querySlotCount * pixelsPerQuery else {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if useDirectScores, output.predictedScores.count < querySlotCount {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if !useDirectScores, output.predictedLogits.count < querySlotCount {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if !output.predictedBoxes.isEmpty, output.predictedBoxes.count < querySlotCount * 4 {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }
        if !output.presenceLogits.isEmpty, output.presenceLogits.count <= batchIndex {
            return SegmentationResponse(segments: [], probabilityMap: nil)
        }

        let imageWidth = Double(inputSize.width)
        let imageHeight = Double(inputSize.height)

        let scoredQueries = scoreQueries(
            output: output,
            batchIndex: batchIndex,
            queryCount: queryCount,
            useDirectScores: useDirectScores
        )

        let limit = min(parameters.maxSegments, scoredQueries.count)
        var segments: [Segment] = []
        segments.reserveCapacity(limit)
        for idx in 0..<limit {
            let entry = scoredQueries[idx]
            segments.append(
                decodeSegment(
                    queryIndex: entry.index,
                    score: entry.score,
                    output: output,
                    batchIndex: batchIndex,
                    queryCount: queryCount,
                    maskHeight: maskHeight,
                    maskWidth: maskWidth,
                    outputHeight: outputHeight,
                    outputWidth: outputWidth,
                    imageWidth: imageWidth,
                    imageHeight: imageHeight,
                    parameters: parameters
                ))
        }

        let probabilityMap = decodeProbabilityMap(
            output: output, outputHeight: outputHeight, outputWidth: outputWidth)

        return SegmentationResponse(segments: segments, probabilityMap: probabilityMap)
    }

    /// Compute a final score per query and return them sorted by score descending.
    /// Uses `output.predictedScores` directly when present (e.g. EfficientSAM IOU scores);
    /// otherwise combines `sigmoid(predLogit) * sigmoid(presenceLogit)` (SAM3).
    private static func scoreQueries(
        output: SegmentationOutput,
        batchIndex: Int,
        queryCount: Int,
        useDirectScores: Bool
    ) -> [(score: Float, index: Int)] {
        // Presence score is shared across all queries; treated as 1.0 when absent.
        let presenceScore: Float =
            output.presenceLogits.isEmpty ? 1.0 : sigmoid(output.presenceLogits[batchIndex])

        var scoredQueries: [(score: Float, index: Int)] = []
        scoredQueries.reserveCapacity(queryCount)
        for queryIndex in 0..<queryCount {
            let score: Float
            if useDirectScores {
                score = output.predictedScores[batchIndex * queryCount + queryIndex]
            } else {
                let logit = output.predictedLogits[batchIndex * queryCount + queryIndex]
                score = sigmoid(logit) * presenceScore
            }
            scoredQueries.append((score: score, index: queryIndex))
        }
        scoredQueries.sort { $0.score > $1.score }
        return scoredQueries
    }

    /// Decode the box + mask for a single query into a `Segment`.
    private static func decodeSegment(
        queryIndex: Int,
        score: Float,
        output: SegmentationOutput,
        batchIndex: Int,
        queryCount: Int,
        maskHeight: Int,
        maskWidth: Int,
        outputHeight: Int,
        outputWidth: Int,
        imageWidth: Double,
        imageHeight: Double,
        parameters: SegmentationParameters
    ) -> Segment {
        // Bounding box (XYXY normalized → pixel coordinates).
        // Empty predictedBoxes means the model produced no box output (e.g. EfficientSAM).
        let box: CGRect
        if output.predictedBoxes.isEmpty {
            box = .zero
        } else {
            let boxBase = (batchIndex * queryCount + queryIndex) * 4
            let x0 = Double(output.predictedBoxes[boxBase + 0])
            let y0 = Double(output.predictedBoxes[boxBase + 1])
            let x1 = Double(output.predictedBoxes[boxBase + 2])
            let y1 = Double(output.predictedBoxes[boxBase + 3])

            // AppKit/macOS uses bottom-left origin, so flip Y for macOS.
            // UIKit/iOS uses top-left origin matching the model output directly.
            #if os(macOS)
            box = CGRect(
                x: x0 * imageWidth,
                y: (1.0 - y1) * imageHeight,
                width: (x1 - x0) * imageWidth,
                height: (y1 - y0) * imageHeight
            )
            #else
            box = CGRect(
                x: x0 * imageWidth,
                y: y0 * imageHeight,
                width: (x1 - x0) * imageWidth,
                height: (y1 - y0) * imageHeight
            )
            #endif
        }

        // Mask: sigmoid → nearest-neighbor upsample → threshold.
        let maskBase = (batchIndex * queryCount + queryIndex) * maskHeight * maskWidth
        let lowResMask = output.predictedMasks[maskBase..<(maskBase + maskHeight * maskWidth)].map {
            sigmoid($0)
        }

        let binaryMask = nearestNeighborUpsampleToBool(
            source: lowResMask,
            sourceHeight: maskHeight, sourceWidth: maskWidth,
            destinationHeight: outputHeight, destinationWidth: outputWidth,
            threshold: parameters.maskThreshold
        )

        return Segment(
            mask: binaryMask, maskWidth: outputWidth, maskHeight: outputHeight, box: box, score: score)
    }

    /// Decode the optional semantic-segmentation probability map (sigmoid + upsample to input size).
    private static func decodeProbabilityMap(
        output: SegmentationOutput,
        outputHeight: Int,
        outputWidth: Int
    ) -> SemanticSegmentationMap? {
        let semanticShape = output.semanticSegmentShape
        guard !output.semanticSegment.isEmpty,
            semanticShape.count >= 4,
            semanticShape[2] > 0,
            semanticShape[3] > 0,
            output.semanticSegment.count >= semanticShape[2] * semanticShape[3]
        else {
            return nil
        }
        let segmentHeight = semanticShape[2]
        let segmentWidth = semanticShape[3]
        let probabilityGrid = output.semanticSegment[0..<(segmentHeight * segmentWidth)].map {
            sigmoid($0)
        }
        let probabilities = nearestNeighborUpsampleToFloat(
            source: probabilityGrid,
            sourceHeight: segmentHeight, sourceWidth: segmentWidth,
            destinationHeight: outputHeight, destinationWidth: outputWidth
        )
        return SemanticSegmentationMap(
            probabilities: probabilities, width: outputWidth, height: outputHeight)
    }

    // MARK: - Helpers
    public static func sigmoid(_ x: Float) -> Float {
        1.0 / (1.0 + exp(-x))
    }

    /// Nearest-neighbor upsample a Float grid and threshold to Bool.
    ///
    /// Matches Python: `Image.fromarray(...).resize((W, H), Image.NEAREST)`
    static func nearestNeighborUpsampleToBool(
        source: [Float],
        sourceHeight: Int, sourceWidth: Int,
        destinationHeight: Int, destinationWidth: Int,
        threshold: Float
    ) -> [Bool] {
        var output = [Bool](repeating: false, count: destinationHeight * destinationWidth)
        for outputRow in 0..<destinationHeight {
            let sourceRow = min(sourceHeight - 1, outputRow * sourceHeight / destinationHeight)
            for outputColumn in 0..<destinationWidth {
                let sourceColumn = min(sourceWidth - 1, outputColumn * sourceWidth / destinationWidth)
                output[outputRow * destinationWidth + outputColumn] =
                    source[sourceRow * sourceWidth + sourceColumn] >= threshold
            }
        }
        return output
    }

    /// Nearest-neighbor upsample a Float grid, preserving values.
    static func nearestNeighborUpsampleToFloat(
        source: [Float],
        sourceHeight: Int, sourceWidth: Int,
        destinationHeight: Int, destinationWidth: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: destinationHeight * destinationWidth)
        for outputRow in 0..<destinationHeight {
            let sourceRow = min(sourceHeight - 1, outputRow * sourceHeight / destinationHeight)
            for outputColumn in 0..<destinationWidth {
                let sourceColumn = min(sourceWidth - 1, outputColumn * sourceWidth / destinationWidth)
                output[outputRow * destinationWidth + outputColumn] = source[sourceRow * sourceWidth + sourceColumn]
            }
        }
        return output
    }
}
