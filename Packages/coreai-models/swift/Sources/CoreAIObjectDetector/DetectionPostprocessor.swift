// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// Decodes raw `DetectionOutput` into `DetectedObject` values.
///
/// Post-processing matches DETR/YOLOS convention:
/// 1. Softmax over the class dimension (last class is "no-object")
/// 2. For each query, take max probability across object classes
/// 3. Filter by threshold
/// 4. Convert boxes from [cx, cy, w, h] normalized → pixel [x, y, w, h] (top-left origin)
enum DetectionPostprocessor {
    /// Decode detection outputs into a sorted `[DetectedObject]` array.
    ///
    /// Returned `CGRect` values use top-left origin (image coordinate space).
    /// Callers rendering into a CGContext (bottom-left origin on macOS) must flip the y axis.
    ///
    /// - Parameters:
    ///   - output: Raw engine outputs (flat Float arrays).
    ///   - inputSize: Original input image size in pixels (used to scale boxes).
    ///   - parameters: Decoding parameters (threshold, max detections, class labels).
    static func decode(
        output: DetectionOutput,
        inputSize: CGSize,
        parameters: DetectionParameters = .default
    ) -> [DetectedObject] {
        let shape = output.logitsShape
        guard shape.count == 3 else {
            return []
        }

        let queryCount = shape[1]
        let classCount = shape[2]
        guard queryCount > 0, classCount > 1 else {
            return []
        }
        guard output.logits.count == queryCount * classCount,
            output.predictedBoxes.count == queryCount * 4
        else {
            return []
        }

        let imageWidth = Double(inputSize.width)
        let imageHeight = Double(inputSize.height)

        var scored = scoreAndFilter(
            output: output,
            queryCount: queryCount,
            classCount: classCount,
            threshold: parameters.threshold
        )
        scored.sort { $0.score > $1.score }
        let limit = min(parameters.maxDetections, scored.count)

        var detections: [DetectedObject] = []
        detections.reserveCapacity(limit)

        for i in 0..<limit {
            let entry = scored[i]
            let queryIndex = entry.queryIndex

            // Box: [cx, cy, w, h] normalized → pixel CGRect (top-left origin)
            let boxBase = queryIndex * 4
            let centerX = Double(output.predictedBoxes[boxBase + 0])
            let centerY = Double(output.predictedBoxes[boxBase + 1])
            let boxWidth = Double(output.predictedBoxes[boxBase + 2])
            let boxHeight = Double(output.predictedBoxes[boxBase + 3])

            let boundingBox = CGRect(
                x: (centerX - boxWidth / 2.0) * imageWidth,
                y: (centerY - boxHeight / 2.0) * imageHeight,
                width: boxWidth * imageWidth,
                height: boxHeight * imageHeight
            )
            let label = parameters.classLabels[entry.labelIndex] ?? "class_\(entry.labelIndex)"

            detections.append(
                DetectedObject(
                    boundingBox: boundingBox, labelIndex: entry.labelIndex, label: label, confidence: entry.score)
            )
        }

        return detections
    }

    // MARK: - Helpers

    private static func scoreAndFilter(
        output: DetectionOutput,
        queryCount: Int,
        classCount: Int,
        threshold: Float
    ) -> [(score: Float, labelIndex: Int, queryIndex: Int)] {
        var scored: [(score: Float, labelIndex: Int, queryIndex: Int)] = []
        scored.reserveCapacity(queryCount)
        // Last class is the "no-object" background class; exclude it from best-label search for DETR/YOLOS.
        let actualClassCount = classCount - 1

        for queryIndex in 0..<queryCount {
            let probs = softmax(output.logits, offset: queryIndex * classCount, count: classCount)

            var bestScore: Float = 0
            var bestLabel = 0
            for classIndex in 0..<actualClassCount {
                if probs[classIndex] > bestScore {
                    bestScore = probs[classIndex]
                    bestLabel = classIndex
                }
            }

            if bestScore >= threshold {
                scored.append((score: bestScore, labelIndex: bestLabel, queryIndex: queryIndex))
            }
        }

        return scored
    }

    /// Compute softmax over a slice of the array.
    static func softmax(_ array: [Float], offset: Int, count: Int) -> [Float] {
        var maxVal: Float = -.greatestFiniteMagnitude
        for i in 0..<count {
            maxVal = max(maxVal, array[offset + i])
        }

        var expSum: Float = 0
        var exps = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let e = exp(array[offset + i] - maxVal)
            exps[i] = e
            expSum += e
        }

        for i in 0..<count {
            exps[i] /= expSum
        }
        return exps
    }
}
