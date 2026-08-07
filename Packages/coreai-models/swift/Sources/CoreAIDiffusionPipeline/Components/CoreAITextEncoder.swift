// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// Core AI text encoder — wraps a CLIP/T5 model function.
public final class CoreAITextEncoder: Sendable {
    public let function: CoreAIDiffusionModelFunction
    public let tokenize: @Sendable (String) -> [Int32]
    private let maxLength: Int

    public init(
        function: CoreAIDiffusionModelFunction,
        tokenize: @escaping @Sendable (String) -> [Int32],
        maxLength: Int = 77
    ) {
        self.function = function
        self.tokenize = tokenize
        self.maxLength = maxLength
    }

    public func loadResources() async throws {
        try await function.loadResources()
    }

    public func unloadResources() async {
        await function.unloadResources()
    }

    public func encode(_ text: String) async throws -> TextEncoderOutput {
        var ids = tokenize(text)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        } else {
            ids += [Int32](repeating: 0, count: maxLength - ids.count)
        }

        let inputDescs = try await function.inputDescriptors
        guard inputDescs.count == 1, let inputName = inputDescs.keys.first else {
            throw CoreAIComponentError.invalidShape(
                "Text encoder must have exactly one input; got \(inputDescs.count)")
        }

        var inputArray = NDArray(shape: [1, maxLength], scalarType: .int32)
        fillNDArray(&inputArray, as: Int32.self, with: ids)

        let outputs = try await function.predictAllOutputs(inputs: [inputName: inputArray])
        let outputDescs = try await function.outputDescriptors

        // Classify by descriptor rank:
        //   rank 3 → token-level hidden state  [1, seq, hiddenDim]
        //   rank 2 → pooled / projection       [1, pooledDim]
        var hiddenStates: NDArray?
        var pooledOutput: NDArray?
        for (name, floats) in outputs {
            let rank = outputDescs[name]?.shape.count ?? 0
            if rank == 3 {
                guard floats.count.isMultiple(of: maxLength) else {
                    throw CoreAIComponentError.invalidShape(
                        "Text encoder output '\(name)' count \(floats.count) "
                            + "is not a multiple of maxLength \(maxLength)")
                }
                let hiddenDim = floats.count / maxLength
                var array = NDArray(shape: [1, maxLength, hiddenDim], scalarType: .float32)
                fillNDArray(&array, as: Float.self, with: floats)
                hiddenStates = array
            } else if rank == 2 {
                var array = NDArray(shape: [1, floats.count], scalarType: .float32)
                fillNDArray(&array, as: Float.self, with: floats)
                pooledOutput = array
            }
        }

        guard let hiddenStates else {
            throw CoreAIComponentError.invalidShape(
                "Text encoder produced no rank-3 hidden state output")
        }
        return TextEncoderOutput(hiddenStates: hiddenStates, pooledOutput: pooledOutput)
    }
}
