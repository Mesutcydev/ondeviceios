// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate

/// Compute weighted sum of Float arrays of equal length using BLAS.
func weightedSum(_ weights: [Float], _ values: [[Float]]) -> [Float] {
    precondition(!values.isEmpty && weights.count == values.count)
    let count = values[0].count
    assert(values.allSatisfy { $0.count == count })
    var result = [Float](repeating: 0.0, count: count)
    for i in 0..<values.count {
        let w = weights[i]
        values[i].withUnsafeBufferPointer { buf in
            cblas_saxpy(Int32(count), w, buf.baseAddress, 1, &result, 1)
        }
    }
    return result
}

/// Double-precision weights overload (DPM-Solver uses Double internally).
func weightedSum(_ weights: [Double], _ values: [[Float]]) -> [Float] {
    weightedSum(weights.map(Float.init), values)
}

/// Evenly spaced floats between [start, end].
func linspace(_ start: Float, _ end: Float, _ count: Int) -> [Float] {
    guard count > 1 else { return count == 1 ? [start] : [] }
    let scale = (end - start) / Float(count - 1)
    return (0..<count).map { Float($0) * scale + start }
}

extension Array {
    subscript(back index: Int) -> Element {
        self[count - index]
    }
}
