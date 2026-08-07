// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Errors from Core AI diffusion component implementations.
public enum CoreAIComponentError: Error, LocalizedError {
    case missingOutput(String, String)
    case invalidShape(String)
    case imageConversionFailed

    public var errorDescription: String? {
        switch self {
        case .missingOutput(let name, let component):
            return "\(component): expected output '\(name)' not found in model results"
        case .invalidShape(let detail):
            return "Invalid tensor shape: \(detail)"
        case .imageConversionFailed:
            return "Failed to convert between CGImage and tensor representation"
        }
    }
}
