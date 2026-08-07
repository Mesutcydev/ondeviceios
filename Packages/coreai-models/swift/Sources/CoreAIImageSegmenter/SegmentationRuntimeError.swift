// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Runtime errors thrown by the segmentation pipeline.
///
/// All cases carry a human-readable description via `LocalizedError.errorDescription`.
public enum SegmentationRuntimeError: Error, LocalizedError, Sendable {
    /// The model asset could not be loaded or compiled (e.g. corrupt file, missing weights).
    case modelLoadFailed(String)
    /// A required output tensor was absent from the executed graph.
    case outputMissing(String)
    /// The requested engine type is not available on this platform or build configuration.
    case unsupportedEngine(String)
    /// The engine has not been initialized before `segment()` was called.
    case notLoaded
    /// A `SegmentationParameters` value or model tensor layout is incompatible with this engine.
    case invalidConfiguration(String)
    /// The model bundle directory does not exist or is not a directory.
    case bundleNotFound(String)
    /// The bundle directory contains no `.aimodel` file.
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .outputMissing(let name):
            return "Expected output tensor missing: \(name)"
        case .unsupportedEngine(let type):
            return "Unsupported engine type: \(type)"
        case .notLoaded:
            return "Engine not loaded"
        case .invalidConfiguration(let reason):
            return "Invalid configuration: \(reason)"
        case .bundleNotFound(let path):
            return "Bundle not found: \(path)"
        case .modelNotFound(let path):
            return "No .aimodel found in bundle: \(path)"
        }
    }
}
