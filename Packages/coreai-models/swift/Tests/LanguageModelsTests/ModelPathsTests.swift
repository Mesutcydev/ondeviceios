// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("ModelPaths")
struct ModelPathsTests {
    @Test("Default search paths")
    func defaultPaths() {
        let paths = ModelPaths()
        #expect(paths.searchPaths == ModelPaths.defaultSearchPaths)
    }

    @Test("Override via init takes precedence")
    func overridePaths() {
        let paths = ModelPaths(override: "/custom/a:/custom/b")
        #expect(paths.searchPaths == ["/custom/a", "/custom/b"])
    }

    @Test("Absolute path resolves directly if exists")
    func absolutePathExists() throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
            "test_model_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let paths = ModelPaths(override: "/nonexistent")
        let result = paths.resolve(tmp.path)
        #expect(result?.path == tmp.path)
    }

    @Test("Absolute path returns nil if not exists")
    func absolutePathMissing() {
        let paths = ModelPaths(override: "/tmp")
        let result = paths.resolve("/nonexistent/model")
        #expect(result == nil)
    }

    @Test("Resolves name in search directory")
    func resolveByName() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("model_search_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let model = dir.appendingPathComponent("mymodel")
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let paths = ModelPaths(override: dir.path)
        let result = paths.resolve("mymodel")
        #expect(result?.path == model.path)
    }

    @Test("Returns nil when name not found")
    func resolveNotFound() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("model_miss_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let paths = ModelPaths(override: dir.path)
        let result = paths.resolve("nonexistent")
        #expect(result == nil)
    }

    @Test("Search order: first directory wins")
    func searchOrder() throws {
        let dir1 = FileManager.default.temporaryDirectory.appendingPathComponent("dir1_\(UUID().uuidString)")
        let dir2 = FileManager.default.temporaryDirectory.appendingPathComponent("dir2_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir1, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dir2, withIntermediateDirectories: true)
        let model1 = dir1.appendingPathComponent("shared")
        let model2 = dir2.appendingPathComponent("shared")
        try FileManager.default.createDirectory(at: model1, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: model2, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: dir1)
            try? FileManager.default.removeItem(at: dir2)
        }

        let paths = ModelPaths(override: "\(dir1.path):\(dir2.path)")
        let result = paths.resolve("shared")
        #expect(result?.path == model1.path)
    }

    @Test("notFoundError includes search paths")
    func errorMessage() {
        let paths = ModelPaths(override: "/a:/b")
        let error = paths.notFoundError(for: "missing")
        #expect(error.contains("/a"))
        #expect(error.contains("/b"))
        #expect(error.contains("missing"))
    }
}
