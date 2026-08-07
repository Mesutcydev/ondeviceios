import Foundation

struct CoreAIModelCapabilities: Codable, Equatable, Sendable {
    let textGeneration: Bool
    let streaming: Bool
    let multiTurn: Bool
    let imageInput: Bool
    let guidedGeneration: Bool
    let toolCalling: Bool
    let reasoning: Bool
    let concurrentExecution: Bool
}

struct CoreAIExpectedModelFile: Codable, Equatable, Sendable {
    let relativePath: String
    let byteCount: Int64?
    let sha256: String?
}

struct CoreAIModelManifest: Codable, Equatable, Sendable {
    let id: String
    let displayName: String
    let version: String
    let modelFamily: String
    let assetPackID: String
    let expectedFiles: [CoreAIExpectedModelFile]
    let totalDownloadBytes: Int64
    let minimumOSVersion: String
    let supportedDeviceFamilies: [String]
    let supportedArchitectures: [String]
    let contextWindow: Int
    let maximumOutputTokens: Int
    let capabilities: CoreAIModelCapabilities
    let tokenizerIdentifier: String?
    let licenseNotice: String
}

/// Installation boundary for Core AI asset packs. A pack may contain one or
/// more `.aimodel` resources, tokenizers, metadata, and AOT artifacts.
@MainActor
final class CoreAIModelStore: ObservableObject {
    static let shared = CoreAIModelStore()
    nonisolated static let defaultModelID = "coreai-qwen3-0.6b"
    private static let installedVersionKey = "CoreAI.installedVersion"

    enum State: Equatable {
        case unavailable(String)
        case missing
        case downloading
        case validating
        case ready(URL, CoreAIModelManifest)
        case failed(String)
    }

    @Published private(set) var state: State = .missing
    private var downloadTask: Task<Void, Never>?

    var modelDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("CoreAIModels", isDirectory: true)
    }

    var installationURL: URL? {
        guard case .ready(let url, _) = state else { return nil }
        return url
    }

    var modelResourcesURL: URL? {
        installationURL?.appendingPathComponent("resources", isDirectory: true)
    }

    var manifest: CoreAIModelManifest? {
        guard case .ready(_, let manifest) = state else { return nil }
        return manifest
    }

    private init() { refresh() }

    func refresh() {
        guard #available(iOS 27.0, *) else {
            state = .unavailable("Core AI requires iOS 27 or later.")
            return
        }
        guard let root = try? installedRoot(),
              let manifest = try? loadAndValidateManifest(at: root) else {
            state = .missing
            return
        }
        state = .ready(root, manifest)
    }

    func importModel(from sourceURL: URL) throws {
        try importModel(from: sourceURL, preferredID: nil, preferredDisplayName: nil)
    }

    func importModel(
        from sourceURL: URL,
        preferredID: String?,
        preferredDisplayName: String?
    ) throws {
        state = .validating
        do {
            try performImport(
                from: sourceURL,
                preferredID: preferredID,
                preferredDisplayName: preferredDisplayName
            )
        } catch {
            refresh()
            // Preserve a previously installed pack; otherwise surface the error.
            if case .ready = state {} else {
                state = .failed(error.localizedDescription)
            }
            throw error
        }
    }

    private func performImport(
        from sourceURL: URL,
        preferredID: String?,
        preferredDisplayName: String?
    ) throws {
        let staging = modelDirectory.appendingPathComponent(".staging-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        let resources = staging.appendingPathComponent("resources", isDirectory: true)
        let isDirectory = (try? sourceURL.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        if isDirectory {
            // Users often pick the Hub checkout root (`…/ios`, or a wrapper
            // that still contains `resources/`). Copy the resolved pack root
            // so metadata.json + .aimodel land directly under `resources/`.
            let packRoot = Self.resolvePackRoot(from: sourceURL)
            try FileManager.default.copyItem(at: packRoot, to: resources)
        } else {
            let ext = sourceURL.pathExtension.lowercased()
            guard ext == "aimodel" || ext == "aimodelc" else {
                throw CoreAIModelStoreError.invalidExtension
            }
            try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
            try FileManager.default.copyItem(
                at: sourceURL,
                to: resources.appendingPathComponent(sourceURL.lastPathComponent)
            )
        }

        let manifest = try loadOrCreateManifest(
            in: resources,
            preferredID: preferredID,
            preferredDisplayName: preferredDisplayName
        )
        try validate(manifest: manifest, resourcesAt: resources)
        let destination = modelDirectory.appendingPathComponent(manifest.version, isDirectory: true)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
        UserDefaults.standard.set(manifest.version, forKey: Self.installedVersionKey)
        state = .ready(destination, manifest)
    }

    /// Prefer the directory that actually contains Core AI resources.
    /// Accepts a bare resource tree, `resources/`, or `ios/`.
    private static func resolvePackRoot(from sourceURL: URL) -> URL {
        let fm = FileManager.default
        func hasModelFiles(_ url: URL) -> Bool {
            !((try? collectModelFiles(in: url)) ?? []).isEmpty
        }
        if fm.fileExists(atPath: sourceURL.appendingPathComponent("metadata.json").path) {
            return sourceURL
        }
        if hasModelFiles(sourceURL) {
            return sourceURL
        }
        let resources = sourceURL.appendingPathComponent("resources", isDirectory: true)
        if fm.fileExists(atPath: resources.appendingPathComponent("metadata.json").path)
            || hasModelFiles(resources) {
            return resources
        }
        let ios = sourceURL.appendingPathComponent("ios", isDirectory: true)
        if fm.fileExists(atPath: ios.appendingPathComponent("metadata.json").path)
            || hasModelFiles(ios) {
            return ios
        }
        // Nested Hub layouts: `<root>/<something>/resources`.
        if let kids = try? fm.contentsOfDirectory(
            at: sourceURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for kid in kids {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: kid.path, isDirectory: &isDir), isDir.boolValue else {
                    continue
                }
                if fm.fileExists(atPath: kid.appendingPathComponent("metadata.json").path)
                    || hasModelFiles(kid) {
                    return kid
                }
                let nested = kid.appendingPathComponent("resources", isDirectory: true)
                if fm.fileExists(atPath: nested.appendingPathComponent("metadata.json").path)
                    || hasModelFiles(nested) {
                    return nested
                }
            }
        }
        return sourceURL
    }

    func download(from remoteURL: URL) {
        downloadTask?.cancel()
        state = .downloading
        downloadTask = Task { [weak self] in
            do {
                let (temporaryURL, response) = try await URLSession.shared.download(from: remoteURL)
                guard let http = response as? HTTPURLResponse, 200..<300 ~= http.statusCode else {
                    throw CoreAIModelStoreError.downloadFailed
                }
                try await MainActor.run { [weak self] in
                    guard let self else { return }
                    try self.importModel(from: temporaryURL)
                }
            } catch is CancellationError {
                await MainActor.run { [weak self] in self?.refresh() }
            } catch {
                await MainActor.run { [weak self] in self?.state = .failed("Core AI model installation failed.") }
            }
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        refresh()
    }

    func removeModel() throws {
        if let root = installationURL {
            try FileManager.default.removeItem(at: root)
        }
        UserDefaults.standard.removeObject(forKey: Self.installedVersionKey)
        refresh()
    }

    private func installedRoot() throws -> URL {
        guard let version = UserDefaults.standard.string(forKey: Self.installedVersionKey) else {
            throw CoreAIModelStoreError.notInstalled
        }
        let root = modelDirectory.appendingPathComponent(version, isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.path) else { throw CoreAIModelStoreError.notInstalled }
        return root
    }

    private func loadAndValidateManifest(at root: URL) throws -> CoreAIModelManifest {
        let manifestURL = root.appendingPathComponent("resources/manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(CoreAIModelManifest.self, from: data)
        try validate(manifest: manifest, resourcesAt: root.appendingPathComponent("resources"))
        return manifest
    }

    private func loadOrCreateManifest(
        in resources: URL,
        preferredID: String? = nil,
        preferredDisplayName: String? = nil
    ) throws -> CoreAIModelManifest {
        let manifestURL = resources.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            return try JSONDecoder().decode(CoreAIModelManifest.self, from: Data(contentsOf: manifestURL))
        }
        let modelFiles = try Self.collectModelFiles(in: resources)
        guard !modelFiles.isEmpty else { throw CoreAIModelStoreError.missingModelResource }
        let relative = modelFiles.map { file -> String in
            let full = file.standardizedFileURL.path
            let root = resources.standardizedFileURL.path
            if full.hasPrefix(root + "/") {
                return String(full.dropFirst(root.count + 1))
            }
            return file.lastPathComponent
        }
        let totalBytes = modelFiles.reduce(Int64(0)) { partial, url in
            partial + Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        let manifest = CoreAIModelManifest(
            id: preferredID ?? Self.defaultModelID,
            displayName: preferredDisplayName ?? "Core AI model",
            version: UUID().uuidString,
            modelFamily: preferredDisplayName ?? "custom",
            assetPackID: preferredID ?? Self.defaultModelID,
            expectedFiles: relative.map {
                CoreAIExpectedModelFile(relativePath: $0, byteCount: nil, sha256: nil)
            },
            totalDownloadBytes: totalBytes,
            minimumOSVersion: "27.0",
            supportedDeviceFamilies: ["iPhone"],
            supportedArchitectures: ["arm64"],
            contextWindow: 32_768,
            maximumOutputTokens: 2_048,
            capabilities: CoreAIModelCapabilities(
                textGeneration: true,
                streaming: true,
                multiTurn: true,
                imageInput: false,
                guidedGeneration: false,
                toolCalling: false,
                reasoning: false,
                concurrentExecution: false
            ),
            tokenizerIdentifier: nil,
            licenseNotice: "Review the model license before distribution."
        )
        try JSONEncoder().encode(manifest).write(to: manifestURL, options: .atomic)
        return manifest
    }

    private static func collectModelFiles(in root: URL) throws -> [URL] {
        var results: [URL] = []
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }
        for case let url as URL in enumerator {
            let ext = url.pathExtension.lowercased()
            let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            // Official packs ship `.aimodel` as a directory bundle (main.mlirb +
            // metadata) rather than a single file.
            if ext == "aimodel" || ext == "aimodelc" {
                results.append(url)
                if isDirectory {
                    enumerator.skipDescendants()
                }
                continue
            }
            if !isDirectory, url.lastPathComponent.lowercased() == "main.mlirb" {
                results.append(url.deletingLastPathComponent())
                enumerator.skipDescendants()
            }
        }
        // De-dupe directory vs main.mlirb discoveries.
        var seen = Set<String>()
        return results.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    private func validate(manifest: CoreAIModelManifest, resourcesAt root: URL) throws {
        guard manifest.minimumOSVersion.hasPrefix("27"),
              manifest.supportedDeviceFamilies.contains(where: {
                  $0.caseInsensitiveCompare("iPhone") == .orderedSame
                      || $0.caseInsensitiveCompare("iphone") == .orderedSame
              }) || manifest.supportedDeviceFamilies.isEmpty else {
            throw CoreAIModelStoreError.unsupportedModel
        }
        // Prefer Apple metadata.json when present — it is the runtime contract.
        let metadata = root.appendingPathComponent("metadata.json")
        let hasMetadata = FileManager.default.fileExists(atPath: metadata.path)
        let hasTokenizer = FileManager.default.fileExists(
            atPath: root.appendingPathComponent("tokenizer/tokenizer.json").path
        )
        let hasModelFile = !(try Self.collectModelFiles(in: root)).isEmpty
        if hasMetadata {
            guard hasModelFile else { throw CoreAIModelStoreError.missingModelResource }
            return
        }
        let rootPath = root.standardizedFileURL.path
        for expected in manifest.expectedFiles {
            let candidate = root.appendingPathComponent(expected.relativePath).standardizedFileURL
            guard candidate.path == rootPath || candidate.path.hasPrefix(rootPath + "/"),
                  FileManager.default.fileExists(atPath: candidate.path),
                  (try? candidate.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
                throw CoreAIModelStoreError.missingModelResource
            }
            if let byteCount = expected.byteCount,
               (try? candidate.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) != byteCount {
                throw CoreAIModelStoreError.invalidManifest
            }
        }
        guard hasModelFile || !manifest.expectedFiles.isEmpty else {
            throw CoreAIModelStoreError.missingModelResource
        }
        _ = hasTokenizer
    }
}

enum CoreAIModelStoreError: LocalizedError {
    case downloadFailed, invalidExtension, notInstalled, missingModelResource, invalidManifest, unsupportedModel

    var errorDescription: String? {
        switch self {
        case .downloadFailed: return "The Core AI model server returned an invalid response."
        case .invalidExtension: return "Core AI models must use the .aimodel format."
        case .notInstalled: return "No Core AI model is installed."
        case .missingModelResource: return "The Core AI asset pack is missing a required model resource."
        case .invalidManifest: return "The Core AI model manifest is invalid."
        case .unsupportedModel: return "This Core AI model is not supported by this iPhone build."
        }
    }
}
