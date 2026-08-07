import Foundation
import UIKit
import UniformTypeIdentifiers
import SwiftUI

// MARK: - LocalModelImportService
// Imports an MLX, Core ML, or GGUF model from the Files app (iCloud Drive, on-device,
// AirDrop staging, etc.) into the app's HFModels directory, then registers it
// with ModelDownloadCenter so the user can pick it from the assistant menu.
//
// Accepted inputs:
//   • A directory (the most common MLX layout — config.json + safetensors)
//   • A complete model file such as .gguf
//   • A Core ML .mlpackage / .mlmodel file

@MainActor
final class LocalModelImportService: ObservableObject {

    static let shared = LocalModelImportService()

    private init() {}

    /// The Files picker must be opened in one document mode at a time.
    /// Mixing \`public.folder\` with file UTIs makes directories navigable in
    /// Files but not selectable on some providers (including exported model
    /// folders). Keep folder selection explicit and use the file mode for
    /// complete GGUF/Core ML artifacts.
    enum ImportKind: Equatable {
        case folder
        case file
    }

    /// Returns the UTType list for the requested document-picker mode.
    static func acceptedTypes(for kind: ImportKind) -> [UTType] {
        // Owned by LocalModelDocumentPickerSession so slim targets (Core AI)
        // can present the picker without compiling this importer.
        // File mode stays file-only: package UTIs + asCopy:true crashes
        // UIDocumentPicker (use Kind.package / folder import instead).
        LocalModelDocumentPickerSession.contentTypes(
            for: kind == .folder ? .folder : .file
        )
    }

    /// Backward-compatible file list for callers that only need model files.
    static var acceptedTypes: [UTType] { acceptedTypes(for: .file) }

    /// Imports a URL chosen from the document picker.
    /// On success, registers the model with ModelDownloadCenter and returns
    /// the registered repo identifier.
    func importModel(from url: URL) async throws -> String {
        // The document picker hands us a security-scoped URL — must call
        // startAccessingSecurityScopedResource before reading.
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        // Destination under Documents/HFModels/local_<name>
        let cleaned = url.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: " ", with: "-")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let hfModelsRoot = docs.appendingPathComponent("HFModels", isDirectory: true)

        // Don't silently clobber an existing import that happens to share
        // the folder name — append a numeric suffix instead
        // (local_foo, local_foo-2, local_foo-3, …).
        var folderName = "local_\(cleaned)"
        var dest = hfModelsRoot.appendingPathComponent(folderName)
        var suffix = 2
        while FileManager.default.fileExists(atPath: dest.path) {
            folderName = "local_\(cleaned)-\(suffix)"
            dest = hfModelsRoot.appendingPathComponent(folderName)
            suffix += 1
        }

        // Free-space gate before copying: importing a multi-GB folder with
        // no room used to fail halfway with an opaque copy error.
        let importBytes: Int64 = {
            var srcIsDir: ObjCBool = false
            FileManager.default.fileExists(atPath: url.path, isDirectory: &srcIsDir)
            if srcIsDir.boolValue {
                return (try? FileManager.default.allocatedSizeOfDirectory(at: url)) ?? 0
            }
            return ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
        }()
        if importBytes > 0, let free = HFModelDownloadManager.freeDiskBytes(),
           free < importBytes + 200_000_000 {
            throw NSError(domain: "LocalImport", code: -3, userInfo: [
                NSLocalizedDescriptionKey:
                    "Not enough disk space to import “\(cleaned)”. Need ~\((importBytes + 200_000_000).formattedBytes) free, only \(free.formattedBytes) available."
            ])
        }

        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)

        // Branch on input type
        var isDir: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir)

        if url.pathExtension.lowercased() == "zip" {
            // Unzip into dest
            try await unzip(url, to: dest)
        } else {
            // A multi-GB GGUF copy can take minutes. FileManager.copyItem is
            // synchronous; running it on this @MainActor service froze the UI
            // and made the app appear dead throughout the import.
            let source = url
            let destination = dest
            let sourceIsDirectory = isDir.boolValue
            try await Task.detached(priority: .userInitiated) {
                let fm = FileManager.default
                if sourceIsDirectory {
                    try fm.copyItem(at: source, to: destination)
                } else {
                    try fm.createDirectory(at: destination, withIntermediateDirectories: true)
                    do {
                        try fm.copyItem(
                            at: source,
                            to: destination.appendingPathComponent(source.lastPathComponent)
                        )
                    } catch {
                        try? fm.removeItem(at: destination)
                        throw error
                    }
                }
            }.value
        }

        // Imported weights are user-restorable from their original source —
        // keep them out of iCloud backups like every downloaded model.
        FileManager.excludeFromBackup(dest)

        // Users often share a model as a folder that contains the actual
        // model directory one level down (e.g. AirDrop / zip re-wrapping adds
        // an enclosing folder). Hoist the real model root up so config.json /
        // the GGUF pair sit where the loader and the readiness check expect.
        normalizeModelLayout(at: dest)

        let repoID = "local/\(folderName)"

        // Persist the repoID so ModelDownloadCenter.scanCustomDownloads
        // recovers the SAME id on the next launch instead of deriving a
        // different one from the folder name (the folder→repoID split is
        // lossy). Without this the imported model came back under a mismatched
        // id after a restart.
        let sidecar = dest.appendingPathComponent(".repoID")
        try? repoID.data(using: .utf8)?.write(to: sidecar, options: [.atomic])

        let downloader = HFModelDownloadManager(
            repoID: repoID, destination: dest
        )
        downloader.checkIfReady()

        // Fail loudly when the folder doesn't actually contain a usable model.
        // Previously import always reported success and registered the entry
        // even when no weights were recognized — so the user saw "Imported"
        // but the model never appeared in Installed (which filters on
        // `isReady`). Roll back the copied bytes so a half-recognized folder
        // doesn't linger as dead weight.
        guard downloader.state == .ready else {
            try? FileManager.default.removeItem(at: dest)
            throw NSError(domain: "LocalImport", code: -2, userInfo: [
                NSLocalizedDescriptionKey:
                    "Couldn't find a usable model in “\(cleaned)”. Import an MLX folder containing config.json plus .safetensors weights, a standalone text .gguf, or a vision GGUF paired with its mmproj-*.gguf file."
            ])
        }

        // Detect the real category from the files on disk (config.json's
        // architecture / file layout) instead of assuming .assistant — that
        // assumption hid imported VLMs from the vision picker.
        let category = LocalModelRegistry.category(in: dest)

        let runtime: ModelRuntime? = LocalModelFileValidator.hasValidGGUFTextModel(in: dest)
            ? .llamaCpp
            : nil

        ModelDownloadCenter.shared.registerCustom(
            repoID: repoID,
            displayName: cleaned,
            subtitle: "local · imported from Files",
            category: category,
            sizeLabel: dirSize(at: dest).formattedBytes,
            docURL: nil,
            downloader: downloader,
            runtime: runtime
        )

        ToastCenter.shared.success("Imported \(cleaned)",
                                    detail: "Available in the model picker.")
        return repoID
    }

    // MARK: - Helpers

    private func dirSize(at url: URL) -> Int64 {
        (try? FileManager.default.allocatedSizeOfDirectory(at: url)) ?? 0
    }

    /// If `dest` isn't itself a model root but contains one nested up to two
    /// levels down, replaces `dest`'s contents with that nested root so the
    /// model files sit at the top level.
    private func normalizeModelLayout(at dest: URL) {
        let fm = FileManager.default
        if isModelRoot(dest) { return }
        guard let found = findModelRoot(under: dest, maxDepth: 2), found != dest else { return }

        // Move the discovered root aside, then swap it in for `dest`.
        let tmp = dest.deletingLastPathComponent()
            .appendingPathComponent("__import_tmp_\(dest.lastPathComponent)")
        try? fm.removeItem(at: tmp)
        do {
            try fm.moveItem(at: found, to: tmp)
            try fm.removeItem(at: dest)
            try fm.moveItem(at: tmp, to: dest)
        } catch {
            try? fm.removeItem(at: tmp)
            print("[LocalImport] normalizeModelLayout failed: \(error)")
        }
    }

    /// A directory is a model root if it holds an MLX `config.json`, a
    /// standalone text GGUF, or a complete GGUF VLM pair. Mirrors
    /// HFModelDownloadManager's readiness check.
    private func isModelRoot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) { return true }
        return Self.hasGGUFPair(in: dir)
            || LocalModelFileValidator.hasValidGGUFTextModel(in: dir)
    }

    private func findModelRoot(under root: URL, maxDepth: Int) -> URL? {
        if isModelRoot(root) { return root }
        guard maxDepth > 0 else { return nil }
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ) else { return nil }
        for entry in entries {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: entry.path, isDirectory: &isDir), isDir.boolValue {
                if let found = findModelRoot(under: entry, maxDepth: maxDepth - 1) {
                    return found
                }
            }
        }
        return nil
    }

    private static func hasGGUFPair(in dir: URL) -> Bool {
        LocalModelFileValidator.hasCompleteGGUFVLMPair(in: dir)
    }

    /// Unzip via Foundation's Process bridge isn't available on iOS, so we use
    /// NSFileCoordinator + the system "Archive Utility" style approach via the
    /// `Compression` framework. For .zip with a simple flat structure this
    /// works; complex zips should be pre-extracted on macOS.
    private func unzip(_ archive: URL, to dest: URL) async throws {
        // iOS doesn't expose a simple zip API without a third-party lib.
        // Throw WITHOUT creating the destination — the old code mkdir'd `dest`
        // first and then threw, orphaning an empty folder. (Also unreachable
        // now that .zip is no longer an accepted type.)
        throw NSError(domain: "LocalImport", code: -1, userInfo: [
            NSLocalizedDescriptionKey:
                "Zip imports aren't supported on iOS yet. Please extract on macOS first, then re-import the folder."
        ])
    }
}

// MARK: - Document picker presentation
// Use ``LocalModelDocumentPickerSession.shared`` — a retained singleton owns
// the picker + delegate so SwiftUI teardown cannot drop Open callbacks.

extension LocalModelDocumentPickerSession.Kind {
    init(_ kind: LocalModelImportService.ImportKind) {
        self = kind == .folder ? .folder : .file
    }
}

extension LocalModelDocumentPickerSession {
    /// Presents the picker and keeps the security-scoped URL valid until
    /// `onPick` finishes (including async work).
    func present(
        importKind: LocalModelImportService.ImportKind,
        onPick: @escaping (URL) async -> Void,
        onCancel: @escaping () -> Void = {}
    ) {
        present(kind: Kind(importKind), onPick: onPick, onCancel: onCancel)
    }
}

// MARK: - LocalModelExportPicker
// Hands an installed model directory to the system document picker. The picker
// performs the copy directly to the user's chosen Files location, so the app
// does not create a second multi-gigabyte staging copy inside its sandbox.

struct LocalModelExportPicker: UIViewControllerRepresentable {

    let modelDirectory: URL
    let onComplete: () -> Void
    let onCancel: () -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(
            forExporting: [modelDirectory],
            asCopy: true
        )
        picker.shouldShowFileExtensions = true
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIDocumentPickerViewController,
        context: Context
    ) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onComplete: onComplete, onCancel: onCancel)
    }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onComplete: () -> Void
        let onCancel: () -> Void

        init(onComplete: @escaping () -> Void, onCancel: @escaping () -> Void) {
            self.onComplete = onComplete
            self.onCancel = onCancel
        }

        func documentPicker(
            _ controller: UIDocumentPickerViewController,
            didPickDocumentsAt urls: [URL]
        ) {
            onComplete()
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            onCancel()
        }
    }
}
