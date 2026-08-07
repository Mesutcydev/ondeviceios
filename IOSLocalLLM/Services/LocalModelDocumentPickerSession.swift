import UIKit
import UniformTypeIdentifiers

/// Strongly retained document picker for local model import.
///
/// Why this exists (AltServer works, ForgeSign/sideload often doesn't):
/// 1. SwiftUI representable coordinators are deallocated while Files is up —
///    `delegate` is weak, so Open looks enabled then does nothing. This
///    singleton owns the picker + delegate for the whole presentation.
/// 2. Folder / package / file modes must stay separate — mixing folder UTIs
///    with file UTIs greys out selection on many providers.
/// 3. Folders and directory-bundle packages (**cannot** use `asCopy: true` —
///    `UIDocumentPicker` crashes). Those modes use open-in-place
///    (`asCopy: false`) + security-scoped access. Complete single files
///    (`.gguf`) still use `asCopy: true` for the sideload path.
/// 4. Present from a dedicated `UIWindow` so Menu/SwiftUI teardown cannot
///    dismiss the host VC out from under Files.
/// 5. Hold the security scope across the async import handoff — nilling the
///    picker immediately used to invalidate folder URLs before copy started.
/// 6. This type must stay free of `LocalModelImportService` so the Core AI
///    target can compile it without the MLX/GGUF importer.
@MainActor
final class LocalModelDocumentPickerSession: NSObject, UIDocumentPickerDelegate {
    static let shared = LocalModelDocumentPickerSession()

    enum Kind: Equatable {
        /// Selectable directories (MLX / Core AI resource trees).
        case folder
        /// Directory-bundle packages such as `.aimodel` / `.mlpackage`.
        case package
        /// Single complete model files such as `.gguf`.
        case file
    }

    private var onPick: ((URL) -> Void)?
    private var onCancel: (() -> Void)?
    private var picker: UIDocumentPickerViewController?
    private var hostWindow: UIWindow?
    private var presentationToken = UUID()
    /// Kept alive so folder security-scoped URLs survive until import finishes.
    private var activeSecurityURL: URL?

    private override init() {
        super.init()
    }

    func present(
        kind: Kind,
        onPick: @escaping (URL) async -> Void,
        onCancel: @escaping () -> Void = {}
    ) {
        tearDownPresentation(animated: false)
        releaseSecurityScope()
        // Wrap async importer so the security scope stays open for the
        // whole copy, then release via finishPick.
        self.onPick = { url in
            Task { @MainActor in
                defer { self.finishPick(url) }
                await onPick(url)
            }
        }
        self.onCancel = onCancel

        let picker = makePicker(kind: kind)
        picker.allowsMultipleSelection = false
        picker.shouldShowFileExtensions = true
        picker.delegate = self
        self.picker = picker

        let token = UUID()
        presentationToken = token
        presentWhenReady(picker, token: token, attempt: 0)
    }

    /// UTI lists owned here so Core AI (and other slim targets) do not need
    /// `LocalModelImportService`.
    static func contentTypes(for kind: Kind) -> [UTType] {
        switch kind {
        case .folder:
            // public.folder is the documented directory-pick type; also
            // advertise public.directory so providers that only expose the
            // parent UTI still treat folders as selectable.
            return [.folder, .directory]
        case .package:
            var types: [UTType] = [.package]
            for ext in ["aimodel", "aimodelc", "mlpackage", "mlmodelc"] {
                if let type = UTType(filenameExtension: ext, conformingTo: .package) {
                    types.append(type)
                }
            }
            if let mlmodel = UTType("com.apple.coreml.model") { types.append(mlmodel) }
            if let mlpkg = UTType("com.apple.coreml.mlpackage") { types.append(mlpkg) }
            return types
        case .file:
            // public.item alone can leave opaque model files disabled in the
            // Files picker. Explicitly include public.data for complete
            // byte-stream model files. Never mix these with folder UTIs.
            var types: [UTType] = [.data]
            let dataExtensions = ["gguf", "safetensors", "bin", "onnx", "npz", "aimodel"]
            types.append(contentsOf: dataExtensions.compactMap {
                UTType(filenameExtension: $0, conformingTo: .data)
            })
            return types
        }
    }

    private func makePicker(kind: Kind) -> UIDocumentPickerViewController {
        switch kind {
        case .folder, .package:
            // Directory / package access requires open-in-place. asCopy:true crashes.
            return UIDocumentPickerViewController(
                forOpeningContentTypes: Self.contentTypes(for: kind),
                asCopy: false
            )
        case .file:
            return UIDocumentPickerViewController(
                forOpeningContentTypes: Self.contentTypes(for: kind),
                asCopy: true
            )
        }
    }

    private func presentWhenReady(
        _ picker: UIDocumentPickerViewController,
        token: UUID,
        attempt: Int
    ) {
        // Wait out SwiftUI Menu / popover dismissal before presenting.
        let delay: TimeInterval = attempt == 0 ? 0.45 : 0.12
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self else { return }
            guard self.presentationToken == token, self.picker === picker else { return }
            picker.delegate = self

            guard let scene = Self.activeWindowScene() else {
                if attempt < 20 {
                    self.presentWhenReady(picker, token: token, attempt: attempt + 1)
                } else {
                    self.finishCancel()
                }
                return
            }

            // Dedicated window — do not present from the SwiftUI Menu host.
            let window = UIWindow(windowScene: scene)
            window.windowLevel = .alert + 1
            let host = UIViewController()
            host.view.backgroundColor = .clear
            window.rootViewController = host
            window.makeKeyAndVisible()
            self.hostWindow = window

            if host.presentedViewController != nil
                || host.isBeingDismissed
                || host.isBeingPresented {
                window.isHidden = true
                self.hostWindow = nil
                if attempt < 20 {
                    self.presentWhenReady(picker, token: token, attempt: attempt + 1)
                } else {
                    self.finishCancel()
                }
                return
            }

            host.present(picker, animated: true)
        }
    }

    func documentPicker(
        _ controller: UIDocumentPickerViewController,
        didPickDocumentsAt urls: [URL]
    ) {
        guard let url = urls.first else {
            finishCancel()
            return
        }

        // Hold security scope for open-in-place folder/package picks until the
        // importer finishes. Call sites use `present(... onPick:)` helpers
        // that release the scope via `finishPick(_:)`.
        releaseSecurityScope()
        if url.startAccessingSecurityScopedResource() {
            activeSecurityURL = url
        }

        let pick = onPick
        onPick = nil
        onCancel = nil

        // Keep window/picker alive briefly so the scope stays valid while
        // the importer starts; then tear down UI only.
        pick?(url)

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            self?.tearDownPresentation(animated: true)
        }
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
        finishCancel()
    }

    /// Call when an import started from ``present`` has finished (success or failure).
    func finishPick(_ url: URL) {
        if activeSecurityURL == url {
            releaseSecurityScope()
        }
        tearDownPresentation(animated: false)
    }

    private func finishCancel() {
        let cancel = onCancel
        onPick = nil
        onCancel = nil
        releaseSecurityScope()
        tearDownPresentation(animated: true)
        cancel?()
    }

    private func releaseSecurityScope() {
        if let url = activeSecurityURL {
            url.stopAccessingSecurityScopedResource()
        }
        activeSecurityURL = nil
    }

    private func tearDownPresentation(animated: Bool) {
        picker?.delegate = nil
        if let picker, picker.presentingViewController != nil {
            picker.dismiss(animated: animated)
        }
        picker = nil
        if let window = hostWindow {
            window.isHidden = true
            window.rootViewController = nil
        }
        hostWindow = nil
    }

    private static func activeWindowScene() -> UIWindowScene? {
        let scenes = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
        return scenes.first(where: { $0.activationState == .foregroundActive })
            ?? scenes.first
    }
}

// MARK: - App Documents browser (sideload-proof folder import)

/// Lists model-looking folders already inside the app sandbox (Files /
/// Finder sharing via `UIFileSharingEnabled`). Does not use
/// `UIDocumentPicker`, so it keeps working after common resign/sideload
/// tooling that breaks open-in-place folder picks.
enum LocalModelDocumentsScanner {
    static func candidateModelFolders() -> [URL] {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let skipNames: Set<String> = [
            "HFModels", "LLMModels", "huggingface", "FastVLMModels",
            "VoiceModels", "BundledVoiceModels", "conversations",
            "Inbox", ".Trash", "CoreAIModels"
        ]

        var results: [URL] = []
        func consider(_ dir: URL) {
            if isModelRoot(dir) {
                results.append(dir)
                return
            }
            // One level of nesting (AirDrop / Finder wrappers / ios/).
            guard let kids = try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) else { return }
            for kid in kids {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: kid.path, isDirectory: &isDir),
                      isDir.boolValue else { continue }
                if isModelRoot(kid) {
                    results.append(kid)
                } else if isModelRoot(kid.appendingPathComponent("resources", isDirectory: true)) {
                    results.append(kid.appendingPathComponent("resources", isDirectory: true))
                } else if isModelRoot(kid.appendingPathComponent("ios", isDirectory: true)) {
                    results.append(kid.appendingPathComponent("ios", isDirectory: true))
                }
            }
            // Also accept `<dir>/resources` or `<dir>/ios` directly.
            let resources = dir.appendingPathComponent("resources", isDirectory: true)
            if isModelRoot(resources) { results.append(resources) }
            let ios = dir.appendingPathComponent("ios", isDirectory: true)
            if isModelRoot(ios) { results.append(ios) }
        }

        // Top-level Documents entries
        if let entries = try? fm.contentsOfDirectory(
            at: docs,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: entry.path, isDirectory: &isDir),
                      isDir.boolValue else { continue }
                if skipNames.contains(entry.lastPathComponent) { continue }
                consider(entry)
            }
        }

        // Inbox (Files "Copy to …")
        let inbox = docs.appendingPathComponent("Inbox", isDirectory: true)
        if let entries = try? fm.contentsOfDirectory(
            at: inbox,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for entry in entries {
                var isDir: ObjCBool = false
                if fm.fileExists(atPath: entry.path, isDirectory: &isDir), isDir.boolValue {
                    consider(entry)
                }
            }
        }

        var seen = Set<String>()
        return results
            .filter { seen.insert($0.standardizedFileURL.path).inserted }
            .sorted {
                $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent)
                    == .orderedAscending
            }
    }

    private static func isModelRoot(_ dir: URL) -> Bool {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else {
            return false
        }
        if fm.fileExists(atPath: dir.appendingPathComponent("config.json").path) {
            return true
        }
        if fm.fileExists(atPath: dir.appendingPathComponent("metadata.json").path) {
            return true
        }
        // Any .gguf / .aimodel at this level counts as a complete artifact folder.
        guard let kids = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            return false
        }
        return kids.contains {
            let ext = $0.pathExtension.lowercased()
            return ext == "gguf" || ext == "aimodel" || ext == "aimodelc"
                || ext == "mlpackage" || ext == "mlmodelc"
        }
    }
}
