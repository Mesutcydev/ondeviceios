import Foundation
import Combine
import CryptoKit

extension Notification.Name {
    static let hfModelDownloadCompleted = Notification.Name("hfModelDownloadCompleted")
    static let hfModelDownloadStarted = Notification.Name("hfModelDownloadStarted")
    static let hfModelDownloadPaused = Notification.Name("hfModelDownloadPaused")
}

// MARK: - HFModelDownloadManager
// Generic resumable downloader for any Hugging Face repository.
//
// Usage:
//   let mgr = HFModelDownloadManager(
//       repoID:      "mlx-community/llava-fastvithd_0.5b_stage3_llm.fp16",
//       destination: URL(fileURLWithPath: "…/FastVLMModels/llava-fastvithd_0.5b_stage3_llm.fp16")
//   )
//   await mgr.start()
//
// File enumeration:
//   1. Uses the HF file-tree API to list every blob in the repo.
//   2. Skips files that are already fully present at the destination.
//   3. Interrupted transfers resume via URLSession resume data that
//      BackgroundDownloadCoordinator persists per source URL (no HTTP
//      Range handling of our own — the background session owns it).
//
// Thread safety: all @Published mutations happen on MainActor.

@MainActor
final class HFModelDownloadManager: ObservableObject, Identifiable {

    // MARK: - Identity / config

    let id: String              // unique key, typically the repoID
    let repoID: String
    let branch: String
    let destination: URL
    /// If non-nil, only download files whose **path** matches the allowlist.
    /// Each entry is matched as a prefix (e.g. `"kitten_tts_nano.mlpackage/"`)
    /// OR exact equality (e.g. `"voices.npz"`).
    var fileAllowlist: [String]?

    // MARK: - Published state

    @Published private(set) var state: DownloadState = .idle
    @Published private(set) var progress: Double = 0           // 0–1
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var currentFile: String = ""
    @Published private(set) var filesDone: Int = 0
    @Published private(set) var filesTotal: Int = 0

    /// Structured category for the most-recent failure. Cleared when
    /// a new run starts. Catalog UI branches on this to decide whether
    /// to show a generic Retry button or the auth-specific "Set Token"
    /// CTA. .none means either no failure or a non-categorized one.
    @Published private(set) var lastFailureKind: FailureKind = .none

    enum FailureKind: Equatable {
        case none
        case generic
        case tokenRequired    // 401/403 with no token sent
        case tokenRejected    // 401/403 with a token sent (invalid / no access)
    }

    enum DownloadState: Equatable {
        case idle
        case enumerating            // fetching file list from HF API
        case downloading
        case paused                 // user paused or the connection stalled
        case ready                  // all files present
        case failed(String)

        var isActive: Bool {
            self == .enumerating || self == .downloading
        }

        var isPaused: Bool {
            self == .paused
        }

        static func == (lhs: DownloadState, rhs: DownloadState) -> Bool {
            switch (lhs, rhs) {
            case (.idle, .idle), (.enumerating, .enumerating),
                 (.downloading, .downloading), (.paused, .paused),
                 (.ready, .ready): return true
            case (.failed(let a), .failed(let b)): return a == b
            default: return false
            }
        }
    }

    // MARK: - Private

    private var downloadTask: Task<Void, Never>?
    /// A cancelled run can still deliver one final URLSession callback. State
    /// changes from an old run must not overwrite a newer resume attempt.
    private var activeRunID: UUID?
    private var pauseRequested = false
    private var lastProgressLogAt: Date?
    private var lastProgressLogBytes: Int64 = 0
    private var completedDownloadBytes: Int64 = 0
    private var activeFileBytes: [String: Int64] = [:]

    // MARK: - Init

    init(
        repoID: String,
        destination: URL,
        branch: String = "main",
        fileAllowlist: [String]? = nil
    ) {
        self.id          = repoID
        self.repoID      = repoID
        self.branch      = branch
        self.destination = destination
        self.fileAllowlist = fileAllowlist
    }

    // MARK: - Public API

    func start() {
        guard !state.isActive else { return }
        if state == .ready {
            checkIfReady()
            guard state != .ready else { return }
        }
        let wasPaused = state == .paused
        let runID = UUID()
        activeRunID = runID
        pauseRequested = false
        downloadTask = Task { [weak self] in
            guard let self else { return }
            await self.run(runID: runID)
        }
        RuntimeLogCenter.emit(
            wasPaused ? "Resuming \(repoID) from saved progress" : "Starting download for \(repoID)",
            subsystem: "download"
        )
        NotificationCenter.default.post(
            name: .hfModelDownloadStarted,
            object: self,
            userInfo: ["repoID": repoID]
        )
    }

    /// Pauses the in-flight download without deleting completed files. The
    /// background coordinator asks URLSession for resume data, so a later
    /// `start()` continues the current file instead of starting over.
    func pause() {
        guard state.isActive else { return }
        pauseRequested = true
        RuntimeLogCenter.emit("Pausing \(repoID) — keeping downloaded files", subsystem: "download")
        downloadTask?.cancel()
        Task { @MainActor [weak self] in
            guard let self else { return }
            await BackgroundDownloadCoordinator.shared.cancelTasks(
                matching: self.destination.path,
                producingResumeData: true
            )
            guard self.activeRunID != nil else { return }
            if self.state.isActive {
                self.state = .paused
            }
        }
    }

    /// Compatibility spelling for older download-center surfaces. A cancel
    /// button now behaves like pause so a user can continue safely later.
    func cancel() { pause() }

    /// Permanently abandons the transfer and removes the files owned by this
    /// downloader. Deletion paths call this indirectly through `delete()`.
    func abandon() {
        cancelDownloadTasks()
        removeTrackedFiles()
        reset()
    }

    /// Cancels the in-flight download task(s) WITHOUT touching files on disk.
    /// Used by delete()/redownload(), which do their own file handling.
    private func cancelDownloadTasks() {
        activeRunID = nil
        pauseRequested = false
        downloadTask?.cancel()
        downloadTask = nil
        // Also cancel any in-flight URLSession download tasks
        Task { @MainActor in
            await BackgroundDownloadCoordinator.shared.cancelTasks(
                matching: destination.path,
                producingResumeData: false
            )
        }
        if state.isActive || state == .paused { state = .idle }
    }

    func delete() throws {
        cancelDownloadTasks()
        // Allowlisted downloads share their destination root with sibling
        // variants (KittenTTS Nano/Mini share VoiceModels/KittenTTS; the two
        // Whisper models share HFModels/ggerganov_whisper.cpp). Removing the
        // whole directory wiped the sibling too — only remove tracked paths,
        // mirroring redownload().
        if fileAllowlist != nil {
            removeTrackedFiles()
        } else if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        reset()
    }

    /// Clears the files this downloader owns and starts a fresh fetch.
    /// Used when the bytes on disk exist but failed a higher-level
    /// validation step (for example a voice bundle whose files are present
    /// but cannot be parsed). For allowlisted downloads we only remove the
    /// tracked paths so sibling variants sharing the same destination root
    /// survive.
    func redownload() {
        cancelDownloadTasks()
        removeTrackedFiles()
        reset()
        start()
    }

    func checkIfReady() {
        // Critical: don't override state when a download is actively running.
        // refreshAllStates() fires on every foreground transition, and without
        // this guard a partially-completed download would flip to .ready as
        // soon as the user backgrounds and returns to the app.
        if state.isActive || state == .paused { return }

        let fm = FileManager.default
        if let allowlist = fileAllowlist {
            // For each rule, ensure at least one matching file exists.
            // Directory rules (ending in '/') require the directory to be non-empty.
            let allPresent = allowlist.allSatisfy { rule in
                let url = destination.appendingPathComponent(rule)
                if rule.hasSuffix("/") {
                    var isDir: ObjCBool = false
                    guard fm.fileExists(atPath: url.path, isDirectory: &isDir),
                          isDir.boolValue,
                          let contents = try? fm.contentsOfDirectory(atPath: url.path),
                          !contents.isEmpty else { return false }
                    // Compiled CoreML bundles always carry coremldata.bin at
                    // their root — a non-empty but partially-downloaded
                    // .mlmodelc must not count as ready.
                    if rule.hasSuffix(".mlmodelc/") {
                        return fm.fileExists(
                            atPath: url.appendingPathComponent("coremldata.bin").path)
                    }
                    return true
                }
                guard fm.fileExists(atPath: url.path),
                      let attrs = try? fm.attributesOfItem(atPath: url.path),
                      let size = attrs[.size] as? Int64 else { return false }
                return size > 0
            }
            state = allPresent ? .ready : .idle
        } else if fm.fileExists(atPath: destination.path) {
            // For repos without an allowlist we expect a marker file.
            //   • MLX layout has `config.json` at the root — primary signal.
            //   • GGUF VLM layout has no config.json; instead it ships a
            //     pair: `<model>.gguf` + `mmproj-<model>.gguf`. Without
            //     this branch the bundled SmolVLM2-500M GGUF (copied into
            //     HFModels/ on first launch by BundledVLMInstaller) was
            //     scanned but rejected as not-ready, hiding the
            //     pre-installed model from Models → Installed and pushing
            //     users to re-download it.
            //   • Bare partial downloads with just a stray .tmp still
            //     don't trigger a false ready.
            state = Self.looksReady(in: destination) ? .ready : .idle
        }
    }

    /// Heuristic for "this directory holds a loadable model". Recognizes more
    /// valid shapes than a bare `config.json` check so legitimately-complete
    /// imports/downloads aren't rejected:
    ///   • `config.json`            — MLX / HF transformers layout
    ///   • `model_index.json`       — diffusers layout
    ///   • a complete GGUF pair     — llama.cpp VLM
    ///   • real weights + a tokenizer/processor companion — covers repos that
    ///     ship `tokenizer_config.json` but no top-level `config.json`
    /// Requiring an actual weights file present (not just JSON) keeps a
    /// partially-downloaded repo from flipping to a false "ready".
    static func looksReady(in dir: URL) -> Bool {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return false }
        // Sharded safetensors: when the index file is present, require EVERY
        // shard it references. config.json lands long before the multi-GB
        // shards finish, so the bare config.json check below marked partial
        // sharded downloads as installed.
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: indexURL.path),
           let data = try? Data(contentsOf: indexURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMap = json["weight_map"] as? [String: String] {
            let shards = Set(weightMap.values)
            return !shards.isEmpty && shards.allSatisfy { shard in
                isNonEmptyFile(dir.appendingPathComponent(shard))
            }
        }
        // Text GGUF files embed their tokenizer and metadata, so a single
        // valid non-mmproj file is a complete model. Keep this in sync with
        // the reference loader's recursive validator so exported/nested
        // model packages are not misclassified.
        if hasGGUFPair(in: dir) || LocalModelFileValidator.hasValidGGUFTextModel(in: dir) {
            return true
        }
        let hasWeights = names.contains { name in
            let lower = name.lowercased()
            guard lower.hasSuffix(".safetensors") || lower.hasSuffix(".gguf")
                    || lower.hasSuffix(".bin") || lower.hasSuffix(".npz")
            else { return false }
            if lower.hasSuffix(".gguf") {
                return LocalModelFileValidator.isValidGGUFFile(dir.appendingPathComponent(name))
            }
            return isNonEmptyFile(dir.appendingPathComponent(name))
        }
        let hasCompanion = names.contains { n in
            n == "config.json" || n == "model_index.json"
                || n == "tokenizer_config.json" || n == "tokenizer.json"
                || n == "preprocessor_config.json" || n == "generation_config.json"
        }
        return hasWeights && hasCompanion
    }

    private static func isNonEmptyFile(_ url: URL) -> Bool {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else { return false }
        return size.int64Value > 0
    }

    /// True when `dir` contains a complete GGUF VLM pair: at least one
    /// `<model>.gguf` (LLM weights) AND at least one `mmproj-*.gguf`
    /// (vision projector). Mirrors LlamaCppVLMService.hasGGUFPair so
    /// the readiness check agrees with the load path.
    private static func hasGGUFPair(in dir: URL) -> Bool {
        LocalModelFileValidator.hasCompleteGGUFVLMPair(in: dir)
    }

    // MARK: - Core download loop

    private func run(runID: UUID) async {
        state = .enumerating
        lastFailureKind = .none

        do {
            // 1. List files from HF API (retried — the tree endpoint is
            //    rate-limited and 429s/5xxs under load like the LFS hosts).
            let allFiles = try await withRetries { try await self.fetchFileList() }
            let files: [HFFileMeta]
            if let allowlist = fileAllowlist {
                files = allFiles.filter { f in
                    allowlist.contains { rule in
                        // Prefix match for directory rules ending in '/'
                        if rule.hasSuffix("/") { return f.path.hasPrefix(rule) }
                        // Equality match for plain file names
                        return f.path == rule || f.name == rule
                    }
                }
            } else {
                files = allFiles
            }

            guard !files.isEmpty else {
                state = .failed("No matching files found in repo \(repoID). Check the repo path or allowlist.")
                return
            }

            guard isCurrent(runID), !Task.isCancelled else { throw CancellationError() }

            // 2. Filter out already-complete files
            let pending = files.filter { !isComplete($0) }

            if pending.isEmpty {
                guard isCurrent(runID) else { return }
                state = .ready
                filesDone = files.count
                filesTotal = files.count
                progress = 1.0
                RuntimeLogCenter.emit("Already complete: \(repoID)", subsystem: "download")
                finishRun(runID)
                return
            }

            // 3. Compute total size. Files with unknown size (still 0 after
            //    HEAD) just don't contribute to the byte total — progress will
            //    still tick by file count.
            totalBytes  = files.map { max(0, $0.size) }.reduce(0, +)
            filesTotal  = files.count
            filesDone   = files.count - pending.count
            completedDownloadBytes = alreadyDownloadedBytes(files)
            downloadedBytes = completedDownloadBytes
            activeFileBytes.removeAll(keepingCapacity: true)

            // Disk space pre-check: refuse if not enough room. Headroom
            // scales with the download — max(200 MB, 5% of pending bytes) —
            // so a 40 GB fetch doesn't squeak by on a 200 MB margin.
            let pendingBytes = pending.map { max(0, $0.size) }.reduce(0, +)
            if pendingBytes >= 1_000_000_000 {
                ToastCenter.shared.info(
                    "Large model download",
                    detail: "About \(pendingBytes.formattedBytes) will be stored on this device. Use Models → Storage Cleanup anytime to keep only the models you need."
                )
            }
            if pendingBytes > 0, let free = Self.freeDiskBytes() {
                let headroom: Int64 = max(200_000_000, pendingBytes / 20)
                if free < pendingBytes {
                    // Hard-fail ONLY when the raw bytes can't possibly fit.
                    // freeDiskBytes() (VolumeAvailableCapacityForImportantUsage)
                    // is an optimistic figure, so a strict free < bytes+headroom
                    // gate could falsely block a download that would actually
                    // succeed. Below this floor it genuinely can't fit.
                    let need  = pendingBytes.formattedBytes
                    let avail = free.formattedBytes
                    state = .failed(
                        "Not enough disk space. Need ~\(need) free, only \(avail) available. " +
                        "Delete other models or apps and retry."
                    )
                    ToastCenter.shared.error("Download cancelled — disk full",
                                              detail: "need \(need), have \(avail)")
                    return
                } else if free < pendingBytes + headroom {
                    // Tight but plausible — warn and let the authoritative
                    // per-file write errors gate it, rather than refusing here.
                    ToastCenter.shared.info("Low disk space",
                                            detail: "This download may fail if space runs out.")
                }
            }

            state = .downloading
            lastProgressLogAt = nil
            lastProgressLogBytes = downloadedBytes
            RuntimeLogCenter.emit(
                "\(repoID): \(files.count) files, \(totalBytes.formattedBytes) total; \(filesDone) already present",
                subsystem: "download"
            )

            // Start a Dynamic Island / Lock Screen Live Activity for this download
            _ = DownloadLiveActivityManager.shared.start(repoID: repoID)

            // 4. Create destination directory
            try FileManager.default.createDirectory(
                at: destination, withIntermediateDirectories: true)
            // Model weights are large and re-downloadable — keep them out
            // of iCloud/iTunes backups. Directory-level exclusion covers
            // every file written beneath it.
            FileManager.excludeFromBackup(destination)

            // Drop a sidecar carrying the original repoID. The folder name is
            // a lossy `/` → `_` substitution and can't be reversed for repos
            // whose author or name contains an underscore (e.g.
            // `author/name_with_underscores`). ModelDownloadCenter reads this
            // file on cold start to recover the exact id.
            let sidecar = destination.appendingPathComponent(".repoID")
            try? repoID.data(using: .utf8)?
                .write(to: sidecar, options: [.atomic])

            // 5. Download files one at a time. URLSession writes each transfer
            // to a temporary file before moving it into the model directory.
            // Keeping one transfer in flight matches the stable reference
            // path and avoids a second large temporary allocation exactly when
            // the last weight shard is being finalized.
            for meta in pending {
                guard isCurrent(runID), !Task.isCancelled else { throw CancellationError() }
                let dest = destination.appendingPathComponent(meta.path)
                let parent = dest.deletingLastPathComponent()
                try FileManager.default.createDirectory(
                    at: parent, withIntermediateDirectories: true)
                activeFileBytes[meta.path] = 0
                RuntimeLogCenter.emit(
                    "\(repoID): downloading \(meta.path)",
                    subsystem: "download"
                )
                updateAggregateProgress()

                _ = try await downloadFile(meta: meta, to: dest, runID: runID)
                guard isCurrent(runID), !Task.isCancelled else { throw CancellationError() }

                // Resync from disk after URLSession finalizes the file. This
                // keeps progress correct when a task resumes from a saved
                // checkpoint or a server reports an unknown content length.
                let fileBytes: Int64 = {
                    guard let attributes = try? FileManager.default
                            .attributesOfItem(atPath: dest.path),
                          let size = attributes[.size] as? NSNumber else {
                        return max(0, meta.size)
                    }
                    return size.int64Value
                }()
                completedDownloadBytes += max(0, fileBytes)
                activeFileBytes.removeValue(forKey: meta.path)
                filesDone += 1
                updateAggregateProgress()

                DownloadLiveActivityManager.shared.update(
                    repoID: repoID,
                    progress: progress,
                    downloadedBytes: downloadedBytes,
                    totalBytes: totalBytes,
                    currentFile: meta.name,
                    filesDone: filesDone,
                    filesTotal: filesTotal
                )
                RuntimeLogCenter.emit(
                    "\(repoID): finished \(meta.name) (\(filesDone)/\(filesTotal))",
                    subsystem: "download"
                )
            }

            guard isCurrent(runID) else { return }
            RuntimeLogCenter.emit(
                "\(repoID): finalizing downloaded files",
                subsystem: "download"
            )
            guard fileAllowlist != nil || Self.looksReady(in: destination) else {
                throw NSError(
                    domain: "HFDownload",
                    code: -8,
                    userInfo: [NSLocalizedDescriptionKey:
                        "The download finished, but the model files did not pass the completeness check. Resume or retry to repair the missing shard."
                    ]
                )
            }
            state = .ready
            progress = 1.0
            currentFile = ""
            DownloadLiveActivityManager.shared.finish(repoID: repoID)
            ToastCenter.shared.success("Downloaded \(repoID)")
            NotificationCenter.default.post(
                name: .hfModelDownloadCompleted,
                object: self,
                userInfo: ["repoID": repoID]
            )
            RuntimeLogCenter.emit("Download complete: \(repoID)", subsystem: "download")
            finishRun(runID)

        } catch is CancellationError {
            guard isCurrent(runID) else { return }
            activeFileBytes.removeAll(keepingCapacity: true)
            updateAggregateProgress()
            if pauseRequested {
                state = .paused
                RuntimeLogCenter.emit(
                    "Paused \(repoID) at \(downloadedBytes.formattedBytes)",
                    subsystem: "download"
                )
                NotificationCenter.default.post(
                    name: .hfModelDownloadPaused,
                    object: self,
                    userInfo: ["repoID": repoID]
                )
            } else {
                state = .idle
            }
            finishRun(runID)
        } catch let error where Self.isUserCancellation(error) {
            guard isCurrent(runID) else { return }
            activeFileBytes.removeAll(keepingCapacity: true)
            updateAggregateProgress()
            if pauseRequested {
                state = .paused
                RuntimeLogCenter.emit(
                    "Paused \(repoID) at \(downloadedBytes.formattedBytes)",
                    subsystem: "download"
                )
            } else {
                state = .idle
            }
            finishRun(runID)
        } catch let authErr as HFAuthError {
            // Auth-specific failure: tag the kind so the catalog UI
            // can offer "Set Token" inline instead of a generic Retry.
            // Skip the failure toast in this case — the inline CTA
            // is a better surface than a transient toast for an
            // actionable error.
            switch authErr {
            case .tokenRequired: lastFailureKind = .tokenRequired
            case .tokenRejected: lastFailureKind = .tokenRejected
            }
            state = .failed(authErr.errorDescription ?? "Authentication required")
            DownloadLiveActivityManager.shared.fail(
                repoID: repoID,
                reason: authErr.errorDescription ?? "Authentication required"
            )
            RuntimeLogCenter.emit(
                "\(repoID): authentication required",
                level: .warning,
                subsystem: "download"
            )
            finishRun(runID)
        } catch {
            guard isCurrent(runID) else { return }
            lastFailureKind = .generic
            if Self.isRetryable(error) {
                state = .paused
                DownloadLiveActivityManager.shared.fail(
                    repoID: repoID,
                    reason: "Connection interrupted; resume to continue"
                )
                ToastCenter.shared.info(
                    "Download paused",
                    detail: "\(repoID) can resume from the saved checkpoint."
                )
                RuntimeLogCenter.emit(
                    "\(repoID): paused after connection interruption — \(error.localizedDescription)",
                    level: .warning,
                    subsystem: "download"
                )
            } else {
                state = .failed(error.localizedDescription)
                DownloadLiveActivityManager.shared.fail(
                    repoID: repoID,
                    reason: error.localizedDescription
                )
                ToastCenter.shared.error("Download failed: \(repoID)",
                                          detail: error.localizedDescription)
                RuntimeLogCenter.emit(
                    "\(repoID): failed — \(error.localizedDescription)",
                    level: .error,
                    subsystem: "download"
                )
            }
            finishRun(runID)
        }
    }

    // MARK: - HF file-tree API

    private struct HFFileMeta: Sendable {
        let path: String    // e.g. "model.safetensors" or "subfolder/file.bin"
        let name: String    // last path component
        let size: Int64
        /// LFS sha256 from the tree API (`lfs.oid`), when the file is an
        /// LFS blob. Nil for plain git blobs and fallback enumerations —
        /// integrity verification is skipped in that case.
        let sha256: String?

        init(path: String, name: String, size: Int64, sha256: String? = nil) {
            self.path = path
            self.name = name
            self.size = size
            self.sha256 = sha256
        }
    }

    private func fetchFileList() async throws -> [HFFileMeta] {
        // HF tree API with recursive=true to enumerate subdirectories
        // (mlpackage repos have files nested under <model>.mlpackage/...).
        // Also append &expand=true for richer metadata where supported.
        let urlString = "https://huggingface.co/api/models/\(repoID)/tree/\(branch)?recursive=true&expand=true"
        guard let url = URL(string: urlString) else { throw URLError(.badURL) }

        var request = URLRequest(url: url)
        request.setValue("ios-local-llm/1.0 (iOS)", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 30
        HFTokenStore.authorize(&request)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw NSError(domain: "HFDownload", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "No HTTP response from huggingface.co"
            ])
        }

        // Diagnostic — log the URL + first 200 chars of body on error
        if http.statusCode != 200 {
            let body = String(data: data.prefix(400), encoding: .utf8) ?? "<binary>"
            print("[HFDownload] \(http.statusCode) for \(urlString)\n  body: \(body)")
        }

        guard http.statusCode == 200 else {
            // Auth-specific paths surface as a structured `HFAuthError`
            // so the catalog UI can offer a "Set Token" CTA instead of
            // a generic failure toast. The distinction:
            //   • token absent + 401  → .tokenRequired (asks for token)
            //   • token present + 401 → .tokenRejected (token invalid/no access)
            //   • 403                 → .tokenRejected (terms-acceptance gated)
            if http.statusCode == 401 || http.statusCode == 403 {
                let tokenPresent = HFTokenStore.authorizationHeaderValue() != nil
                if tokenPresent {
                    throw HFAuthError.tokenRejected(repoID: repoID)
                } else {
                    throw HFAuthError.tokenRequired(repoID: repoID)
                }
            }
            let reason: String
            switch http.statusCode {
            case 404: reason = "Repo \(repoID) not found (404). Verify it exists at huggingface.co/\(repoID)"
            case 429: reason = "Rate limited by huggingface.co (429). Wait a minute and retry."
            case 500...599: reason = "huggingface.co server error (\(http.statusCode)). Retry shortly."
            default:  reason = "huggingface.co returned HTTP \(http.statusCode) for \(repoID)"
            }
            throw NSError(domain: "HFDownload", code: http.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: reason])
        }

        guard let raw = try? JSONSerialization.jsonObject(with: data) else {
            let body = String(data: data.prefix(400), encoding: .utf8) ?? "<binary>"
            print("[HFDownload] JSON parse failed for \(urlString)\n  body: \(body)")
            throw NSError(domain: "HFDownload", code: -2, userInfo: [
                NSLocalizedDescriptionKey: "Could not parse file tree from huggingface.co"
            ])
        }

        // Tree API can return either a top-level array OR a {tree: [...]} dict
        // depending on endpoint version. Handle both.
        let json: [[String: Any]]
        if let arr = raw as? [[String: Any]] {
            json = arr
        } else if let dict = raw as? [String: Any], let tree = dict["tree"] as? [[String: Any]] {
            json = tree
        } else if let dict = raw as? [String: Any], let siblings = dict["siblings"] as? [[String: Any]] {
            json = siblings    // /api/models/{repo} returns "siblings" with {rfilename}
        } else {
            let preview = String(data: data.prefix(200), encoding: .utf8) ?? ""
            print("[HFDownload] Unexpected JSON shape from \(urlString)\n  preview: \(preview)")
            throw NSError(domain: "HFDownload", code: -3, userInfo: [
                NSLocalizedDescriptionKey: "Unexpected response shape from huggingface.co"
            ])
        }

        var blobs = json.compactMap { item -> HFFileMeta? in
            // HF returns:
            //   tree API     → {path, type:"file"|"directory", size, lfs?}
            //   models API   → {rfilename, size?}
            let path = (item["path"] as? String) ?? (item["rfilename"] as? String)
            guard let path else { return nil }
            // Reject directories (type-based or trailing slash)
            if let type_ = item["type"] as? String, type_ == "directory" || type_ == "tree" {
                return nil
            }
            if path.hasSuffix("/") { return nil }

            let size: Int64 = {
                if let n = item["size"] as? NSNumber { return n.int64Value }
                if let n = item["size"] as? Int      { return Int64(n) }
                if let lfs = item["lfs"] as? [String: Any] {
                    if let n = lfs["size"] as? NSNumber { return n.int64Value }
                    if let n = lfs["size"] as? Int      { return Int64(n) }
                }
                return 0
            }()
            // LFS blobs carry their sha256 as `lfs.oid` — captured so the
            // download can be integrity-checked after it completes.
            let oid = (item["lfs"] as? [String: Any])?["oid"] as? String
            let name = URL(fileURLWithPath: path).lastPathComponent
            return HFFileMeta(path: path, name: name, size: size, sha256: oid)
        }

        // Diagnostic — log what we found
        print("[HFDownload] \(repoID): tree returned \(json.count) entries, \(blobs.count) files")

        // For any blob still reporting size 0, resolve real size via HEAD so
        // progress + isComplete checks work correctly. Probed concurrently
        // (mirrors probeStandardMLXFiles) — the previous sequential loop
        // serialized one round-trip per zero-size blob.
        let zeroSizeIndices = blobs.indices.filter { blobs[$0].size <= 0 }
        if !zeroSizeIndices.isEmpty {
            let resolved = await withTaskGroup(of: (Int, Int64?).self) { group in
                for i in zeroSizeIndices {
                    let path = blobs[i].path
                    group.addTask { [weak self] in
                        (i, await self?.headSize(for: path))
                    }
                }
                var sizes: [Int: Int64] = [:]
                for await (i, size) in group where (size ?? 0) > 0 {
                    sizes[i] = size
                }
                return sizes
            }
            for (i, real) in resolved {
                blobs[i] = HFFileMeta(path: blobs[i].path,
                                      name: blobs[i].name,
                                      size: real,
                                      sha256: blobs[i].sha256)
            }
        }

        // Fallback A: try /api/models/{repo} siblings endpoint
        if blobs.isEmpty {
            print("[HFDownload] Tree returned 0 files, trying siblings endpoint")
            if let sib = try? await fetchFileListViaSiblings(), !sib.isEmpty {
                return sib
            }
        }

        // Fallback B: probe a standard MLX-model file list via HEAD requests.
        // If the user is downloading e.g. `mlx-community/Qwen3-4B-Instruct-...`
        // we expect a known set of files. We HEAD each, keep the ones that
        // exist. This works even when both tree + siblings endpoints fail.
        if blobs.isEmpty {
            print("[HFDownload] Siblings empty, probing standard MLX file list")
            let probed = await probeStandardMLXFiles()
            if !probed.isEmpty { return probed }
        }

        return blobs
    }

    /// HEAD-probe a standard MLX/CoreML repo layout. Keeps every file that
    /// actually exists (200/206 response).
    private func probeStandardMLXFiles() async -> [HFFileMeta] {
        // Common files across MLX-quantised LLM repos + LLaVA-style VLM repos
        let candidates: [String] = [
            // Config / tokenizer
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
            "preprocessor_config.json",
            "generation_config.json",
            "chat_template.jinja",
            // Single safetensors
            "model.safetensors",
            // Sharded safetensors (probe up to 8 shards for big models)
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ]
        var results: [HFFileMeta] = []
        await withTaskGroup(of: HFFileMeta?.self) { group in
            for path in candidates {
                group.addTask { [weak self] in
                    guard let self else { return nil }
                    guard let size = await self.headSize(for: path), size > 0 else { return nil }
                    return HFFileMeta(
                        path: path,
                        name: URL(fileURLWithPath: path).lastPathComponent,
                        size: size
                    )
                }
            }
            for await result in group {
                if let r = result { results.append(r) }
            }
        }
        print("[HFDownload] Standard-file probe found \(results.count) files in \(repoID)")
        return results
    }

    /// Fallback file enumeration via the /api/models/{repo} endpoint.
    /// Returns less metadata (sizes via HEAD only) but works when tree fails.
    private func fetchFileListViaSiblings() async throws -> [HFFileMeta] {
        let urlString = "https://huggingface.co/api/models/\(repoID)"
        guard let url = URL(string: urlString) else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.setValue("ios-local-llm/1.0 (iOS)", forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        HFTokenStore.authorize(&request)
        let (data, _) = try await URLSession.shared.data(for: request)
        guard let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let siblings = dict["siblings"] as? [[String: Any]] else { return [] }
        var blobs: [HFFileMeta] = []
        for s in siblings {
            guard let fname = s["rfilename"] as? String else { continue }
            let size = await headSize(for: fname) ?? 0
            blobs.append(HFFileMeta(
                path: fname,
                name: URL(fileURLWithPath: fname).lastPathComponent,
                size: size
            ))
        }
        print("[HFDownload] \(repoID): siblings returned \(blobs.count) files")
        return blobs
    }

    /// HEAD request to resolve a single file's content length when the tree
    /// API omitted it (LFS-pointer files in some repos).
    private func headSize(for path: String) async -> Int64? {
        let encoded = path
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path
        let urlString = "https://huggingface.co/\(repoID)/resolve/\(branch)/\(encoded)"
        guard let url = URL(string: urlString) else { return nil }
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        request.setValue("ios-local-llm/1.0 (iOS)", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 15
        HFTokenStore.authorize(&request)
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse,
                  http.statusCode == 200 else { return nil }
            if let len = http.value(forHTTPHeaderField: "Content-Length"),
               let bytes = Int64(len) { return bytes }
            if let lfsLen = http.value(forHTTPHeaderField: "X-Linked-Size"),
               let bytes = Int64(lfsLen) { return bytes }
            return nil
        } catch {
            return nil
        }
    }

    // MARK: - Single-file download
    //
    // Routes through BackgroundDownloadCoordinator so downloads keep running
    // when the user backgrounds the app. URLSession's background config does
    // not support partial-file Range resume directly, but it handles its own
    // resume data internally across app launches.

    private func downloadFile(
        meta: HFFileMeta,
        to destination: URL,
        runID: UUID
    ) async throws -> Int64 {
        // Percent-encode each path component so files with spaces, '+', etc work
        let encoded = meta.path
            .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? meta.path
        let hfURL = "https://huggingface.co/\(repoID)/resolve/\(branch)/\(encoded)"
        guard let url = URL(string: hfURL) else { throw URLError(.badURL) }

        // Skip if file is already fully present (requires real known size).
        if meta.size > 0,
           FileManager.default.fileExists(atPath: destination.path),
           let attrs = try? FileManager.default.attributesOfItem(atPath: destination.path),
           let sz = attrs[.size] as? Int64, sz == meta.size {
            return sz
        }

        // Retry transient failures (dropped connection, timeout, 5xx, 429)
        // with exponential backoff + jitter (Retry-After honored when the
        // server sent one). A single flaky packet used to fail the entire
        // repo download; HuggingFace LFS endpoints in particular drop long
        // connections under load. Permanent errors (404 / 401 / 403) are not
        // retried — they won't fix themselves and the user needs to see them.
        let maxAttempts = 3
        var attempt = 0
        while true {
            do {
                _ = try await BackgroundDownloadCoordinator.shared.download(
                    from: url,
                    to: destination,
                    expectedSize: meta.size
                ) { [weak self] received, _ in
                    Task { @MainActor [weak self] in
                        guard let self,
                              self.isCurrent(runID),
                              self.activeFileBytes[meta.path] != nil else { return }
                        self.activeFileBytes[meta.path] = max(0, received)
                        self.updateAggregateProgress()
                    }
                }
                // Integrity check: LFS blobs carry their sha256 in the tree
                // metadata. Verify the bytes on disk match; a mismatch is
                // deleted and rethrown as retryable so the loop re-fetches.
                // Skipped when no oid is known (plain git blobs, fallbacks).
                if let expected = meta.sha256 {
                    let actual = try await Self.sha256Hex(of: destination)
                    if actual.caseInsensitiveCompare(expected) != .orderedSame {
                        try? FileManager.default.removeItem(at: destination)
                        throw NSError(domain: "HFDownload", code: Self.checksumMismatchCode,
                                      userInfo: [NSLocalizedDescriptionKey:
                                        "Checksum mismatch for \(meta.name) — file corrupted in transit."])
                    }
                }
                return meta.size
            } catch {
                attempt += 1
                if Task.isCancelled || Self.isUserCancellation(error) { throw CancellationError() }
                // Per-file 401/403 lose the structured auth classification
                // the tree fetch has — map them to HFAuthError here so
                // run()'s handler sets lastFailureKind to the auth kind and
                // the catalog shows the "Set Token" CTA instead of a
                // generic Retry.
                let ns = error as NSError
                if ns.domain == "HFDownload", ns.code == 401 || ns.code == 403 {
                    if HFTokenStore.authorizationHeaderValue() != nil {
                        throw HFAuthError.tokenRejected(repoID: repoID)
                    } else {
                        throw HFAuthError.tokenRequired(repoID: repoID)
                    }
                }
                guard attempt < maxAttempts, Self.isRetryable(error) else { throw error }
                print("[HFDownload] \(meta.name): attempt \(attempt) failed (\(error.localizedDescription)) — retrying")
                RuntimeLogCenter.emit(
                    "\(repoID): \(meta.name) attempt \(attempt) failed; retrying — \(error.localizedDescription)",
                    level: .warning,
                    subsystem: "download"
                )
                // Reset this file's live counter so a retry does not double-
                // count bytes from the failed attempt alongside its new task.
                if isCurrent(runID) {
                    activeFileBytes[meta.path] = 0
                    updateAggregateProgress()
                }
                try? await Task.sleep(nanoseconds: Self.retryDelayNs(attempt: attempt, error: error))
            }
        }
    }

    /// Whether a download error is worth retrying. Network-layer hiccups and
    /// server-side throttling/5xx are transient; client errors (404/401/403)
    /// are not.
    private static func isRetryable(_ error: Error) -> Bool {
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut,
                 NSURLErrorNetworkConnectionLost,
                 NSURLErrorNotConnectedToInternet,
                 NSURLErrorCannotConnectToHost,
                 NSURLErrorDNSLookupFailed,
                 NSURLErrorResourceUnavailable,
                 NSURLErrorCannotFindHost:
                return true
            default:
                return false
            }
        }
        // BackgroundDownloadCoordinator surfaces HTTP failures as
        // domain "HFDownload" with the status code as `code`. A checksum
        // mismatch is also retryable — re-fetching usually fixes it.
        if ns.domain == "HFDownload" {
            return ns.code == 429 || (500...599).contains(ns.code)
                || ns.code == checksumMismatchCode
                || ns.code == -6
        }
        return false
    }

    /// Synthetic "HFDownload"-domain code for a post-download sha256
    /// mismatch (real HTTP statuses are positive).
    private static let checksumMismatchCode = -4

    /// True when the error is URLSession's representation of a user cancel.
    /// URLSession cancels surface as NSURLErrorCancelled, not
    /// CancellationError — without this mapping a Cancel tap fell into the
    /// generic failure path and showed a "Download failed" toast.
    private static func isUserCancellation(_ error: Error) -> Bool {
        let ns = error as NSError
        return ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled
    }

    /// Delay before retry number `attempt` (1-based). Honors a server
    /// `Retry-After` header when the failed response carried one (attached
    /// to the error's userInfo by BackgroundDownloadCoordinator); otherwise
    /// exponential backoff with ±25% jitter. Capped at 60 s either way.
    private static func retryDelayNs(attempt: Int, error: Error) -> UInt64 {
        let capSeconds: Double = 60
        if let retryAfter = (error as NSError).userInfo["Retry-After"] as? String,
           let seconds = Double(retryAfter), seconds > 0 {
            return UInt64(min(seconds, capSeconds) * 1_000_000_000)
        }
        let base = 1.5 * pow(2, Double(attempt - 1))      // 1.5s, 3s, 6s, …
        let jittered = base * Double.random(in: 0.75...1.25)
        return UInt64(min(jittered, capSeconds) * 1_000_000_000)
    }

    /// Runs `operation`, retrying transient failures (see `isRetryable`)
    /// with the same backoff policy as the per-file download loop.
    private func withRetries<T>(
        maxAttempts: Int = 3,
        _ operation: () async throws -> T
    ) async throws -> T {
        var attempt = 0
        while true {
            do {
                return try await operation()
            } catch {
                attempt += 1
                if Task.isCancelled || Self.isUserCancellation(error) { throw CancellationError() }
                guard attempt < maxAttempts, Self.isRetryable(error) else { throw error }
                print("[HFDownload] \(repoID): attempt \(attempt) failed (\(error.localizedDescription)) — retrying")
                try? await Task.sleep(nanoseconds: Self.retryDelayNs(attempt: attempt, error: error))
            }
        }
    }

    /// Streamed sha256 of a file, computed off the main actor so multi-GB
    /// weights don't stall the UI. Reads in 4 MB chunks via CryptoKit.
    private static nonisolated func sha256Hex(of url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 4 << 20), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    // MARK: - Helpers

    private func isCurrent(_ runID: UUID) -> Bool {
        activeRunID == runID
    }

    private func updateAggregateProgress() {
        let inFlight = activeFileBytes.values.reduce(Int64(0), +)
        let aggregate = completedDownloadBytes + inFlight
        downloadedBytes = totalBytes > 0 ? min(totalBytes, aggregate) : aggregate
        if totalBytes > 0 {
            progress = min(1.0, Double(downloadedBytes) / Double(totalBytes))
        } else {
            progress = Double(filesDone) / Double(max(filesTotal, 1))
        }

        let activePaths = activeFileBytes.keys.sorted()
        if activePaths.count <= 1 {
            currentFile = activePaths.first ?? ""
        } else {
            currentFile = "\(activePaths[0]) + \(activePaths.count - 1) more"
        }
        logProgressIfNeeded()
    }

    private func finishRun(_ runID: UUID) {
        guard isCurrent(runID) else { return }
        activeRunID = nil
        downloadTask = nil
        pauseRequested = false
    }

    private func logProgressIfNeeded() {
        let now = Date()
        let elapsed = now.timeIntervalSince(lastProgressLogAt ?? .distantPast)
        let bytesDelta = downloadedBytes - lastProgressLogBytes
        guard elapsed >= 5 || bytesDelta >= 64 * 1_024 * 1_024 else { return }
        lastProgressLogAt = now
        lastProgressLogBytes = downloadedBytes
        let percent = String(format: "%.1f%%", progress * 100)
        RuntimeLogCenter.emit(
            "\(repoID): \(percent) · \(downloadedBytes.formattedBytes) downloaded",
            subsystem: "download"
        )
    }

    /// Returns the device's free disk space in bytes, or nil on failure.
    nonisolated static func freeDiskBytes() -> Int64? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        guard let values = try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let capacity = values.volumeAvailableCapacityForImportantUsage else { return nil }
        return Int64(capacity)
    }

    private func isComplete(_ meta: HFFileMeta) -> Bool {
        // Treat unknown-size entries as never-complete so the download is
        // forced to run instead of silently skipping the file.
        guard meta.size > 0 else { return false }
        let path = destination.appendingPathComponent(meta.path).path
        guard FileManager.default.fileExists(atPath: path),
              let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int64 else { return false }
        // Exact match: an oversized file (error body appended, truncated
        // retry merge, …) is just as wrong as a short one.
        return size == meta.size
    }

    private func removeTrackedFiles() {
        let fm = FileManager.default
        if let allowlist = fileAllowlist {
            for rule in allowlist {
                let url = destination.appendingPathComponent(rule)
                if fm.fileExists(atPath: url.path) {
                    try? fm.removeItem(at: url)
                }
                // Prune now-empty intermediate dirs up from THIS file's parent
                // (e.g. `nano/foo.mlmodelc/` → `nano/`) — but never the shared
                // `destination` root, which a sibling variant still lives under.
                pruneEmptyParents(below: url.deletingLastPathComponent())
            }
            return
        }
        if fm.fileExists(atPath: destination.path) {
            try? fm.removeItem(at: destination)
        }
    }

    /// Removes empty directories walking UP from `start`, stopping BEFORE the
    /// shared `destination` root. The root is NEVER deleted here: allowlisted
    /// variants (KittenTTS Nano/Mini, the two Whisper models) share one
    /// destination, and deleting it on one variant's removal would orphan the
    /// other. Previously this started AT `destination` and could remove it
    /// outright once empty.
    private func pruneEmptyParents(below start: URL) {
        let fm = FileManager.default
        let stopPath = destination.standardizedFileURL.path
        var current = start.standardizedFileURL
        while current.path != stopPath && current.path.hasPrefix(stopPath + "/") {
            guard let entries = try? fm.contentsOfDirectory(atPath: current.path),
                  entries.isEmpty else { break }
            try? fm.removeItem(at: current)
            let parent = current.deletingLastPathComponent().standardizedFileURL
            if parent.path == current.path { break }
            current = parent
        }
    }

    private func alreadyDownloadedBytes(_ files: [HFFileMeta]) -> Int64 {
        files.filter { isComplete($0) }.map { $0.size }.reduce(0, +)
    }

    private func reset(keepState: Bool = false) {
        if !keepState { state = .idle }
        progress = 0
        downloadedBytes = 0
        totalBytes = 0
        currentFile = ""
        filesDone = 0
        filesTotal = 0
        completedDownloadBytes = 0
        activeFileBytes.removeAll(keepingCapacity: true)
    }
}
