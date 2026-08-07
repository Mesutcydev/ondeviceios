import Foundation
import Combine

// MARK: - DownloadableModel
// Metadata + download manager for a single model package.

@MainActor
final class DownloadableModel: ObservableObject, Identifiable {

    let id: String
    /// Canonical Hugging Face repository. `id` may be an app preset slug
    /// (for example `bonsai-27b-1bit`), which is not loadable by HubApi.
    let sourceRepoID: String
    let displayName: String
    let subtitle: String
    let sizeLabel: String
    let category: Category
    let isRequired: Bool
    let docURL: String?

    // ── Catalog presentation metadata ───────────────────────────────
    //
    // All inferred when not provided so legacy call sites stay
    // compatible. The Catalog UI consumes these to draw vendor
    // thumbnails, capability pills, family grouping, gated lock
    // icons. None of these affect the download itself.

    /// Capabilities to show as pills on the row (vision / thinking /
    /// best / recommended / newRelease / gated).
    let capabilities: Set<ModelCapability>

    /// Publisher — inferred from the repo id (`google/`, `Qwen/`,
    /// `mlx-community/`, …) when not provided. Drives the thumbnail.
    let vendor: ModelVendor

    /// Group key for catalog rows. Multiple entries with the same
    /// familyID collapse into one summary row ("Qwen 3 VL · 2 models").
    let familyID: String

    /// Long description shown when the row is expanded / on the
    /// family detail sheet. Falls back to `subtitle` when nil.
    let longDescription: String?

    /// Explicit runtime hint when the catalog knows which backend
    /// executes this model. Lets pickers and registries avoid repo-name
    /// guessing for curated entries.
    let runtime: ModelRuntime?

    /// Best-effort working-set estimate and context length when known.
    /// Curated assistant rows populate these from AssistantModelCatalog;
    /// imported/custom models can fall back to heuristics.
    let approxRAMBytes: Int64?
    let contextWindowTokens: Int?
    let platformCompatibility: ModelPlatformCompatibility?

    /// In-app voice engine implemented for this repo, if any. Nil means
    /// "stored locally only" — the app recognizes the files but has no
    /// selectable runtime for them yet.
    let supportedVoiceEngine: VoiceEngineKind?

    enum Category { case assistant, vlm, voice, imageGen }

    /// A package can serve more than its primary catalog category. Today this
    /// is used by unified text+vision models: they stay a single download but
    /// surface in both Assistant and Lens.
    func supportsCategory(_ requested: Category) -> Bool {
        if category == requested { return true }
        return requested == .vlm && capabilities.contains(.vision)
    }

    // The underlying downloader. Nil for entries that aren't downloaded
    // through this catalog (none in current usage — every catalog entry
    // sets this).
    let downloader: HFModelDownloadManager?

    init(
        id: String,
        displayName: String,
        subtitle: String,
        sizeLabel: String,
        category: Category,
        isRequired: Bool = false,
        docURL: String? = nil,
        downloader: HFModelDownloadManager? = nil,
        // ── Catalog metadata (all optional) ─────────────────────────
        // `repoID` is the source of truth for vendor + family + gated
        // inference. Passed explicitly because `id` may be a slug
        // (`qwen3-4b`) rather than a full repo path.
        repoID: String? = nil,
        capabilities: Set<ModelCapability> = [],
        vendor: ModelVendor? = nil,
        familyID: String? = nil,
        longDescription: String? = nil,
        runtime: ModelRuntime? = nil,
        approxRAMBytes: Int64? = nil,
        contextWindowTokens: Int? = nil,
        platformCompatibility: ModelPlatformCompatibility? = nil,
        supportedVoiceEngine: VoiceEngineKind? = nil
    ) {
        self.id               = id
        self.displayName      = displayName
        self.subtitle         = subtitle
        self.sizeLabel        = sizeLabel
        self.category         = category
        self.isRequired       = isRequired
        self.docURL           = docURL
        self.downloader       = downloader

        // Source for inference. Prefer the explicit repoID arg, then
        // the downloader's repoID, then the subtitle (which IS the
        // repoID for most catalog entries today). subtitle is
        // non-optional so the chain bottoms out there — no need for
        // a further `?? id` fallback.
        let inferenceSource: String = repoID
            ?? downloader?.repoID
            ?? subtitle
        self.sourceRepoID = inferenceSource
        self.vendor   = vendor   ?? ModelVendor.infer(from: inferenceSource)
        self.familyID = familyID ?? ModelFamily.inferID(from: inferenceSource)

        // Merge explicit capabilities with auto-detected gated flag —
        // this way curated entries don't have to remember to add .gated
        // by hand for `meta-llama/...` repos.
        var caps = capabilities
        if KnownGatedRepos.isGated(repoID: inferenceSource) {
            caps.insert(.gated)
        }
        self.capabilities = caps

        self.longDescription = longDescription
        self.runtime = runtime
        self.approxRAMBytes = approxRAMBytes
        self.contextWindowTokens = contextWindowTokens
        self.platformCompatibility = platformCompatibility
        self.supportedVoiceEngine = supportedVoiceEngine
    }

    // MARK: - State pass-through

    var state: HFModelDownloadManager.DownloadState { downloader?.state ?? .idle }
    var progress: Double { downloader?.progress ?? 0 }
    var downloadedBytes: Int64 { downloader?.downloadedBytes ?? 0 }
    var totalBytes: Int64 { downloader?.totalBytes ?? 0 }
    var currentFile: String { downloader?.currentFile ?? "" }
    var isReady: Bool { state == .ready }
    var lastFailureKind: HFModelDownloadManager.FailureKind {
        downloader?.lastFailureKind ?? .none
    }

    func start() {
        // Imported models use synthetic `local/<folder>` identities. They are
        // already on disk and must never be interpreted as Hugging Face repos
        // (which produced a guaranteed huggingface.co/local/... 404). Refresh
        // readiness in case the row was built before import validation ended.
        if sourceRepoID.lowercased().hasPrefix("local/") {
            downloader?.checkIfReady()
            if downloader?.state != .ready {
                ToastCenter.shared.error(
                    "Local model files need attention",
                    detail: "Re-import this model from Files; local models cannot be downloaded from Hugging Face."
                )
            }
            return
        }
        if let compatibility = platformCompatibility,
           !compatibility.supportsCurrentPlatform {
            ToastCenter.shared.error(
                "Model not supported on this device",
                detail: compatibility.detail
            )
            return
        }

        let footprint = approxRAMBytes ?? MemoryAdvisor.estimatedFootprint(for: id)
        if footprint > 0, case .over = MemoryAdvisor.fit(forFootprint: footprint) {
            ToastCenter.shared.error(
                "Model won't run safely on this device",
                detail: "This model needs more working memory than iOS can reliably provide. Choose a smaller model."
            )
            return
        }
        downloader?.start()
    }
    func cancel() { downloader?.cancel() }
    func pause() { downloader?.pause() }
    func resume() { downloader?.start() }
    /// Permanently abandons an incomplete transfer and removes its partial
    /// files. Kept separate from `cancel()` because older callers use cancel
    /// as a pause-for-resume operation.
    func cancelDownload() { downloader?.abandon() }
    func delete() throws { try downloader?.delete() }
    func redownload() { downloader?.redownload() }
    func checkIfReady() { downloader?.checkIfReady() }
}

// MARK: - ModelDownloadCenter

@MainActor
final class ModelDownloadCenter: ObservableObject {

    static let shared = ModelDownloadCenter()

    // ── Gemma 4 12B staging gate ──────────────────────────────────────────
    // The Swift runtime was migrated to mlx-swift-lm 3.x specifically to get
    // the `gemma4` VLM loader (text+vision; audio not yet supported upstream).
    // That bump changes the inference runtime for EVERY model, so Gemma 4 is
    // kept off the visible catalog until the bump is verified on-device via
    // TestFlight. Flip this to `true` once a real device confirms the 12B
    // loads + generates and existing models (Qwen3-VL, Gemma 3) still work.
    // See the curated-VLM list in `buildCatalog()` for the gated entry.
    static let gemma4CatalogEnabled = false

    // All registered downloadable models in display order.
    @Published private(set) var models: [DownloadableModel] = [] {
        didSet { rebindDownloaders() }
    }

    private var cancellables: Set<AnyCancellable> = []

    // Forwarding subscriptions from each downloader's @Published `state` to
    // this center's objectWillChange. DownloadableModel and
    // HFModelDownloadManager are nested ObservableObjects whose changes do NOT
    // propagate to a view observing the center — so without this, tapping
    // "Download" flipped the downloader to .enumerating but the catalog list
    // never re-rendered to move the row into the in-flight section ("Download
    // does nothing"). We forward `$state` only (not `progress`): state drives
    // the section a model lives in; live progress is rendered by InstallingRow,
    // which observes the downloader directly, so per-tick re-renders here would
    // be pure waste.
    private var downloaderBindings: Set<AnyCancellable> = []

    private func rebindDownloaders() {
        downloaderBindings.removeAll(keepingCapacity: true)
        for m in models {
            m.downloader?.$state
                .removeDuplicates()
                .sink { [weak self] _ in self?.objectWillChange.send() }
                .store(in: &downloaderBindings)
        }
    }

    private init() {
        migrateBackupExclusion()
        migrateStaleSettings()
        buildCatalog()
        // Defer the custom-download scan (+ role repair + state refresh) off
        // the synchronous launch path: its per-folder allocatedSizeOfDirectory
        // deep-walk blocked the first frame for users with many/large
        // downloaded repos. The curated catalog (buildCatalog) is already
        // populated for immediate display; custom repos appear a beat later,
        // and ContentView.onAppear also kicks a refreshAllStates().
        Task { @MainActor [weak self] in
            guard let self else { return }
            self.scanCustomDownloads()
            self.reconcileInstalledRegistry()
            self.repairInvalidRoleAssignments()
            self.refreshAllStates()
        }
        // Keep the cached storage stats fresh when any download completes.
        // Also reconcile the installed-model registry so freshly-downloaded
        // models appear in pickers immediately — no relaunch needed.
        NotificationCenter.default.publisher(for: .hfModelDownloadCompleted)
            .sink { [weak self] note in
                Task { @MainActor in
                    self?.refreshStorageStats()
                    // Reconcile the registry with disk — newly downloaded
                    // models get discovered and published to pickers.
                    InstalledModelRegistry.shared.reconcileWithDisk()
                    self?.reconcileInstalledRegistry()
                    MemoryAdvisor.invalidateFootprintCache()
                }
            }
            .store(in: &cancellables)
        // Best-effort: figure out which FastVLM mirror is alive today and
        // rebuild the catalog if it differs from the saved value. Runs once,
        // off the main thread, no UI blocking.
        Task { @MainActor in
            let before = AppSettings.shared.fastVLMRepoID
            if let resolved = await FastVLMRepoAutoDiscovery.shared.discover(),
               resolved != before {
                self.rebuildFastVLMEntry(with: resolved)
            }
        }
    }

    /// Replaces the existing FastVLM catalog entry when auto-discovery finds
    /// a different working repo.
    private func rebuildFastVLMEntry(with repoID: String) {
        guard let idx = models.firstIndex(where: { $0.id == FastVLMService.modelID }) else { return }
        // Don't yank a download out from under the user. If the current FastVLM
        // entry is mid-download, replacing it with a new downloader would orphan
        // the in-flight task (its bytes still land in the same FastVLMModels dir
        // but nothing tracks them). Skip the swap while active — discovery
        // re-runs on the next launch when it's idle.
        if models[idx].downloader?.state.isActive == true { return }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dest = docs
            .appendingPathComponent("FastVLMModels")
            .appendingPathComponent(FastVLMConfig.mlxModelDirectory)
        let new = DownloadableModel(
            id: FastVLMService.modelID,
            displayName: "FastVLM MLX Weights",
            subtitle: repoID,
            sizeLabel: "~400 MB",
            category: .vlm,
            isRequired: true,
            docURL: "https://huggingface.co/\(repoID)",
            downloader: HFModelDownloadManager(repoID: repoID, destination: dest)
        )
        new.checkIfReady()
        models[idx] = new
        ToastCenter.shared.info("FastVLM repo updated", detail: repoID)
    }

    /// One-time migration: mark every model-storage root as excluded from
    /// iCloud/iTunes backup. Weights are re-downloadable and were silently
    /// inflating user backups. Directory-level exclusion covers children,
    /// so flagging the roots is enough for everything already on disk; new
    /// downloads/imports set the flag on their own destination.
    private func migrateBackupExclusion() {
        let key = "didExcludeModelStorageFromBackup"
        guard !UserDefaults.standard.bool(forKey: key) else { return }
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let roots = ["HFModels", "LLMModels", "FastVLMModels", "GGUFModels",
                     "VoiceModels", "huggingface"]
            .map { docs.appendingPathComponent($0, isDirectory: true) }
        for root in roots where fm.fileExists(atPath: root.path) {
            FileManager.excludeFromBackup(root)
        }
        UserDefaults.standard.set(true, forKey: key)
    }

    /// One-shot migration that clears settings carrying values from older
    /// builds where the default repo ID was broken (private/gated).
    private func migrateStaleSettings() {
        let s = AppSettings.shared
        // Old default that 401'd:
        let brokenFastVLM = "mlx-community/llava-fastvithd_0.5b_stage3_llm.fp16"
        if s.fastVLMRepoID == brokenFastVLM {
            s.fastVLMRepoID = "apple/FastVLM-0.5B-MLX"
            print("[ModelDownloadCenter] Migrated stale FastVLM repo ID")
        }
        // REMOVED: a previous version of this code reset
        // assistantModelID from "qwen3-4b" to "qwen2.5-coder-1.5b" on
        // every singleton init, because of a key-layout mismatch
        // between Qwen3 weights and the old mlx-swift-examples Qwen3
        // model class. mlx-swift-examples now ships Qwen3.swift and
        // Qwen3MoE.swift with the correct layout — the migration is
        // stale and was actively reverting users' explicit picks of
        // Qwen3-4B, including reverting them back to Qwen2.5-Coder-1.5B
        // on every cold launch. If a future model-id ever needs a
        // similar migration, gate it on `hasPickedAssistantModel ==
        // false` so explicit user choices aren't clobbered.
    }

    /// Older builds could let a voice repo masquerade as the active camera
    /// model (or other cross-role combinations) after a bad category guess.
    /// Repair those persisted ids against the category map we just rebuilt
    /// from disk so the wrong model stops poisoning the picker state.
    private func repairInvalidRoleAssignments() {
        let settings = AppSettings.shared

        func matchingModel(_ storedID: String) -> DownloadableModel? {
            models.first { $0.id == storedID || $0.sourceRepoID == storedID }
        }

        let assistantStored = LocalModelRegistry.unwrapAssistantSelectionID(settings.assistantModelID)
        if let assistantModel = matchingModel(assistantStored),
           !assistantModel.supportsCategory(.assistant) {
            settings.assistantModelID = AssistantModelCatalog.presets.first?.id ?? ""
            settings.hasPickedAssistantModel = false
        }

        let voiceConversationStored = LocalModelRegistry.unwrapAssistantSelectionID(settings.voiceConversationModelID)
        if let voiceConversationModel = matchingModel(voiceConversationStored),
           !voiceConversationModel.supportsCategory(.assistant) {
            settings.voiceConversationModelID = ""
        }

        let visualStored = LocalModelRegistry.storedVisionSelectionID(settings.cameraVisualModelID)
        if let visualModel = matchingModel(visualStored),
           !visualModel.supportsCategory(.vlm) {
            settings.cameraVisualModelID = ""
            settings.hasPickedCameraVisualModel = false
        }
    }

    // MARK: - Custom (HF-searched) downloads

    /// Registers a model downloaded ad-hoc via HF Search, so it shows up in
    /// the catalog and can be re-opened / deleted later.
    func registerCustom(
        repoID: String,
        displayName: String,
        subtitle: String,
        category: DownloadableModel.Category,
        sizeLabel: String,
        docURL: String? = nil,
        downloader: HFModelDownloadManager,
        runtime: ModelRuntime? = nil
    ) {
        // Avoid duplicates — by repoID AND by on-disk destination. Two ids can
        // point at the same directory (e.g. a curated catalog VLM whose slash
        // form and a custom scan's underscore form both resolve to the same
        // HFModels/<repo> folder); deduping on the standardized path collapses
        // them to one row instead of showing the model twice.
        let destPath = downloader.destination.standardizedFileURL.path
        guard !models.contains(where: {
            $0.id == repoID || $0.downloader?.destination.standardizedFileURL.path == destPath
        }) else { return }
        let model = DownloadableModel(
            id: repoID,
            displayName: displayName,
            subtitle: subtitle,
            sizeLabel: sizeLabel,
            category: category,
            isRequired: false,
            docURL: docURL,
            downloader: downloader,
            runtime: runtime
        )
        models.append(model)
        model.checkIfReady()
    }

    /// Drops a custom model from the catalog (called after delete).
    func unregisterCustom(repoID: String) {
        models.removeAll { $0.id == repoID && !$0.isRequired && $0.id.contains("/") }
    }

    /// Returns the existing downloader for `repoID` if registered, or nil.
    /// HFSearchRow uses this so the search row shares progress state with the
    /// catalog instead of spawning a parallel downloader.
    func existingDownloader(forRepoID repoID: String) -> HFModelDownloadManager? {
        models.first {
            $0.id.caseInsensitiveCompare(repoID) == .orderedSame
                || $0.sourceRepoID.caseInsensitiveCompare(repoID) == .orderedSame
        }?.downloader
    }

    /// Makes the model center reflect every valid record in the authoritative
    /// installed registry. The conversation picker already consumes that
    /// registry directly; without this merge, a valid community model could
    /// appear in chat while remaining absent from the Models tab.
    func reconcileInstalledRegistry() {
        reconcileInstalledRegistry(records: InstalledModelRegistry.shared.records)
    }

    /// Injectable variant used by regression tests. A registry-backed entry
    /// replaces a matching non-ready catalog entry so the actual on-disk
    /// location wins over a stale downloader destination.
    func reconcileInstalledRegistry(records: [InstalledModelRecord]) {
        var updated = models
        var changed = false

        for record in records where record.validationState.isActivatable {
            let existingIndex = updated.firstIndex {
                $0.sourceRepoID.caseInsensitiveCompare(record.repoID) == .orderedSame
                    || $0.id.caseInsensitiveCompare(record.repoID) == .orderedSame
            }
            if let existingIndex, updated[existingIndex].isReady {
                continue
            }

            let existing = existingIndex.map { updated[$0] }
            let downloader = HFModelDownloadManager(
                repoID: record.repoID,
                destination: record.localURL
            )
            downloader.checkIfReady()

            let category = LocalModelRegistry.category(in: record.localURL)
            var capabilities = record.capabilities
            capabilities.formUnion(existing?.capabilities ?? [])
            if category == .vlm { capabilities.insert(.vision) }

            let installed = DownloadableModel(
                id: existing?.id ?? record.repoID,
                displayName: existing?.displayName ?? record.displayName,
                subtitle: existing?.subtitle
                    ?? "\(record.engine) · \(record.quantization ?? "unknown")",
                sizeLabel: record.downloadBytes > 0
                    ? record.downloadBytes.formattedBytes
                    : existing?.sizeLabel ?? "—",
                category: existing?.category ?? category,
                isRequired: existing?.isRequired ?? false,
                docURL: existing?.docURL
                    ?? "https://huggingface.co/\(record.repoID)",
                downloader: downloader,
                repoID: record.repoID,
                capabilities: capabilities,
                vendor: existing?.vendor,
                familyID: existing?.familyID,
                longDescription: existing?.longDescription,
                runtime: record.engine,
                approxRAMBytes: existing?.approxRAMBytes,
                contextWindowTokens: existing?.contextWindowTokens,
                platformCompatibility: existing?.platformCompatibility,
                supportedVoiceEngine: existing?.supportedVoiceEngine
            )

            if let existingIndex {
                updated[existingIndex] = installed
            } else {
                updated.append(installed)
            }
            changed = true
        }

        if changed {
            models = updated
        }
    }

    /// Scans `Documents/HFModels/` for previously-downloaded HF repos and
    /// re-registers them so they survive app restarts.
    private func scanCustomDownloads() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let root = docs.appendingPathComponent("HFModels", isDirectory: true)
        guard let entries = try? FileManager.default.contentsOfDirectory(
            atPath: root.path
        ) else { return }

        let fm = FileManager.default
        for entry in entries {
            let dest = root.appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dest.path, isDirectory: &isDir), isDir.boolValue else { continue }

            // Try the folder itself first.
            if registerIfReady(folderName: entry, directory: dest) { continue }

            // Recursive fallback: a model can sit one level down (e.g. a Files
            // import that wrapped the real model folder, or a repo stored as
            // <author>/<name>). Scan immediate subdirectories so those aren't
            // silently invisible.
            guard let subs = try? fm.contentsOfDirectory(atPath: dest.path) else { continue }
            for sub in subs {
                let subDir = dest.appendingPathComponent(sub)
                var subIsDir: ObjCBool = false
                guard fm.fileExists(atPath: subDir.path, isDirectory: &subIsDir), subIsDir.boolValue else { continue }
                _ = registerIfReady(folderName: sub, directory: subDir)
            }
        }
    }

    /// Registers the model at `directory` if its files are complete. Returns
    /// true when a model was found (already registered or newly added).
    @discardableResult
    private func registerIfReady(folderName: String, directory: URL) -> Bool {
        // Prefer the lossless sidecar written at download time. Folder name
        // encoding is `author/name → author_name`, which is lossy whenever
        // either side already contains an underscore.
        let repoID = recoverRepoID(from: folderName, in: directory)

        let downloader = HFModelDownloadManager(repoID: repoID, destination: directory)
        downloader.checkIfReady()
        guard downloader.state == .ready else { return false }

        if models.contains(where: { $0.id == repoID }) { return true }

        let dirSize = (try? FileManager.default.allocatedSizeOfDirectory(at: directory)) ?? 0
        let runtime: ModelRuntime? = LocalModelFileValidator.hasValidGGUFTextModel(in: directory)
            ? .llamaCpp
            : nil
        let model = DownloadableModel(
            id: repoID,
            displayName: repoID.split(separator: "/").last.map(String.init) ?? repoID,
            subtitle: repoID,
            sizeLabel: dirSize > 0 ? dirSize.formattedBytes : "—",
            // Inspect config.json so a model downloaded via HFSearch keeps its
            // real category across restarts — otherwise every restored entry
            // came back as .assistant, and VLMs vanished from the vision
            // picker even though the files were on disk.
            category: LocalModelRegistry.category(in: directory),
            isRequired: false,
            docURL: "https://huggingface.co/\(repoID)",
            downloader: downloader,
            runtime: runtime
        )
        models.append(model)
        // Also register in the installed-model registry so community/
        // custom downloads appear in pickers immediately without a relaunch.
        if let record = InstalledModelRegistry.validateDirectory(directory, repoID: repoID) {
            InstalledModelRegistry.shared.register(record)
        }
        return true
    }

    /// Recovers the original `author/name` repo id from a download folder.
    /// New downloads drop a `.repoID` sidecar inside the folder; for legacy
    /// folders (created before the sidecar existed) we fall back to splitting
    /// on the FIRST underscore only — that's correct as long as the author
    /// part has no underscores, which is the common case on HuggingFace. The
    /// previous global `_` → `/` substitution corrupted any repo with
    /// underscores in either component.
    private func recoverRepoID(from folderName: String, in directory: URL) -> String {
        let sidecar = directory.appendingPathComponent(".repoID")
        if let data = try? Data(contentsOf: sidecar),
           let id = String(data: data, encoding: .utf8)?
                       .trimmingCharacters(in: .whitespacesAndNewlines),
           !id.isEmpty {
            return id
        }
        if let firstUnderscore = folderName.firstIndex(of: "_") {
            let author = folderName[..<firstUnderscore]
            let name = folderName[folderName.index(after: firstUnderscore)...]
            return "\(author)/\(name)"
        }
        return folderName
    }

    // MARK: - Catalog

    private func buildCatalog() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

        // ── 1. Assistant LLMs ─────────────────────────────────────────────────
        // One Download Center entry per AssistantModelCatalog preset. Each
        // writes to Documents/LLMModels/<dirName>/ where <dirName> is the
        // repoID's last path component — that's the directory
        // CodingAssistantService.preStagedDirectory() looks for, so a
        // pre-staged download is picked up directly without a second
        // round-trip through HubApi.
        //
        // Hardcoding a single Qwen3-4B entry (the previous shape) meant
        // users on any other preset (Qwen3-1.7B, Qwen3-8B, Llama 3.2,
        // Qwen2.5-Coder, etc.) had no way to pre-stage their model —
        // every cold launch re-downloaded silently via HubApi.
        for preset in AssistantModelCatalog.presets {
            let dirName = preset.repoID.split(separator: "/").last.map(String.init)
                ?? preset.repoID
            let dest = docs.appendingPathComponent("LLMModels")
                .appendingPathComponent(dirName)
            // Prefer a preset's published package size. Conventional 4-bit
            // models can still use the historical RAM × 0.6 estimate, but
            // Bonsai's 1/2-bit packages are far smaller and need an explicit
            // value for an honest download label and disk preflight.
            let downloadBytes = preset.downloadSizeBytes
                ?? Int64(Double(preset.approxRAMBytes) * 0.6)
            let sizeLabel = "~\(downloadBytes.formattedBytes)"

            models.append(DownloadableModel(
                id: preset.id,
                // The section/selector already says "Coding Assistant"; the old
                // per-row suffix was pure noise and clipped the long Qwen ids.
                displayName: preset.displayName,
                // Human one-liner ("4-bit · 2.3 GB · refreshed flagship"); the
                // raw repo id is demoted to the card's muted meta line.
                subtitle: preset.subtitle,
                sizeLabel: sizeLabel,
                category: .assistant,
                isRequired: false,
                docURL: "https://huggingface.co/\(preset.repoID)",
                downloader: HFModelDownloadManager(
                    repoID: preset.repoID,
                    destination: dest
                ),
                repoID: preset.repoID,
                capabilities: preset.capabilities,
                runtime: preset.runtime,
                approxRAMBytes: preset.approxRAMBytes,
                contextWindowTokens: preset.contextWindowTokens,
                platformCompatibility: preset.platformCompatibility
            ))
        }

        // ── 2. FastVLM MLX weights (Qwen2-0.5B + projector) ──────────────────
        // Apple's official public FastVLM weights on HuggingFace.
        // The repo ID is read from AppSettings so users can swap to a
        // different mirror (apple/*, mlx-community/*, or their own) if the
        // default ever changes.
        let fastVLMRepo = AppSettings.shared.fastVLMRepoID
        let fastVLMDest = docs
            .appendingPathComponent("FastVLMModels")
            .appendingPathComponent(FastVLMConfig.mlxModelDirectory)

        models.append(DownloadableModel(
            id: FastVLMService.modelID,
            displayName: "FastVLM MLX Weights",
            subtitle: "On-device vision encoder · required for Lens",
            sizeLabel: "~400 MB",
            category: .vlm,
            isRequired: true,
            docURL: "https://huggingface.co/\(fastVLMRepo)",
            downloader: HFModelDownloadManager(
                repoID: fastVLMRepo,
                destination: fastVLMDest
            ),
            approxRAMBytes: MemoryAdvisor.estimatedFootprint(for: FastVLMService.modelID)
        ))

        // ── 2b. Curated VLM catalog ──────────────────────────────────────────
        // These appear in the Models tab → Catalog section so users can
        // download them with one tap, bypassing HF Search. HF Search
        // tends to surface Google's gated `google/gemma-3-*` repos first
        // (401 without an HF login), and our preferred mlx-community
        // quantized mirrors don't always rank high enough to find
        // without the exact repo path.
        //
        // Every entry here lands in `Documents/HFModels/<flattened>/`
        // (matches the path checked by `LensInferenceLoop.stagedDirectory`
        // and `CodingAssistantService.preStagedDirectory`).
        struct CuratedVLM {
            let repoID: String
            let displayName: String
            let sizeLabel: String
            let approxRAMBytes: Int64
            /// Capability pills shown on the row. .vision is implicit
            /// for every VLM and added below — list only the extras
            /// (.thinking, .best, .recommended, .newRelease).
            let capabilities: Set<ModelCapability>
            /// Optional one-paragraph description for the detail sheet.
            let longDescription: String?
            /// Repositories such as GGUF collections contain several complete
            /// quantizations. Download only the validated model/projector pair
            /// instead of every variant in the repository.
            var fileAllowlist: [String]? = nil
            /// Execution backend is part of the curated compatibility contract;
            /// do not infer it from a repository naming convention.
            var runtime: ModelRuntime = .mlx
        }
        var curatedVLMs: [CuratedVLM] = [
            // Qwen3-VL family — Qwen3VL.swift in mlx-swift-examples.
            // Compatible (no upstream bug). Sizes picked to fit the
            // per-process memory cap on each device class.
            CuratedVLM(
                repoID: "mlx-community/Qwen3-VL-2B-Instruct-4bit",
                displayName: "Qwen 3 VL 2B Instruct (4-bit)",
                sizeLabel: "~1.7 GB",
                approxRAMBytes: 2_700_000_000,
                capabilities: [.thinking],
                longDescription: "Compact multimodal model from the Qwen team. Strong at general scene description and document understanding; ideal for mid-tier devices."
            ),
            CuratedVLM(
                repoID: "mlx-community/Qwen3-VL-4B-Instruct-4bit",
                displayName: "Qwen 3 VL 4B Instruct (4-bit)",
                sizeLabel: "~2.9 GB",
                approxRAMBytes: 4_600_000_000,
                capabilities: [.thinking, .best],
                longDescription: "Larger Qwen 3 VL variant with sharper visual reasoning. Recommended for Pro / Max devices with ≥6 GB process memory."
            ),
            // Gemma 3 4B IT multimodal. Route this family through the
            // vendored llama.cpp + mtmd backend rather than MLX. Gemma's MLX
            // processor forces a 896² SigLIP pass and previously crossed the
            // iOS process ceiling during vision prefill. The same app already
            // vendors mtmd's Gemma 3 implementation; its bounded context and
            // mmap-backed Q4 weights make the model practical on Max devices.
            // The allowlist is essential: this repo contains Q4, Q8 and f16
            // variants (15+ GB total), but one text model + mmproj is a complete
            // runnable VLM pair.
            CuratedVLM(
                repoID: "ggml-org/gemma-3-4b-it-GGUF",
                displayName: "Gemma 3 4B IT (Q4_K_M GGUF)",
                sizeLabel: "~3.4 GB",
                // 2.49 GB Q4 text weights + 851 MB f16 vision projector.
                // Runtime estimate includes the bounded 1K context, mtmd
                // vision graph and transient load/warmup buffers.
                approxRAMBytes: 5_350_000_000,
                capabilities: [],
                longDescription: "Google's multimodal Gemma 3 with its 896² SigLIP encoder, running through llama.cpp + mtmd with a memory-bounded Lens profile.",
                fileAllowlist: [
                    "gemma-3-4b-it-Q4_K_M.gguf",
                    "mmproj-model-f16.gguf",
                ],
                runtime: .llamaCpp
            ),
            // SmolVLM2-500M-Video-Instruct GGUF — runs through
            // llama.cpp + mtmd (LlamaCppVLMService), NOT MLX. The
            // MLX integration of this model family is broken
            // upstream (see LENS_PIPELINE.md); the GGUF + llama.cpp
            // path doesn't share that bug. ~520 MB total (LLM Q8_0
            // + mmproj Q8_0). Fast, well-tested OCR / screen-reading.
            CuratedVLM(
                repoID: "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
                displayName: "SmolVLM 2 500M (Q8 GGUF)",
                sizeLabel: "~520 MB",
                approxRAMBytes: 850_000_000,
                capabilities: [.recommended],
                longDescription: "Tiny but capable VLM by Hugging Face, running through llama.cpp + mtmd. Fast on every device tier — great default for screen reading and OCR.",
                runtime: .llamaCpp
            ),
        ]
        // Gemma 4 12B (4-bit) — Google's unified encoder-free multimodal
        // model. Loads via the `gemma4` type registered in mlx-swift-lm 3.x
        // (text+vision; audio parsed but not yet processed). ~7 GB on disk,
        // ~16 GB real footprint → .max-tier only. STAGED: appended only when
        // `gemma4CatalogEnabled` is flipped after on-device verification of
        // the runtime bump (see the flag near the top of this file).
        if Self.gemma4CatalogEnabled {
            curatedVLMs.append(
                CuratedVLM(
                    repoID: "mlx-community/gemma-4-12B-it-4bit",
                    displayName: "Gemma 4 12B IT (4-bit)",
                    sizeLabel: "~7 GB",
                    approxRAMBytes: 16_000_000_000,
                    capabilities: [.best, .newRelease],
                    longDescription: "Google's unified, encoder-free multimodal Gemma 4. Vision + text in a single backbone; strong document and scene understanding. Large — best on Max-tier devices with ≥16 GB unified memory."
                )
            )
        }
        for vlm in curatedVLMs {
            let flattened = vlm.repoID.replacingOccurrences(of: "/", with: "_")
            let dest = docs.appendingPathComponent("HFModels")
                .appendingPathComponent(flattened)
            // Every entry in this list is a VLM, so .vision is implicit.
            var caps = vlm.capabilities
            caps.insert(.vision)
            models.append(DownloadableModel(
                id: vlm.repoID,
                displayName: vlm.displayName,
                // Human description first; repo id is demoted to the meta line.
                subtitle: vlm.longDescription ?? vlm.repoID,
                sizeLabel: vlm.sizeLabel,
                category: .vlm,
                isRequired: false,
                docURL: "https://huggingface.co/\(vlm.repoID)",
                downloader: HFModelDownloadManager(
                    repoID: vlm.repoID,
                    destination: dest,
                    fileAllowlist: vlm.fileAllowlist
                ),
                repoID: vlm.repoID,
                capabilities: caps,
                longDescription: vlm.longDescription,
                runtime: vlm.runtime,
                approxRAMBytes: vlm.approxRAMBytes
            ))
        }

        // ── 3. Official KittenTTS 0.8 ONNX variants ───────────────────────────
        // The repo's real layout (probed via HF tree API):
        //   nano/kittentts_10s.mlmodelc/...     ← pre-compiled CoreML
        //   nano/voices.npz                      ← voice embeddings
        //   mini/kittentts_mini_10s.mlmodelc/...
        //   mini/voices.npz
        // Previous allowlist asked for `kitten_tts_nano.mlpackage/` which
        // doesn't exist in this repo — so the download fetched nothing into
        // the expected location and the validator reported "not found" on
        // every cold launch (the user-reported "re-download every time"
        // bug). Allowlist now matches real paths exactly.
        for configuration in KittenManifest.configurations {
            let title: String
            let memory: Int64
            switch configuration.id {
            case .micro08:
                title = "KittenTTS Micro 0.8"
                memory = 300_000_000
            case .nano08Int8:
                title = "KittenTTS Nano 0.8 INT8"
                memory = 220_000_000
            case .mini08:
                title = "KittenTTS Mini 0.8"
                memory = 550_000_000
            }
            let bytes = configuration.artifacts.reduce(Int64(0)) { $0 + $1.expectedBytes }
            models.append(DownloadableModel(
                id: configuration.id.rawValue,
                displayName: title,
                subtitle: "Official KittenML ONNX · 8 English voices · 24 kHz",
                sizeLabel: "~\(bytes.formattedBytes)",
                category: .voice,
                docURL: "https://huggingface.co/\(configuration.repositoryID)",
                downloader: HFModelDownloadManager(
                    repoID: configuration.repositoryID,
                    destination: configuration.installedDirectory,
                    branch: configuration.revision,
                    fileAllowlist: configuration.artifacts.map(\.relativePath)
                ),
                approxRAMBytes: memory,
                supportedVoiceEngine: .kittenTTS
            ))
        }

        // ── 5. Kokoro-82M CoreML ─────────────────────────────────────────────
        // Uses aufklarer/Kokoro-82M-CoreML — the cleanest public CoreML
        // export of Kokoro-82M today. Layout (probed via HF tree API):
        //
        //   kokoro_5s.mlmodelc/   ← main acoustic model, pre-compiled
        //   G2PEncoder.mlmodelc/  ← grapheme-to-phoneme encoder
        //   G2PDecoder.mlmodelc/  ← grapheme-to-phoneme decoder
        //   voices/<voice>.json   ← per-voice style vectors (57 voices)
        //   config.json, pipeline_config.json, g2p_vocab.json,
        //   vocab_index.json
        //
        // Allowlist matches these exact paths so the downloader produces
        // a usable directory that VoiceModelBundleValidator.kokoroModelURL
        // and KokoroTTSService.load can find offline. Earlier the catalog
        // had no Kokoro entry at all — users could pick the engine but
        // never download it through the app.
        let kokoroDest = docs.appendingPathComponent("VoiceModels/Kokoro")
        models.append(DownloadableModel(
            id: "kokoro-82m",
            displayName: "Kokoro-82M",
            subtitle: "aufklarer/Kokoro-82M-CoreML · 57 voices, 9 languages",
            sizeLabel: "~100 MB",
            category: .voice,
            docURL: "https://huggingface.co/aufklarer/Kokoro-82M-CoreML",
            downloader: HFModelDownloadManager(
                repoID: "aufklarer/Kokoro-82M-CoreML",
                destination: kokoroDest,
                fileAllowlist: [
                    "kokoro_5s.mlmodelc/",
                    "G2PEncoder.mlmodelc/",
                    "G2PDecoder.mlmodelc/",
                    "voices/",
                    "config.json",
                    "pipeline_config.json",
                    "g2p_vocab.json",
                    "vocab_index.json",
                ]
            ),
            approxRAMBytes: 400_000_000,
            supportedVoiceEngine: .kokoro
        ))

        // ── 6. Whisper.cpp base.en — on-device English STT ────────────────────
        // Replaces SFSpeechRecognizer's network-dependent path for users who
        // download this model. ~142 MB, ~3× realtime on A17 Pro. Lives in
        // HFModels/ggerganov_whisper.cpp/ so WhisperModelCatalog.installedModelPath
        // can resolve it via the standard staged-directory probe.
        let whisperDest = docs
            .appendingPathComponent("HFModels")
            .appendingPathComponent("ggerganov_whisper.cpp")
        models.append(DownloadableModel(
            id: "whisper-base-en",
            displayName: "Whisper base.en",
            subtitle: "ggerganov/whisper.cpp · English STT, on-device",
            sizeLabel: "~142 MB",
            category: .voice,
            docURL: "https://huggingface.co/ggerganov/whisper.cpp",
            downloader: HFModelDownloadManager(
                repoID: "ggerganov/whisper.cpp",
                destination: whisperDest,
                fileAllowlist: ["ggml-base.en.bin"]
            )
        ))

        // ── 6. Whisper.cpp base (multilingual) — 99-language STT ──────────────
        // Slightly larger (~142 MB same as english-only base), but covers 99
        // languages so users who dictate in non-English can pick this instead.
        // Lives in the same staged dir; WhisperModelCatalog falls back to
        // whichever .bin it finds first.
        models.append(DownloadableModel(
            id: "whisper-base-multi",
            displayName: "Whisper base (multilingual)",
            subtitle: "ggerganov/whisper.cpp · 99 languages, on-device",
            sizeLabel: "~142 MB",
            category: .voice,
            docURL: "https://huggingface.co/ggerganov/whisper.cpp",
            downloader: HFModelDownloadManager(
                repoID: "ggerganov/whisper.cpp",
                destination: whisperDest,
                fileAllowlist: ["ggml-base.bin"]
            )
        ))
    }

    // MARK: - Deletion

    /// Single funnel for deleting a model from the UI. Guards against
    /// dangling "active model" selections: when the model is the active
    /// assistant / camera / voice pick, the owning service is unloaded and
    /// the selection reset to its default before files are removed, so no
    /// stale id survives the delete. Failures surface as a toast instead
    /// of being swallowed by `try?`.
    func handleDeletion(of model: DownloadableModel) {
        resetActiveSelections(for: model)
        do {
            try model.delete()
        } catch {
            ToastCenter.shared.error("Couldn't delete \(model.displayName)",
                                      detail: error.localizedDescription)
            return
        }
        // Custom (HF-searched / imported) models also leave the catalog —
        // built-in entries stay so the user can re-download.
        if !model.isRequired {
            unregisterCustom(repoID: model.id)
        }
        HapticManager.impact(.medium)
        ToastCenter.shared.info("Deleted \(model.displayName)")
        refreshStorageStats()
    }

    /// Clears any active selection pointing at `model` (back to its default)
    /// and unloads the matching service so deleted weights aren't left
    /// mapped in memory.
    private func resetActiveSelections(for model: DownloadableModel) {
        let settings = AppSettings.shared

        let assistantSelection = LocalModelRegistry.unwrapAssistantSelectionID(settings.assistantModelID)
        if assistantSelection == model.id || assistantSelection == model.sourceRepoID {
            CodingAssistantService.shared.unload()
            settings.assistantModelID = AssistantModelCatalog.presets.first?.id ?? ""
            settings.hasPickedAssistantModel = false
        }
        let voiceConversationSelection = LocalModelRegistry.unwrapAssistantSelectionID(settings.voiceConversationModelID)
        if voiceConversationSelection == model.id || voiceConversationSelection == model.sourceRepoID {
            settings.voiceConversationModelID = ""
        }

        // Camera / visual persists the canonical source repo, which can differ
        // from the catalog preset id for one-package, dual-role models.
        let visualSelection = LocalModelRegistry.storedVisionSelectionID(settings.cameraVisualModelID)
        if visualSelection == model.id
            || visualSelection == model.sourceRepoID
            || visualSelection == model.subtitle {
            if model.id == FastVLMService.modelID {
                FastVLMService.shared.unload()
            } else {
                MLXVisionService.shared.unload()
            }
            settings.cameraVisualModelID = ""        // default = FastVLM
            settings.hasPickedCameraVisualModel = false
        }

    }

    // MARK: - Refresh

    func refreshAllStates() {
        reconcileInstalledRegistry()
        for model in models { model.checkIfReady() }
        // A download may have completed or a model been removed; drop the
        // cached on-disk sizes so the Models-tab fit badges re-measure.
        MemoryAdvisor.invalidateFootprintCache()
        refreshStorageStats()
    }

    // MARK: - Convenience

    var fastvlmModel: DownloadableModel? { models.first { $0.id == FastVLMService.modelID } }
    var qwen3Model:   DownloadableModel? { models.first { $0.id == "qwen3-4b" } }

    // MARK: - Storage stats (cached, computed off-main)
    //
    // The directory walks behind these cover multi-GB trees; computing them
    // synchronously on MainActor froze the UI. The UI reads these cached
    // values; `refreshStorageStats()` recomputes them in the background on
    // the existing triggers (init / refreshAllStates / delete / cleanup /
    // download completion).

    @Published private(set) var totalStorageUsed: Int64 = 0
    @Published private(set) var orphanedDownloadBytes: Int64 = 0

    /// Recomputes `totalStorageUsed` and `orphanedDownloadBytes` off the
    /// main actor and publishes the results.
    func refreshStorageStats() {
        // Snapshot inputs on MainActor. Shared destinations (KittenTTS
        // Nano/Mini, the two Whisper variants) are deduped via the Set so
        // each unique directory is summed exactly once.
        var readyDirs = Set<String>()
        for m in models {
            guard let d = m.downloader, d.state == .ready else { continue }
            readyDirs.insert(d.destination.standardizedFileURL.path)
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        // The HubApi cache (assistant lazy-downloads + image-gen models)
        // is real model storage too — it was invisible to this number.
        // It's disjoint from the Download Center roots, so no dedupe needed.
        let hubCache = docs.appendingPathComponent("huggingface", isDirectory: true)
        let orphans = orphanedDownloadDirectories()

        Task.detached(priority: .utility) { [readyDirs, orphans] in
            let fm = FileManager.default
            var total: Int64 = 0
            for path in readyDirs {
                total += (try? fm.allocatedSizeOfDirectory(
                    at: URL(fileURLWithPath: path, isDirectory: true))) ?? 0
            }
            total += (try? fm.allocatedSizeOfDirectory(at: hubCache)) ?? 0
            let orphanBytes = orphans.reduce(Int64(0)) { sum, url in
                sum + ((try? fm.allocatedSizeOfDirectory(at: url)) ?? 0)
            } + BackgroundDownloadCoordinator.staleResumeDataBytes()
            await MainActor.run { [total, orphanBytes] in
                self.totalStorageUsed = total
                self.orphanedDownloadBytes = orphanBytes
            }
        }
    }

    // MARK: - Orphaned / partial download cleanup
    //
    // A download that's cancelled or fails partway leaves completed files on
    // disk inside its destination folder. Those folders never reach `.ready`,
    // so `scanCustomDownloads()` skips them — meaning they're invisible in the
    // catalog and there's no way to delete them from the UI. They quietly pile
    // up (a user reported ~33 GB of these leftovers). This surfaces them and
    // lets the user reclaim the space with one tap.

    /// Roots that hold one-repo-per-subdirectory model downloads. Voice models
    /// live under shared, allowlisted folders (KittenTTS/Kokoro variants share
    /// a directory) so they're deliberately excluded — partial cleanup there
    /// would risk deleting a sibling variant's files.
    private var modelStorageRoots: [URL] {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return ["HFModels", "LLMModels", "GGUFModels", "FastVLMModels"]
            .map { docs.appendingPathComponent($0, isDirectory: true) }
    }

    /// Destination paths cleanup must never touch: any model that's ready,
    /// actively downloading/enumerating, or required (e.g. FastVLM).
    private func protectedDestinationPaths() -> Set<String> {
        var protected = Set<String>()
        for m in models {
            guard let d = m.downloader else { continue }
            let keep = m.isReady || d.state.isActive || m.isRequired
            if keep {
                protected.insert(d.destination.standardizedFileURL.path)
            }
        }
        // HubApi-cache directories holding a usable model are not orphans:
        //   • any assistant preset (plus the current selection, which can be
        //     a non-preset downloaded/custom id) whose weights resolve via
        //     preStagedDirectory — the exact directory MLX loads from;
        //   • image-generation repos ImageGenerationService reports installed.
        for preset in AssistantModelCatalog.presets {
            if let staged = CodingAssistantService.preStagedDirectory(for: preset) {
                protected.insert(staged.standardizedFileURL.path)
            }
        }
        // Non-preset current selection (downloaded:/imported:/custom: ids).
        // Resolved against our own `models` rather than via
        // AssistantModelCatalog.currentSelection() — that helper reaches
        // back into ModelDownloadCenter.shared, which would re-enter the
        // singleton while init is still running.
        let storedAssistant = AppSettings.shared.assistantModelID
        if !AssistantModelCatalog.presets.contains(where: { $0.id == storedAssistant }),
           let descriptor = LocalModelRegistry.descriptor(
               forStoredAssistantID: storedAssistant, catalog: models),
           let assistantModel = descriptor.assistantModel,
           let staged = CodingAssistantService.preStagedDirectory(
               for: assistantModel) {
            protected.insert(staged.standardizedFileURL.path)
        }
        // Belt-and-suspenders: ALWAYS protect the active assistant's hub-cache
        // directories derived straight from its repoID — even when the
        // descriptor above fails to resolve (it can miss for downloaded:/custom:
        // ids while the scan is still running). Otherwise the orphan sweep
        // ("Clean up partial downloads") deletes the loaded model's weights.
        var bareRepo = storedAssistant
        for prefix in ["downloaded:", "imported:", "custom:"] where bareRepo.hasPrefix(prefix) {
            bareRepo = String(bareRepo.dropFirst(prefix.count))
        }
        let parts = bareRepo.split(separator: "/")
        if parts.count == 2 {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let hf = docs.appendingPathComponent("huggingface", isDirectory: true)
            protected.insert(hf.appendingPathComponent("models/\(bareRepo)").standardizedFileURL.path)
            protected.insert(hf.appendingPathComponent("hub/models--\(parts[0])--\(parts[1])").standardizedFileURL.path)
        }

        let imageGen = ImageGenerationService.shared
        for m in ImageGenerationService.catalog where imageGen.isInstalled(m) {
            protected.insert(imageGen.repoDirectory(m.id).standardizedFileURL.path)
        }
        return protected
    }

    /// Per-repo directories inside the HubApi cache (Documents/huggingface).
    /// Two layouts exist there:
    ///   • models/<author>/<name>        — HubApi downloadBase default
    ///   • hub/models--<author>--<name>  — HF Hub canonical cache
    private func hubCacheRepoDirectories() -> [URL] {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let hfRoot = docs.appendingPathComponent("huggingface", isDirectory: true)

        func subdirectories(of url: URL) -> [URL] {
            guard let entries = try? fm.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ) else { return [] }
            return entries.filter { entry in
                var isDir: ObjCBool = false
                return fm.fileExists(atPath: entry.path, isDirectory: &isDir)
                    && isDir.boolValue
            }
        }

        var repos: [URL] = []
        // models/<author>/<name> — repos sit one level below the author dir.
        for author in subdirectories(of: hfRoot.appendingPathComponent("models", isDirectory: true)) {
            repos.append(contentsOf: subdirectories(of: author))
        }
        repos.append(contentsOf: subdirectories(of: hfRoot.appendingPathComponent("hub", isDirectory: true)))
        return repos
    }

    /// Directories under the model storage roots that don't correspond to a
    /// ready / in-flight / required model — leftovers from a download that was
    /// cancelled or failed partway.
    func orphanedDownloadDirectories() -> [URL] {
        let fm = FileManager.default
        let protected = protectedDestinationPaths()
        var candidates: [URL] = []
        for root in modelStorageRoots {
            guard let entries = try? fm.contentsOfDirectory(
                at: root,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: .skipsHiddenFiles
            ) else { continue }
            for entry in entries {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: entry.path, isDirectory: &isDir),
                      isDir.boolValue else { continue }
                candidates.append(entry)
            }
        }
        // HubApi cache repos (Documents/huggingface) used to be invisible
        // here, so partial lazy-downloads piled up with no way to clean
        // them. Usable assistant caches and installed image-gen repos are
        // protected above.
        candidates.append(contentsOf: hubCacheRepoDirectories())

        var orphans: [URL] = []
        for entry in candidates {
            let p = entry.standardizedFileURL.path
            // Skip anything that IS a protected destination, or an
            // ancestor/descendant of one (shared-root layouts).
            let collides = protected.contains { prot in
                prot == p || prot.hasPrefix(p + "/") || p.hasPrefix(prot + "/")
            }
            if collides { continue }
            orphans.append(entry)
        }
        return orphans
    }

    /// Deletes orphaned/partial download leftovers. Returns the number of bytes
    /// freed. Safe to call anytime — in-flight and ready models are protected.
    @discardableResult
    func cleanupOrphanedDownloads() -> Int64 {
        let fm = FileManager.default
        var freed = BackgroundDownloadCoordinator.clearAllResumeData()
        for url in orphanedDownloadDirectories() {
            let size = (try? fm.allocatedSizeOfDirectory(at: url)) ?? 0
            do {
                try fm.removeItem(at: url)
                freed += size
            } catch {
                print("[ModelDownloadCenter] Failed to remove orphan \(url.lastPathComponent): \(error)")
            }
        }
        refreshStorageStats()
        return freed
    }
}

// MARK: - FileManager+BackupExclusion

// MARK: - FileManager+DirectorySize

extension FileManager {
    func allocatedSizeOfDirectory(at url: URL) throws -> Int64 {
        guard let enumerator = self.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: .skipsHiddenFiles
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            total += Int64(size)
        }
        return total
    }
}
