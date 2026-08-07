import SwiftUI
import Combine

/// Download surface for the server-only product. Voice and camera-specific
/// catalog entries are intentionally omitted because this target does not
/// ship those runtimes.
struct LASModelDownloadsView: View {
    @ObservedObject private var center = ModelDownloadCenter.shared
    @ObservedObject private var tokenStore = HFTokenStore.shared
    @ObservedObject private var systemStatus = SystemStatusService.shared
    @ObservedObject private var safetyMonitor = DeviceSafetyMonitor.shared
    @StateObject private var hubSearch = HFSearchService()
    @State private var searchText = ""
    @State private var searchTask: Task<Void, Never>?
    @State private var searchPending = false
    @State private var showingHFToken = false
    @State private var isImportingModel = false
    @State private var importStatus: String?
    @State private var importError: String?

    private var models: [DownloadableModel] {
        center.models
            .filter { $0.category == .assistant || $0.category == .vlm }
            .filter { model in
                guard !searchText.isEmpty else { return true }
                let query = searchText.localizedLowercase
                return model.displayName.localizedCaseInsensitiveContains(query)
                    || model.sourceRepoID.localizedCaseInsensitiveContains(query)
                    || model.subtitle.localizedCaseInsensitiveContains(query)
            }
            .sorted { lhs, rhs in
                stateRank(lhs.state) != stateRank(rhs.state)
                    ? stateRank(lhs.state) < stateRank(rhs.state)
                    : lhs.displayName < rhs.displayName
            }
    }

    var body: some View {
        VStack(spacing: 0) {
            LASDetailHeader("Models") {
                LASEmptyHeaderSlot()
            }

            LASSearchField(text: $searchText, prompt: "Search models")
                .padding(.horizontal, LASDesignTokens.pageInset)
                .padding(.bottom, LASDesignTokens.component)
                .submitLabel(.search)
                .onSubmit { runHubSearch() }
                .onChange(of: searchText) { _, query in
                    scheduleHubSearch(query)
                }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: LASDesignTokens.component) {
                    if isSearchingHub {
                        hubSearchContent
                    } else {
                        headerCard
                        memoryCard

                    if models.isEmpty {
                        ContentUnavailableView(
                            searchText.isEmpty ? "No downloadable models" : "No matching models",
                            systemImage: "shippingbox",
                            description: Text(
                                searchText.isEmpty
                                    ? "The model catalog is still loading."
                                    : "Try a different model name or repository."
                            )
                        )
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 28)
                        .glassSurface(.card, cornerRadius: 22)
                        .overlay {
                            RoundedRectangle(cornerRadius: 22, style: .continuous)
                                .stroke(LASDesignTokens.hairline, lineWidth: 1)
                        }
                    } else {
                        let active = models.filter { $0.state.isActive || $0.state == .paused }
                        let installed = models.filter { $0.isReady }
                        let available = models.filter {
                            !$0.isReady && !$0.state.isActive && $0.state != .paused
                        }

                        if !active.isEmpty {
                            LASModelSection(title: "IN PROGRESS") {
                                ForEach(active) { model in
                                    LASDownloadRow(model: model)
                                }
                            }
                        }

                        if !installed.isEmpty {
                            LASModelSection(title: "INSTALLED") {
                                ForEach(installed) { model in
                                    LASDownloadRow(model: model)
                                }
                            }
                        }

                        if !available.isEmpty {
                            LASModelSection(title: "CATALOG") {
                                ForEach(available) { model in
                                    LASDownloadRow(model: model)
                                }
                            }
                        }
                    }
                    }
                }
                .padding(.horizontal, LASDesignTokens.pageInset)
                .padding(.bottom, 32)
            }
            .scrollIndicators(.hidden)
        }
        .background(LASPageBackground())
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            center.refreshAllStates()
            systemStatus.startObserving()
        }
        .onDisappear {
            searchTask?.cancel()
            systemStatus.stopObserving()
        }
        .sheet(isPresented: $showingHFToken) {
            LASHFTokenSettingsView()
        }
        .alert(
            "Import failed",
            isPresented: Binding(
                get: { importError != nil },
                set: { if !$0 { importError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { importError = nil }
        } message: {
            Text(importError ?? "The model could not be imported.")
        }
    }

    private var isSearchingHub: Bool {
        !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    @ViewBuilder
    private var hubSearchContent: some View {
        HStack(spacing: LASDesignTokens.tight) {
            Label("HUGGING FACE", systemImage: "network")
                .font(.caption.monospaced().weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer()
            if !hubSearch.results.isEmpty {
                Text("\(hubSearch.results.count) RESULTS")
                    .font(.caption2.monospaced().weight(.semibold))
                    .foregroundStyle(.secondary)
            }
        }

        if searchPending || hubSearch.isSearching {
            VStack(spacing: LASDesignTokens.row) {
                ProgressView()
                Text("Searching Hugging Face…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 42)
            .lasCard(radius: 22, padding: LASDesignTokens.component)
        } else if let error = hubSearch.lastError {
            VStack(alignment: .leading, spacing: LASDesignTokens.row) {
                Label("Hugging Face search failed", systemImage: "wifi.exclamationmark")
                    .font(.headline)
                Text(error)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                HStack(spacing: LASDesignTokens.tight) {
                    Button("Retry", action: runHubSearch)
                        .buttonStyle(LASModelPrimaryActionStyle())
                    Button("HF token") { showingHFToken = true }
                        .buttonStyle(LASModelSecondaryActionStyle())
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .lasCard(radius: 22, padding: LASDesignTokens.component)
        } else if hubSearch.results.isEmpty {
            ContentUnavailableView(
                "No Hub models found",
                systemImage: "magnifyingglass",
                description: Text("Try a broader model name or paste owner/repository.")
            )
            .frame(maxWidth: .infinity)
            .padding(.vertical, 28)
            .lasCard(radius: 22, padding: LASDesignTokens.component)
        } else {
            ForEach(hubSearch.results) { model in
                LASHubModelRow(model: model)
            }
        }
    }

    private func scheduleHubSearch(_ rawQuery: String) {
        searchTask?.cancel()
        let query = HFSearchService.normalizedQuery(rawQuery)
        guard !query.isEmpty else {
            searchPending = false
            hubSearch.clear()
            return
        }
        searchPending = true
        searchTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .milliseconds(300))
                try Task.checkCancellation()
                await hubSearch.search(query: query, filter: .all, limit: 30)
                if !Task.isCancelled { searchPending = false }
            } catch {
                if !Task.isCancelled { searchPending = false }
            }
        }
    }

    private func runHubSearch() {
        searchTask?.cancel()
        let query = HFSearchService.normalizedQuery(searchText)
        guard !query.isEmpty else {
            searchPending = false
            hubSearch.clear()
            return
        }
        searchPending = true
        searchTask = Task { @MainActor in
            await hubSearch.search(query: query, filter: .all, limit: 30)
            if !Task.isCancelled { searchPending = false }
        }
    }

    private var headerCard: some View {
        LASModelLibraryCard(
            hasToken: tokenStore.hasToken,
            isImporting: isImportingModel,
            importStatus: importStatus,
            onImportFolder: { beginImport(.folder) },
            onImportFile: { beginImport(.file) },
            onManageToken: { showingHFToken = true }
        )
    }

    private var memoryCard: some View {
        LASModelsCapacityStrip(
            thermal: thermalLabel,
            thermalColor: thermalColor,
            availableRAM: MemoryAdvisor.availableRAM.formattedBytes,
            processBudget: systemStatus.snapshot.availableForML.formattedBytes,
            freeStorage: systemStatus.snapshot.diskFree.formattedBytes
        )
    }

    private var thermalLabel: String {
        switch safetyMonitor.effectiveThermalState {
        case .nominal: return "NOMINAL"
        case .fair: return "FAIR"
        case .serious: return "WARM"
        case .critical: return "CRITICAL"
        @unknown default: return "UNKNOWN"
        }
    }

    private var thermalColor: Color {
        switch safetyMonitor.effectiveThermalState {
        case .nominal: return .secondary
        case .fair, .serious: return .orange
        case .critical: return .red
        @unknown default: return .secondary
        }
    }

    private func beginImport(_ kind: LocalModelImportService.ImportKind) {
        importError = nil
        importStatus = nil
        LocalModelDocumentPickerSession.shared.present(
            importKind: kind,
            onPick: { url in await importModel(at: url) }
        )
    }

    private func importModel(at url: URL) async {
        isImportingModel = true
        importStatus = "Importing \(url.deletingPathExtension().lastPathComponent)…"
        defer { isImportingModel = false }
        do {
            let repoID = try await LocalModelImportService.shared.importModel(from: url)
            center.refreshAllStates()
            importStatus = "Imported \(repoID). Available in the model picker."
        } catch {
            importStatus = nil
            importError = error.localizedDescription
        }
    }

    private func stateRank(_ state: HFModelDownloadManager.DownloadState) -> Int {
        switch state {
        case .downloading: return 0
        case .enumerating: return 1
        case .paused: return 2
        case .failed: return 3
        case .idle: return 4
        case .ready: return 5
        }
    }
}

private struct LASHubModelRow: View {
    let model: HFModelSummary

    @StateObject private var downloader: HFModelDownloadManager
    @State private var estimatedSize: Int64?
    @State private var didRequestSize = false

    init(model: HFModelSummary) {
        self.model = model
        if let existing = ModelDownloadCenter.shared.existingDownloader(forRepoID: model.id) {
            _downloader = StateObject(wrappedValue: existing)
        } else {
            let documents = FileManager.default.urls(
                for: .documentDirectory,
                in: .userDomainMask
            )[0]
            let destination = documents
                .appendingPathComponent("HFModels", isDirectory: true)
                .appendingPathComponent(
                    model.id.replacingOccurrences(of: "/", with: "_")
                )
            _downloader = StateObject(
                wrappedValue: HFModelDownloadManager(
                    repoID: model.id,
                    destination: destination
                )
            )
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            HStack(alignment: .top, spacing: LASDesignTokens.row) {
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .semibold))
                    .frame(width: 42, height: 42)
                    .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 13))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
                    Text(model.modelName)
                        .font(.headline)
                        .lineLimit(2)
                    Text(model.author.isEmpty ? model.id : model.author)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    HStack(spacing: LASDesignTokens.tight) {
                        if model.isMLX { formatChip("MLX") }
                        if model.isGGUF { formatChip("GGUF") }
                        if model.isCoreML { formatChip("CORE ML") }
                        if let estimatedSize {
                            Text(estimatedSize.formattedBytes)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer(minLength: 4)
                Link(destination: model.hfURL ?? URL(string: "https://huggingface.co")!) {
                    Image(systemName: "arrow.up.right")
                        .font(.system(size: 12, weight: .semibold))
                        .frame(width: 34, height: 34)
                        .background(Color.primary.opacity(0.05), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Open on Hugging Face")
            }

            action
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard(radius: 22, padding: LASDesignTokens.component)
        .task {
            guard !didRequestSize else { return }
            didRequestSize = true
            estimatedSize = await HFSearchService.estimatedSize(for: model.id)
        }
    }

    @ViewBuilder
    private var action: some View {
        switch downloader.state {
        case .idle:
            Button {
                registerAndStart()
            } label: {
                Label("Download", systemImage: "arrow.down.circle")
            }
            .buttonStyle(LASModelPrimaryActionStyle())
        case .failed:
            Button {
                registerAndStart()
            } label: {
                Label("Retry", systemImage: "arrow.clockwise")
            }
            .buttonStyle(LASModelPrimaryActionStyle())
        case .enumerating:
            Label("Checking files…", systemImage: "magnifyingglass")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        case .downloading:
            VStack(alignment: .leading, spacing: LASDesignTokens.tight) {
                ProgressView(value: downloader.progress)
                    .tint(.primary)
                HStack {
                    Text("\(Int(downloader.progress * 100))%")
                        .font(.caption.monospaced())
                    Spacer()
                    Button("Pause") { downloader.pause() }
                        .buttonStyle(LASModelSecondaryActionStyle())
                    Button("Cancel", role: .destructive) { downloader.abandon() }
                        .font(.subheadline.weight(.semibold))
                }
            }
        case .paused:
            HStack(spacing: LASDesignTokens.tight) {
                Button("Resume") { downloader.start() }
                    .buttonStyle(LASModelPrimaryActionStyle())
                Button("Cancel", role: .destructive) { downloader.abandon() }
                    .font(.subheadline.weight(.semibold))
            }
        case .ready:
            Label("Downloaded", systemImage: "checkmark.circle.fill")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.green)
        }
    }

    private func formatChip(_ label: String) -> some View {
        Text(label)
            .font(.caption2.monospaced().weight(.semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Color.primary.opacity(0.06), in: Capsule())
    }

    private var icon: String {
        switch LocalModelRegistry.category(for: model) {
        case .assistant: "cpu"
        case .vlm: "eye"
        case .voice: "waveform"
        case .imageGen: "wand.and.stars"
        }
    }

    private func registerAndStart() {
        let category = LocalModelRegistry.category(for: model)
        guard category == .assistant || category == .vlm else {
            ToastCenter.shared.info(
                "Unsupported runtime",
                detail: "This LAS build currently serves language and vision models."
            )
            return
        }
        ModelDownloadCenter.shared.registerCustom(
            repoID: model.id,
            displayName: model.modelName,
            subtitle: model.id,
            category: category,
            sizeLabel: estimatedSize?.formattedBytes ?? "—",
            docURL: model.hfURL?.absoluteString,
            downloader: downloader
        )
        downloader.start()
        HapticManager.impact(.medium)
    }
}

private struct LASModelLibraryCard: View {
    let hasToken: Bool
    let isImporting: Bool
    let importStatus: String?
    let onImportFolder: () -> Void
    let onImportFile: () -> Void
    let onManageToken: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.component) {
            HStack(alignment: .top, spacing: LASDesignTokens.row) {
                Image(systemName: "shippingbox.fill")
                    .font(.system(size: 22, weight: .semibold))
                    .frame(width: 44, height: 44)
                    .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 14))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
                    Text("Model library")
                        .font(.title3.weight(.semibold))
                    Text("Download, resume, import, and export local model files.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            ViewThatFits {
                HStack(spacing: LASDesignTokens.tight) {
                    importMenu
                    tokenButton
                }
                VStack(alignment: .leading, spacing: LASDesignTokens.tight) {
                    importMenu
                    tokenButton
                }
            }

            Text("Files → On My iPhone → On Device: LAS")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)

            if let importStatus {
                Label(
                    importStatus,
                    systemImage: isImporting ? "arrow.triangle.2.circlepath" : "checkmark.circle"
                )
                .font(.caption)
                .foregroundStyle(isImporting ? Color.secondary : Color.green)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard(radius: 22, padding: LASDesignTokens.component)
    }

    private var importMenu: some View {
        Menu {
            Button("Import model folder from Files", systemImage: "folder.badge.plus", action: onImportFolder)
            Button("Import complete model file from Files", systemImage: "doc.badge.plus", action: onImportFile)
        } label: {
            Label("Import model", systemImage: "folder.badge.plus")
        }
        .buttonStyle(LASModelSecondaryActionStyle())
        .disabled(isImporting)
        .accessibilityLabel("Import model")
    }

    private var tokenButton: some View {
        Button(action: onManageToken) {
            Label(
                hasToken ? "Manage HF token" : "Add HF token",
                systemImage: hasToken ? "checkmark.shield" : "key"
            )
        }
        .buttonStyle(LASModelSecondaryActionStyle())
    }
}

private struct LASModelsCapacityStrip: View {
    let thermal: String
    let thermalColor: Color
    let availableRAM: String
    let processBudget: String
    let freeStorage: String

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            LASSectionLabel(title: "CAPACITY", trailing: thermal, trailingColor: thermalColor)
            ViewThatFits {
                HStack(spacing: LASDesignTokens.tight) {
                    capacityItems
                }
                LazyVGrid(
                    columns: [GridItem(.flexible()), GridItem(.flexible())],
                    spacing: LASDesignTokens.tight
                ) {
                    capacityItems
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard(radius: 20, padding: 14)
    }

    @ViewBuilder
    private var capacityItems: some View {
        LASCapacityItem(title: "Thermal", value: thermal, symbol: "thermometer.medium", color: thermalColor)
        LASCapacityItem(title: "Available", value: availableRAM, symbol: "memorychip")
        LASCapacityItem(title: "Process budget", value: processBudget, symbol: "gauge.with.dots.needle.67percent")
        LASCapacityItem(title: "Storage", value: freeStorage, symbol: "internaldrive")
    }
}

private struct LASCapacityItem: View {
    let title: String
    let value: String
    let symbol: String
    var color: Color = .secondary

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
            Image(systemName: symbol)
                .foregroundStyle(color)
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(verbatim: value)
                .font(.caption.weight(.semibold).monospaced())
                .lineLimit(1)
                .minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, minHeight: 70, alignment: .leading)
        .padding(LASDesignTokens.tight)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct LASModelSection<Content: View>: View {
    let title: LocalizedStringKey
    private let content: Content

    init(title: LocalizedStringKey, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            LASSectionLabel(title: title)
        VStack(spacing: LASDesignTokens.tight) {
            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct LASModelPrimaryActionStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        LASThemedActionLabel(configuration: configuration)
    }

    private struct LASThemedActionLabel: View {
        let configuration: Configuration
        @Environment(\.koduTheme) private var T

        var body: some View {
            configuration.label
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, 14)
                .frame(minHeight: 44)
                .background(
                    T.accentStrong.opacity(configuration.isPressed ? 0.78 : 1),
                    in: RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
                )
                .overlay {
                    RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
                        .stroke(Color.white.opacity(0.16), lineWidth: 1)
                }
                .scaleEffect(configuration.isPressed ? 0.98 : 1)
                .animation(.easeOut(duration: 0.16), value: configuration.isPressed)
        }
    }
}

private struct LASModelSecondaryActionStyle: ButtonStyle {
    var foreground: Color = .primary

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(foreground)
            .padding(.horizontal, 14)
            .frame(minHeight: 44)
            .background(
                Color.primary.opacity(configuration.isPressed ? 0.10 : 0.045),
                in: RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
                    .stroke(Color.primary.opacity(0.10), lineWidth: 1)
            }
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .animation(.easeOut(duration: 0.16), value: configuration.isPressed)
    }
}

private struct LASDownloadRow: View {
    @ObservedObject var model: DownloadableModel
    @ObservedObject private var modelService = CodingAssistantService.shared
    @StateObject private var observer: LASDownloadObserver
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var deleteError: String?
    @State private var showingCancelConfirmation = false
    @State private var showingDeleteConfirmation = false
    @State private var exportDirectory: URL?

    init(model: DownloadableModel) {
        self.model = model
        _observer = StateObject(wrappedValue: LASDownloadObserver(model: model))
    }

    /// Active/paused downloads use the crystallize card; everything else keeps
    /// the original catalog row so load/delete/export paths stay unchanged.
    private var showsCrystallizeProgress: Bool {
        !reduceMotion && (observer.state.isActive || observer.state == .paused)
    }

    var body: some View {
        Group {
            if showsCrystallizeProgress {
                crystallizeRow
            } else {
                standardRow
            }
        }
        .confirmationDialog(
            "Cancel this download?",
            isPresented: $showingCancelConfirmation,
            titleVisibility: .visible
        ) {
            Button("Cancel download", role: .destructive) {
                model.cancelDownload()
                ToastCenter.shared.info("Download cancelled", detail: "Partial files were removed.")
            }
            Button("Keep download", role: .cancel) {}
        } message: {
            Text("The saved checkpoint and all partial files for this model will be removed.")
        }
        .confirmationDialog(
            "Delete \(model.displayName)?",
            isPresented: $showingDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete model", role: .destructive, action: deleteModel)
            Button("Keep model", role: .cancel) {}
        } message: {
            Text("The downloaded model files will be removed from this device.")
        }
        .sheet(
            isPresented: Binding(
                get: { exportDirectory != nil },
                set: { if !$0 { exportDirectory = nil } }
            )
        ) {
            if let exportDirectory {
                LocalModelExportPicker(
                    modelDirectory: exportDirectory,
                    onComplete: {
                        self.exportDirectory = nil
                        ToastCenter.shared.success(
                            "Model exported",
                            detail: "Available in Files for copying or sharing."
                        )
                    },
                    onCancel: { self.exportDirectory = nil }
                )
            }
        }
        .alert(
            "Could not delete model",
            isPresented: Binding(
                get: { deleteError != nil },
                set: { if !$0 { deleteError = nil } }
            )
        ) {
            Button("OK", role: .cancel) { deleteError = nil }
        } message: {
            Text(deleteError ?? "The model could not be deleted.")
        }
    }

    /// Crystallize progress card + the same pause/resume/cancel controls.
    private var crystallizeRow: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            CrystallizeDownloadCard(
                isDownloading: Binding(
                    get: { observer.state.isActive || observer.state == .paused },
                    set: { _ in }
                ),
                progress: Binding(
                    get: { min(max(observer.progress, 0), 1) * 100 },
                    set: { _ in }
                ),
                speedMBps: Binding(
                    get: { observer.speedMBps },
                    set: { _ in }
                ),
                etaSeconds: Binding(
                    get: { observer.etaSeconds },
                    set: { _ in }
                ),
                fileName: crystallizeFileName,
                fileSize: model.sizeLabel
            )
            .frame(maxWidth: .infinity)

            HStack(alignment: .firstTextBaseline) {
                Text(statusText)
                    .font(.caption)
                    .foregroundStyle(statusColor)
                Spacer(minLength: 8)
                if observer.state == .paused {
                    Text("checkpoint saved")
                        .font(.caption.monospaced())
                        .foregroundStyle(.orange)
                }
            }

            HStack(spacing: LASDesignTokens.tight) {
                actionButton
                Spacer(minLength: 0)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Original catalog row (idle / installed / failed / Reduce Motion).
    private var standardRow: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            HStack(alignment: .top, spacing: LASDesignTokens.row) {
                Image(systemName: model.category == .vlm ? "eye" : "cpu")
                    .font(.system(size: 19, weight: .semibold))
                    .foregroundStyle(model.category == .vlm ? .purple : .blue)
                    .frame(width: 40, height: 40)
                    .background(Color.primary.opacity(0.045), in: RoundedRectangle(cornerRadius: 12))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.body.weight(.semibold))
                        .lineLimit(2)
                    Text(model.sourceRepoID)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                    Text(statusText)
                        .font(.caption)
                        .foregroundStyle(statusColor)
                }
                Spacer(minLength: 4)
                if observer.state == .ready {
                    modelMenu
                } else if observer.state != .idle {
                    statusIcon
                }
            }

            if observer.state.isActive || observer.state == .paused {
                LASDotDownloadProgress(
                    progress: observer.progress,
                    isActive: observer.state.isActive,
                    isPaused: observer.state == .paused
                )

                HStack {
                    Text(progressDetail)
                    Spacer()
                    if observer.state == .paused {
                        Text("saved")
                            .foregroundStyle(.orange)
                    }
                }
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            }

            HStack(spacing: LASDesignTokens.tight) {
                if observer.state == .ready {
                    if isResident {
                        Button("Loaded", systemImage: "checkmark.circle.fill") {}
                            .buttonStyle(LASModelSecondaryActionStyle())
                            .disabled(true)
                    } else {
                        Button("Load model", systemImage: "play.circle.fill", action: loadInstalledModel)
                            .buttonStyle(LASModelPrimaryActionStyle())
                    }
                } else {
                    actionButton
                }
                Spacer(minLength: 0)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard(radius: 20, padding: 14)
    }

    private var crystallizeFileName: String {
        let file = observer.currentFile.trimmingCharacters(in: .whitespacesAndNewlines)
        if !file.isEmpty { return file }
        // Prefer a path-looking leaf from the repo id when no active file yet.
        if let leaf = model.sourceRepoID.split(separator: "/").last, !leaf.isEmpty {
            return String(leaf)
        }
        return model.displayName
    }

    @ViewBuilder
    private var actionButton: some View {
        switch observer.state {
        case .idle, .failed:
            Button {
                model.start()
            } label: {
                Label(
                    observer.isFailure ? "Retry" : "Download",
                    systemImage: observer.isFailure
                        ? "arrow.clockwise"
                        : "arrow.down.circle"
                )
            }
            .buttonStyle(LASModelPrimaryActionStyle())

        case .paused:
            HStack(spacing: 10) {
                Button {
                    model.resume()
                } label: {
                    Label("Resume", systemImage: "play.circle.fill")
                }
                .buttonStyle(LASModelPrimaryActionStyle())

                Button("Cancel", role: .destructive) {
                    showingCancelConfirmation = true
                }
                .buttonStyle(LASModelSecondaryActionStyle(foreground: .red))
            }

        case .enumerating, .downloading:
            Button {
                model.pause()
            } label: {
                Label("Pause", systemImage: "pause.circle.fill")
            }
            .buttonStyle(LASModelSecondaryActionStyle())

        case .ready:
            EmptyView()
        }
    }

    private var modelMenu: some View {
        Menu {
            if model.downloader?.destination != nil {
                Button("Export to Files", systemImage: "square.and.arrow.up") {
                    exportDirectory = model.downloader?.destination
                }
            }
            Button("Delete model", systemImage: "trash", role: .destructive) {
                showingDeleteConfirmation = true
            }
        } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 18, weight: .semibold))
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Model actions for \(model.displayName)")
    }

    private var statusIcon: some View {
        Group {
            switch observer.state {
            case .ready:
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
            case .downloading:
                ProgressView().controlSize(.small)
            case .enumerating:
                Image(systemName: "magnifyingglass").foregroundStyle(.orange)
            case .paused:
                Image(systemName: "pause.circle.fill").foregroundStyle(.orange)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
            case .idle:
                Image(systemName: "arrow.down.circle").foregroundStyle(.secondary)
            }
        }
        .frame(width: 44, height: 44)
        .accessibilityHidden(true)
    }

    private var statusText: String {
        switch observer.state {
        case .idle: return "Not downloaded · \(model.sizeLabel)"
        case .enumerating: return "Preparing file list…"
        case .downloading: return "Downloading \(observer.currentFile)"
        case .paused: return "Paused — resume keeps the saved checkpoint"
        case .ready: return "Installed · \(model.sizeLabel)"
        case .failed(let message): return "Failed: \(message)"
        }
    }

    private var progressDetail: String {
        if observer.totalBytes > 0 {
            return "\(observer.downloadedBytes.formattedBytes) / \(observer.totalBytes.formattedBytes)"
        }
        return "\(Int(observer.progress * 100))%"
    }

    private var statusColor: Color {
        switch observer.state {
        case .ready: return .green
        case .failed: return .red
        case .paused: return .orange
        case .downloading, .enumerating: return .blue
        case .idle: return .secondary
        }
    }

    private var isResident: Bool {
        guard modelService.activeModel.repoID.caseInsensitiveCompare(model.sourceRepoID) == .orderedSame else {
            return false
        }
        switch modelService.state {
        case .ready, .generating: return true
        case .unloaded, .loading, .failed: return false
        }
    }

    private func loadInstalledModel() {
        guard let assistant = LocalModelRegistry
            .descriptor(for: model, forcedRole: .assistant)
            .assistantModel else {
            ToastCenter.shared.error(
                "Model cannot be loaded",
                detail: "The installed files do not describe a supported text model."
            )
            return
        }
        modelService.startSwitchTo(assistant)
        ToastCenter.shared.info("Loading model", detail: model.displayName)
    }

    private func deleteModel() {
        do {
            try model.delete()
        } catch {
            deleteError = error.localizedDescription
        }
    }
}

private struct LASDotDownloadProgress: View {
    let progress: Double
    let isActive: Bool
    let isPaused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.koduTheme) private var T

    var body: some View {
        let doneColor = T.accentStrong
        let pendingColor = T.ink.opacity(0.35)
        TimelineView(
            .animation(
                minimumInterval: 1.0 / 20.0,
                paused: reduceMotion || !isActive
            )
        ) { timeline in
            Canvas { context, size in
                let count = 34
                let step = size.width / CGFloat(max(count - 1, 1))
                let completed = Int((progress * Double(count)).rounded(.down))
                let pulse = 0.65 + 0.35 * sin(timeline.date.timeIntervalSinceReferenceDate * 3.5)
                for index in 0..<count {
                    let radius: CGFloat = index == completed && isActive ? 3.8 : 3.0
                    let rect = CGRect(
                        x: CGFloat(index) * step - radius,
                        y: size.height / 2 - radius,
                        width: radius * 2,
                        height: radius * 2
                    )
                    let isDone = index < completed
                    let color = isPaused
                        ? Color.orange
                        : (isDone ? doneColor : pendingColor)
                    context.fill(
                        Path(ellipseIn: rect),
                        with: .color(color.opacity(index == completed ? pulse : 0.95))
                    )
                }
            }
        }
        .frame(height: 14)
        .accessibilityLabel("\(Int(progress * 100)) percent downloaded")
    }
}

@MainActor
private final class LASDownloadObserver: ObservableObject {
    @Published private(set) var state: HFModelDownloadManager.DownloadState = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var currentFile = ""
    @Published private(set) var speedMBps: Double = 0
    @Published private(set) var etaSeconds: Int?

    private let model: DownloadableModel
    private var cancellables: Set<AnyCancellable> = []
    private var lastSampleAt: Date?
    private var lastSampleBytes: Int64 = 0

    init(model: DownloadableModel) {
        self.model = model
        sync()
        model.downloader?.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.sync() }
            .store(in: &cancellables)
    }

    var isFailure: Bool {
        if case .failed = state { return true }
        return false
    }

    private func sync() {
        state = model.state
        progress = model.progress
        downloadedBytes = model.downloadedBytes
        totalBytes = model.totalBytes
        currentFile = model.currentFile
        updateThroughput()
    }

    /// Best-effort speed / ETA from byte deltas. Display-only; never drives
    /// download control flow.
    private func updateThroughput() {
        guard state == .downloading else {
            lastSampleAt = nil
            lastSampleBytes = 0
            speedMBps = 0
            etaSeconds = nil
            return
        }

        let now = Date()
        guard let lastSampleAt else {
            self.lastSampleAt = now
            lastSampleBytes = downloadedBytes
            return
        }

        let elapsed = now.timeIntervalSince(lastSampleAt)
        guard elapsed >= 0.45 else { return }

        let deltaBytes = downloadedBytes - lastSampleBytes
        self.lastSampleAt = now
        lastSampleBytes = downloadedBytes

        guard deltaBytes > 0, elapsed > 0 else {
            speedMBps = 0
            etaSeconds = nil
            return
        }

        let bytesPerSecond = Double(deltaBytes) / elapsed
        speedMBps = bytesPerSecond / (1024.0 * 1024.0)

        if totalBytes > downloadedBytes, bytesPerSecond > 1 {
            let remaining = Double(totalBytes - downloadedBytes)
            etaSeconds = max(1, Int((remaining / bytesPerSecond).rounded()))
        } else if progress >= 1 {
            etaSeconds = nil
            speedMBps = 0
        } else {
            etaSeconds = nil
        }
    }
}
