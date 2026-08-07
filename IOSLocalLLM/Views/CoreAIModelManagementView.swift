#if CORE_AI_SERVER_APP
import SwiftUI

struct CoreAIModelManagementView: View {
    @ObservedObject private var store = CoreAIModelStore.shared
    @ObservedObject private var downloads = CoreAIDownloadCenter.shared
    @ObservedObject private var search = HFSearchService.shared
    @ObservedObject private var tokenStore = HFTokenStore.shared
    @ObservedObject private var settings = AppSettings.shared

    @State private var showingTokenSheet = false
    @State private var searchText = ""
    @State private var searchTask: Task<Void, Never>?
    @State private var customRepo = ""
    @State private var customPath = "ios"
    @State private var showExactRepo = false
    @State private var showDocumentsImporter = false

    private var activeDownloads: [CoreAIHFDownloadManager] {
        downloads.downloads.filter { manager in
            switch manager.state {
            case .enumerating, .downloading, .installing, .paused, .failed:
                return true
            case .idle, .ready:
                return false
            }
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                installedSection
                if !activeDownloads.isEmpty {
                    activeDownloadsSection
                }
                zooSection
                hfSearchSection
                importSection
            }
            .padding(.horizontal, LASDesignTokens.pageInset)
            .padding(.top, 8)
            .padding(.bottom, 40)
        }
        .background(LASPageBackground().ignoresSafeArea())
        .navigationTitle("Models")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showingTokenSheet = true
                } label: {
                    Image(systemName: tokenStore.hasToken ? "key.fill" : "key")
                }
                .accessibilityLabel("Hugging Face token")
            }
        }
        .sheet(isPresented: $showingTokenSheet) {
            LASHFTokenSettingsView()
        }
        .onChange(of: searchText) { _, value in
            searchTask?.cancel()
            searchTask = Task {
                try? await Task.sleep(for: .milliseconds(350))
                guard !Task.isCancelled else { return }
                await search.search(query: value, filter: .all, limit: 40)
            }
        }
        .refreshable {
            store.refresh()
            downloads.removeFinished()
        }
    }

    private static func importPickedModel(from url: URL) async {
        // LocalModelDocumentPickerSession already holds the security scope
        // for open-in-place folder/package picks until this returns.
        do {
            try CoreAIModelStore.shared.importModel(from: url)
            CodingAssistantService.shared.startLoad()
            ToastCenter.shared.success("Model pack imported")
        } catch {
            ToastCenter.shared.error(error.localizedDescription)
        }
    }

    // MARK: - Installed

    private var installedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "INSTALLED")
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(installedTitle)
                            .font(.headline)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(installedDetail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer(minLength: 0)
                    statusPill(installedStatusChip, color: installedStatusChipColor)
                }

                if case .ready(_, let manifest) = store.state {
                    HStack(spacing: 8) {
                        metaChip(manifest.modelFamily)
                        metaChip("ctx \(manifest.contextWindow)")
                        metaChip(ByteCountFormatter.string(
                            fromByteCount: manifest.totalDownloadBytes,
                            countStyle: .file
                        ))
                    }

                    HStack(spacing: 8) {
                        Button {
                            CodingAssistantService.shared.startLoad()
                        } label: {
                            Label("Load", systemImage: "bolt.fill")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(LASPrimaryButtonStyle())

                        Button(role: .destructive) {
                            try? store.removeModel()
                        } label: {
                            Label("Remove", systemImage: "trash")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(LASSecondaryButtonStyle())
                    }
                } else {
                    Text("Pick a zoo pack below, search Hugging Face, or import a resource folder from Files.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .lasCard()
        }
    }

    // MARK: - Downloads

    private var activeDownloadsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "DOWNLOADS", trailing: "\(activeDownloads.count)")
            ForEach(activeDownloads) { manager in
                CoreAIDownloadRow(manager: manager)
            }
        }
    }

    // MARK: - Zoo

    private var zooSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                LASSectionLabel(title: "CORE AI ZOO")
                Spacer(minLength: 8)
                Link(destination: CoreAIZooCatalog.zooHomeURL) {
                    Label("Hub", systemImage: "arrow.up.right")
                        .font(.caption.weight(.semibold))
                }
            }
            Text("Curated iPhone packs. Official recipes use the `ios/` tree.")
                .font(.caption)
                .foregroundStyle(.secondary)

            ForEach(CoreAIZooCatalog.iphoneLanguageModels) { model in
                CoreAIZooModelCard(model: model)
            }
        }
    }

    // MARK: - HF search

    private var hfSearchSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "HUGGING FACE")
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search CoreAI packs", text: $searchText)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }
            .padding(.horizontal, 14)
            .frame(minHeight: 48)
            .glassSurface(.card, cornerRadius: 14)

            Toggle("Use saved HF token for gated repos", isOn: $settings.useHFToken)
                .font(.subheadline)

            if search.isSearching {
                ProgressView("Searching Hub…")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            } else if let error = search.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }

            ForEach(filteredSearchResults) { summary in
                CoreAIHFSearchRow(summary: summary)
            }

            DisclosureGroup(isExpanded: $showExactRepo) {
                VStack(alignment: .leading, spacing: 8) {
                    TextField("owner/repo", text: $customRepo)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.caption.monospaced())
                    TextField("Subtree · e.g. ios", text: $customPath)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.caption.monospaced())
                    Button {
                        let repo = customRepo.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !repo.isEmpty else { return }
                        let prefix = customPath.trimmingCharacters(in: .whitespacesAndNewlines)
                        downloads.manager(
                            repoID: repo,
                            pathPrefix: prefix.isEmpty ? nil : prefix,
                            displayName: repo
                        ).start()
                    } label: {
                        Text("Download exact pack")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LASPrimaryButtonStyle())
                    .disabled(customRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.top, 8)
            } label: {
                Text("Exact repo download")
                    .font(.subheadline.weight(.semibold))
            }
            .lasCard()
        }
    }

    // MARK: - Import

    private var importSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "IMPORT")
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    LocalModelDocumentPickerSession.shared.present(
                        kind: .folder,
                        onPick: { url in
                            await CoreAIModelManagementView.importPickedModel(from: url)
                        }
                    )
                } label: {
                    Label("Import resource folder", systemImage: "folder.badge.plus")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASPrimaryButtonStyle())

                Button {
                    // .aimodel is a directory bundle — must use package mode
                    // (asCopy: false). File mode with asCopy: true crashes.
                    LocalModelDocumentPickerSession.shared.present(
                        kind: .package,
                        onPick: { url in
                            await CoreAIModelManagementView.importPickedModel(from: url)
                        }
                    )
                } label: {
                    Label("Import .aimodel package", systemImage: "shippingbox")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASSecondaryButtonStyle())

                Button {
                    showDocumentsImporter = true
                } label: {
                    Label("Import from App Documents", systemImage: "internaldrive")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASSecondaryButtonStyle())

                Text("Prefer the full resource directory (metadata.json + tokenizer + .aimodel). If Files crashes on sideload builds, copy the pack into Core AI: LAS via the Files app, then use App Documents.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .lasCard()
            .sheet(isPresented: $showDocumentsImporter) {
                LocalModelDocumentsImportSheet(appDocumentsName: "Core AI: LAS") { url in
                    showDocumentsImporter = false
                    Task {
                        await CoreAIModelManagementView.importPickedModel(from: url)
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private var filteredSearchResults: [HFModelSummary] {
        let needle = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty || !search.results.isEmpty else { return [] }
        return search.results
            .filter { summary in
                let id = summary.id.lowercased()
                return id.contains("coreai")
                    || id.contains("aimodel")
                    || needle.contains("coreai")
                    || !needle.isEmpty
            }
            .prefix(10)
            .map { $0 }
    }

    private var installedTitle: String {
        store.manifest?.displayName ?? "No model installed"
    }

    private var installedDetail: String {
        switch store.state {
        case .ready(_, let manifest):
            return "\(manifest.version.prefix(8)) · ready for local API"
        case .downloading: return "Downloading…"
        case .validating: return "Validating pack…"
        case .missing: return "Download a zoo pack or import a resource folder"
        case .unavailable(let message), .failed(let message): return message
        }
    }

    private var installedStatusChip: String {
        switch store.state {
        case .ready: return "ready"
        case .downloading: return "downloading"
        case .validating: return "validating"
        case .missing: return "empty"
        case .unavailable, .failed: return "error"
        }
    }

    private var installedStatusChipColor: Color {
        switch store.state {
        case .ready: return .green
        case .downloading, .validating: return .blue
        case .missing: return .secondary
        case .unavailable, .failed: return .red
        }
    }

    private func statusPill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .foregroundStyle(color)
            .background(color.opacity(0.14), in: Capsule())
    }

    private func metaChip(_ text: String) -> some View {
        Text(text)
            .font(.caption2.monospaced())
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.secondary.opacity(0.10), in: Capsule())
    }
}

// MARK: - Zoo card

private struct CoreAIZooModelCard: View {
    let model: CoreAIZooModel
    @ObservedObject private var downloads = CoreAIDownloadCenter.shared

    private var manager: CoreAIHFDownloadManager? {
        downloads.existing(id: model.id)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.displayName)
                        .font(.headline)
                    Text(model.subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer(minLength: 0)
                Link(destination: model.hubURL) {
                    Image(systemName: "arrow.up.right.square")
                        .font(.body.weight(.semibold))
                }
                .accessibilityLabel("Open Hugging Face page")
            }

            HStack(spacing: 8) {
                Text(ByteCountFormatter.string(fromByteCount: model.approxDownloadBytes, countStyle: .file))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                if model.supportsThinking {
                    chip("thinking", .green)
                }
                if model.supportsTools {
                    chip("tools", .blue)
                }
                if let pathPrefix = model.pathPrefix {
                    chip(pathPrefix, .secondary)
                }
            }

            if let manager {
                CoreAIDownloadControls(manager: manager)
            } else {
                Button {
                    downloads.start(model: model)
                } label: {
                    Label("Download", systemImage: "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASPrimaryButtonStyle())
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard()
    }

    private func chip(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2.weight(.bold))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .foregroundStyle(color == .secondary ? Color.secondary : color)
            .background(
                (color == .secondary ? Color.secondary : color).opacity(0.12),
                in: Capsule()
            )
    }
}

// MARK: - Search row

private struct CoreAIHFSearchRow: View {
    let summary: HFModelSummary
    @ObservedObject private var downloads = CoreAIDownloadCenter.shared

    private var managerID: String {
        let prefix = summary.id.lowercased().contains("official") ? "ios" : ""
        return "hf:\(summary.id)#\(prefix)"
    }

    private var manager: CoreAIHFDownloadManager? {
        downloads.existing(id: managerID)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top) {
                Text(summary.id)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(2)
                Spacer()
                if let url = URL(string: "https://huggingface.co/\(summary.id)") {
                    Link(destination: url) {
                        Image(systemName: "arrow.up.right.square")
                    }
                }
            }
            Text("\(summary.downloads) Hub downloads")
                .font(.caption2)
                .foregroundStyle(.secondary)

            if let manager {
                CoreAIDownloadControls(manager: manager)
            } else {
                Button {
                    downloads.manager(
                        repoID: summary.id,
                        pathPrefix: summary.id.lowercased().contains("official") ? "ios" : nil,
                        displayName: summary.id
                    ).start()
                } label: {
                    Text("Download")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASPrimaryButtonStyle())
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard()
    }
}

// MARK: - Download row / controls

private struct CoreAIDownloadRow: View {
    @ObservedObject var manager: CoreAIHFDownloadManager

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(manager.displayName)
                .font(.subheadline.weight(.semibold))
            Text(manager.repoID)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
            CoreAIDownloadControls(manager: manager)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .lasCard()
    }
}

private struct CoreAIDownloadControls: View {
    @ObservedObject var manager: CoreAIHFDownloadManager

    private var isFailed: Bool {
        if case .failed = manager.state { return true }
        return false
    }

    private var showsProgress: Bool {
        manager.state.isActive
            || manager.state == .paused
            || isFailed
            || manager.progress > 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if showsProgress {
                ProgressView(value: max(0.01, manager.progress))
                    .tint(isFailed ? .red : .accentColor)

                HStack(alignment: .firstTextBaseline) {
                    Text(statusLabel)
                        .font(.caption2)
                        .foregroundStyle(isFailed ? .red : .secondary)
                        .lineLimit(2)
                    Spacer(minLength: 8)
                    if manager.totalBytes > 0 {
                        Text(byteLabel)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }

                HStack {
                    if !manager.currentFile.isEmpty {
                        Text(manager.currentFile.split(separator: "/").last.map(String.init) ?? manager.currentFile)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    Spacer()
                    if manager.speedMBps > 0.05 {
                        Text(String(format: "%.1f MB/s", manager.speedMBps))
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }

            HStack(spacing: 8) {
                switch manager.state {
                case .idle, .ready:
                    Button {
                        manager.start()
                    } label: {
                        Text(manager.state == .ready ? "Re-download" : "Download")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LASPrimaryButtonStyle())
                case .failed:
                    Button {
                        manager.resume()
                    } label: {
                        Text("Retry")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LASPrimaryButtonStyle())
                    Button("Cancel", role: .destructive) { manager.cancel() }
                        .buttonStyle(LASSecondaryButtonStyle())
                        .frame(maxWidth: .infinity)
                case .enumerating, .downloading, .installing:
                    Button("Pause") { manager.pause() }
                        .buttonStyle(LASSecondaryButtonStyle())
                        .frame(maxWidth: .infinity)
                    Button("Cancel", role: .destructive) { manager.cancel() }
                        .buttonStyle(LASSecondaryButtonStyle())
                        .frame(maxWidth: .infinity)
                case .paused:
                    Button {
                        manager.resume()
                    } label: {
                        Text("Resume")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(LASPrimaryButtonStyle())
                    Button("Cancel", role: .destructive) { manager.cancel() }
                        .buttonStyle(LASSecondaryButtonStyle())
                        .frame(maxWidth: .infinity)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusLabel: String {
        switch manager.state {
        case .enumerating, .installing:
            if !manager.statusDetail.isEmpty { return manager.statusDetail }
        case .failed:
            if !manager.statusDetail.isEmpty { return manager.statusDetail }
        default:
            break
        }
        switch manager.state {
        case .idle: return "Ready"
        case .enumerating: return "Listing Hub files…"
        case .downloading:
            return "\(manager.filesDone)/\(manager.filesTotal) files · \(Int(manager.progress * 100))%"
        case .paused: return "Paused · \(Int(manager.progress * 100))%"
        case .installing: return "Installing…"
        case .ready: return "Installed"
        case .failed(let message): return message
        }
    }

    private var byteLabel: String {
        let done = ByteCountFormatter.string(fromByteCount: manager.downloadedBytes, countStyle: .file)
        let total = ByteCountFormatter.string(fromByteCount: manager.totalBytes, countStyle: .file)
        return "\(done) / \(total)"
    }
}

#endif
