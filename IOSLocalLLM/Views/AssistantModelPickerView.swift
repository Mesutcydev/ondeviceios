import SwiftUI

// MARK: - AssistantModelPickerView
// Bottom-sheet picker that lets the user swap which model the chat tab uses.
// Reachable from the Assistant tab's model name pill in the status bar.

struct AssistantModelPickerView: View {

    let downloadedOnly: Bool

    @ObservedObject private var assistant = CodingAssistantService.shared
    @ObservedObject private var center = ModelDownloadCenter.shared
    @ObservedObject private var registry = InstalledModelRegistry.shared
    @ObservedObject private var safety = DeviceSafetyMonitor.shared
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.dismiss) private var dismiss
    @Environment(\.koduTheme) private var T

    @State private var selectedID: String = AssistantModelCatalog.currentSelection().id
    @State private var customRepoID: String = ""
    @State private var showApplePrivateCloudDisclosure = false
    @State private var isImporting = false
    @State private var isActivating = false

    init(downloadedOnly: Bool = false) {
        self.downloadedOnly = downloadedOnly
    }

    private var recommendedAssistant: AssistantModel? {
        AssistantModelCatalog.model(forID: DeviceTierAdvisor.recommendedModelID)
    }

    /// Models that have actually been pulled onto the device — catalog-pipeline
    /// downloads (center.models) AND community/imported models from the installed
    /// registry. Built-in catalog presets are excluded so the section doesn't
    /// double up. Uses explicit engine/capability/validation checks instead of
    /// fragile name-based heuristics like `.contains("4bit")`.
    private var downloadedModels: [DownloadableModel] {
        // 1. Catalog-pipeline downloads (already center.models)
        let catalogReady = center.models.filter { m in
            guard m.isReady else { return false }
            guard !m.isRequired else { return false }
            guard m.category == .assistant else { return false }
            return !AssistantModelCatalog.presets.contains {
                $0.repoID.caseInsensitiveCompare(m.sourceRepoID) == .orderedSame
            }
        }

        var seen = Set<String>()
        var result = catalogReady.filter { seen.insert($0.id).inserted }

        // 2. Registry-only models not in the catalog pipeline. These are
        //    HF Search results, local imports, or community repos that
        //    `buildCatalog()` doesn't have a preset for. Without this merge,
        //    a successfully-downloaded community model with valid files
        //    would be invisible in the picker.
        for rec in registry.records {
            guard seen.insert(rec.repoID).inserted else { continue }
            guard rec.validationState.isActivatable else { continue }
            guard rec.engine == .mlx || rec.engine == .llamaCpp else { continue }
            guard !AssistantModelCatalog.presets.contains(where: {
                $0.repoID.caseInsensitiveCompare(rec.repoID) == .orderedSame
            }) else { continue }

            let downloader = center.existingDownloader(forRepoID: rec.repoID)
                ?? HFModelDownloadManager(repoID: rec.repoID, destination: rec.localURL)
            downloader.checkIfReady()
            let wrapper = DownloadableModel(
                id: rec.repoID,
                displayName: rec.displayName,
                subtitle: String("\(rec.engine) · \(rec.quantization ?? "unknown") · \(rec.downloadBytes > 0 ? rec.downloadBytes.formattedBytes : "on disk")"),
                sizeLabel: rec.downloadBytes > 0 ? rec.downloadBytes.formattedBytes : "—",
                category: .assistant,
                isRequired: false,
                docURL: "https://huggingface.co/\(rec.repoID)",
                downloader: downloader,
                capabilities: rec.capabilities,
                runtime: rec.engine
            )
            result.append(wrapper)
        }

        return result
    }

    /// Complete downloaded-only list used by API/server surfaces. Unlike the
    /// regular picker section, this includes downloaded catalog presets too.
    /// The installed registry is authoritative: a valid on-disk preset can
    /// exist even when its catalog downloader has not republished `.ready`
    /// yet (the source of valid Bonsai/Nanbeige/Ornith installs disappearing
    /// from the in-conversation picker).
    private var downloadedOnlyAssistantModels: [AssistantModel] {
        var seen = Set<String>()
        var result: [AssistantModel] = []

        for record in registry.records {
            guard record.validationState.isActivatable else { continue }
            guard record.engine == .mlx || record.engine == .llamaCpp else { continue }
            guard LocalModelRegistry.category(
                repoID: record.repoID,
                pipelineTag: nil
            ) == .assistant else {
                continue
            }

            let key = record.repoID.lowercased()
            guard seen.insert(key).inserted else { continue }
            if let preset = AssistantModelCatalog.presets.first(where: {
                $0.repoID.caseInsensitiveCompare(record.repoID) == .orderedSame
            }) {
                result.append(preset)
            } else {
                let downloader = center.existingDownloader(forRepoID: record.repoID)
                    ?? HFModelDownloadManager(
                        repoID: record.repoID,
                        destination: record.localURL
                    )
                downloader.checkIfReady()
                let wrapper = DownloadableModel(
                    id: record.repoID,
                    displayName: record.displayName,
                    subtitle: "\(record.engine) · \(record.quantization ?? "unknown") · \(record.downloadBytes > 0 ? record.downloadBytes.formattedBytes : "on disk")",
                    sizeLabel: record.downloadBytes > 0
                        ? record.downloadBytes.formattedBytes
                        : "—",
                    category: .assistant,
                    isRequired: false,
                    docURL: "https://huggingface.co/\(record.repoID)",
                    downloader: downloader,
                    capabilities: record.capabilities,
                    runtime: record.engine
                )
                result.append(assistantModel(from: wrapper))
            }
        }

        // Catalog state still contributes ready models that have not reached
        // the registry yet during a just-finished download.
        for downloadable in center.models
        where downloadable.isReady && downloadable.category == .assistant {
            let key = downloadable.sourceRepoID.lowercased()
            guard seen.insert(key).inserted else { continue }
            if let preset = AssistantModelCatalog.presets.first(where: {
                $0.repoID.caseInsensitiveCompare(downloadable.sourceRepoID) == .orderedSame
            }) {
                result.append(preset)
            } else {
                result.append(assistantModel(from: downloadable))
            }
        }
        return result.sorted {
            let lhsActive = $0.id == assistant.activeModel.id
            let rhsActive = $1.id == assistant.activeModel.id
            if lhsActive != rhsActive { return lhsActive }
            return $0.displayName.localizedCaseInsensitiveCompare($1.displayName)
                == .orderedAscending
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    header
                    if ApplePrivateCloud.isSupportedOnCurrentOS {
                        ApplePrivateCloudPickerSection(
                            status: assistant.applePrivateCloudStatus,
                            isActive: assistant.activeExecutionLocation == .applePrivateCloud,
                            isDefault: settings.assistantModelID == ApplePrivateCloud.modelID,
                            isActivating: isActivating,
                            showsSetDefault: downloadedOnly,
                            reasoningLevel: settings.applePCCReasoningLevel,
                            onSelect: chooseApplePrivateCloud,
                            onSetDefault: setApplePrivateCloudAsDefault,
                            onReasoningChange: {
                                settings.applePCCReasoningLevel = $0
                            },
                            onShowLimitOptions: ApplePrivateCloud.showLimitIncreaseOptions
                        )
                    }
                    if downloadedOnly {
                        if downloadedOnlyAssistantModels.isEmpty {
                            VStack(spacing: 18) {
                                ContentUnavailableView(
                                    "No downloaded assistant models",
                                    systemImage: "internaldrive",
                                    description: Text(
                                        "Download an assistant model from the Models tab first."
                                    )
                                )
                                Button {
                                    dismiss()
                                    Task { @MainActor in
                                        try? await Task.sleep(for: .milliseconds(250))
                                        AppBridge.shared.requestTab(.models)
                                    }
                                } label: {
                                    Label("Browse Models", systemImage: "cube.box")
                                        .font(T.sans(15, .semibold))
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 13)
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(T.accent)
                                .padding(.horizontal, 24)
                            }
                            .padding(.top, 48)
                        } else {
                            modelSection(title: "Downloaded") {
                                LazyVStack(spacing: 12) {
                                    ForEach(downloadedOnlyAssistantModels) { model in
                                        row(for: model)
                                    }
                                }
                            }
                        }
                    } else {
                        if let warning = safety.statusLabel,
                           safety.thermalState != .nominal || safety.lowPowerMode {
                            thermalBanner(text: warning)
                        }
                        if let recommendedAssistant {
                            recommendedSection(model: recommendedAssistant)
                        }
                        // Models the user has already pulled onto the device sit
                        // directly under "recommended" — surfacing what's ready to
                        // switch to instantly, above the full preset catalog.
                        if !downloadedModels.isEmpty {
                            downloadedSection
                        }
                        presetsSection
                        importSection
                        customSection
                    }
                }
                .padding(.bottom, 32)
            }
            .sheet(isPresented: $showApplePrivateCloudDisclosure) {
                ApplePrivateCloudPrivacyDisclosureView {
                    settings.applePCCPrivacyConsentVersion =
                        ApplePrivateCloud.privacyDisclosureVersion
                    activateApplePrivateCloud()
                }
            }
            .background(LiquidPinkBackdrop())
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.foregroundColor(T.ink)
                }
            }
            .onAppear {
                selectedID = assistant.activeSelectionID
                if ApplePrivateCloud.isSupportedOnCurrentOS {
                    Task { await assistant.refreshApplePrivateCloudStatus() }
                }
            }
        }
    }

    private func chooseApplePrivateCloud() {
        guard !isActivating else { return }
        HapticManager.impact(.light)
        if settings.hasCurrentApplePCCPrivacyConsent {
            activateApplePrivateCloud()
        } else {
            showApplePrivateCloudDisclosure = true
        }
    }

    private func activateApplePrivateCloud() {
        selectedID = ApplePrivateCloud.modelID
        isActivating = true
        Task {
            let selected = await assistant.selectApplePrivateCloud(
                persistAsDefault: !downloadedOnly
            )
            isActivating = false
            if selected { dismiss() }
        }
    }

    private func setApplePrivateCloudAsDefault() {
        if settings.hasCurrentApplePCCPrivacyConsent {
            settings.assistantModelID = ApplePrivateCloud.modelID
            settings.hasPickedAssistantModel = true
            ToastCenter.shared.success(
                "Default model updated",
                detail: "\(ApplePrivateCloud.displayName) will be used for new chats."
            )
        } else {
            showApplePrivateCloudDisclosure = true
        }
    }

    // MARK: - Thermal banner

    @ViewBuilder
    private func thermalBanner(text: String) -> some View {
        let color: Color = {
            switch safety.statusColor {
            case "red":    return T.bad
            case "orange": return T.warn
            default:       return T.ink2
            }
        }()
        HStack(spacing: 10) {
            Image(systemName: "thermometer.medium")
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(color)
            VStack(alignment: .leading, spacing: 2) {
                Text(text)
                    .font(T.mono(11, .semibold))
                    .foregroundColor(color)
                KMono(
                    text: "loading a model right now may worsen the state. consider waiting.",
                    size: 9, color: T.ink3
                )
                .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 6).fill(color.opacity(0.10)))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(color.opacity(0.4), lineWidth: 1))
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            KCaption(text: "ASSISTANT")
            KPageTitle(
                title: downloadedOnly ? "Switch model" : "Choose default model",
                size: 30
            )
            Text(
                downloadedOnly
                    ? "Choose a model for this conversation, or set one as the default for new chats."
                    : "New conversations and future launches will start with this model."
            )
                .font(T.sans(13))
                .foregroundColor(T.ink2)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private func recommendedSection(model: AssistantModel) -> some View {
        modelSection(title: "Recommended") {
            Button {
                selectedID = model.id
                HapticManager.impact(.light)
                Task {
                    await assistant.switchTo(
                        model,
                        persistAsDefault: !downloadedOnly
                    )
                    dismiss()
                }
            } label: {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(T.accent)
                        .frame(width: 48, height: 48)
                        .background(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .fill(T.accentSoft)
                        )

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .center, spacing: 6) {
                            KModelName(model.displayName,
                                       font: T.display(17, .semibold),
                                       color: T.ink)
                            if model.id == assistant.activeModel.id,
                               assistant.state == .ready {
                                KActivePill(text: "active")
                            } else {
                                KActivePill(text: "recommended")
                            }
                        }
                        KFlowLayout(horizontalSpacing: 6, verticalSpacing: 5) {
                            assistantFormatPill(text: DeviceTierAdvisor.current.label)
                            if model.approxRAMBytes > 0 {
                                assistantRamPill(
                                    bytes: model.approxRAMBytes,
                                    verdict: MemoryAdvisor.verdictWithCurrentlyLoaded(for: model.id)
                                )
                            }
                        }
                        Text(DeviceTierAdvisor.rationale)
                            .font(T.sans(12))
                            .foregroundColor(T.ink2)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer(minLength: 4)

                    Image(systemName: "chevron.right")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundColor(T.accent)
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .kGlass(cornerRadius: 22, fallbackFill: T.surface)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Presets

    private var presetsSection: some View {
        let visiblePresets = AssistantModelCatalog.presets.filter { model in
            guard model.platformCompatibility?.supportsCurrentPlatform ?? true else {
                return false
            }
            switch MemoryAdvisor.fit(forFootprint: MemoryAdvisor.estimatedFootprint(for: model.id)) {
            case .fits:  return true
            case .tight: return settings.showEdgeModels
            case .over:  return false
            }
        }
        let recommendedID = recommendedAssistant?.id
        let remainingPresets = visiblePresets.filter { $0.id != recommendedID }
        return modelSection(title: "Models") {
            LazyVStack(spacing: 12) {
                ForEach(remainingPresets) { model in
                    row(for: model)
                }
            }
        }
    }

    private func modelSection<Content: View>(
        title: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                KCaption(text: title)
                Rectangle().fill(T.rule).frame(height: 1)
            }
            content()
        }
        .padding(.horizontal, 16)
        .padding(.top, 22)
    }

    @ViewBuilder
    private func row(for model: AssistantModel) -> some View {
        let isSelected = selectedID == model.id
        let verdict = MemoryAdvisor.verdictWithCurrentlyLoaded(for: model.id)
        let vendor = ModelVendor.infer(from: model.repoID)
        let footprint = MemoryAdvisor.estimatedFootprint(for: model.id)
        let avgTPS = ModelUsageTracker.shared.avgTPS(for: model.id)
        let isDefault = settings.assistantModelID == model.id

        VStack(alignment: .trailing, spacing: 8) {
            Button {
                guard !isActivating else { return }
                selectedID = model.id
                HapticManager.impact(.light)
                isActivating = true
                Task {
                    Diagnostics.shared.breadcrumb(
                        "set active tapped · preset · id=\(model.id) · repoID=\(model.repoID)",
                        category: "picker"
                    )
                    await assistant.switchTo(
                        model,
                        persistAsDefault: !downloadedOnly
                    )
                    isActivating = false
                    dismiss()
                }
            } label: {
                VStack(alignment: .leading, spacing: 12) {
                    HStack(alignment: .top, spacing: 12) {
                        KVendorThumb(vendor: vendor, size: .card)

                        VStack(alignment: .leading, spacing: 4) {
                            HStack(alignment: .center, spacing: 6) {
                                KModelName(model.displayName,
                                           font: T.display(17, .semibold),
                                           color: T.ink)
                                if model.id == assistant.activeModel.id,
                                   assistant.state == .ready {
                                    KActivePill(text: "active")
                                }
                                if isDefault {
                                    KActivePill(text: "default")
                                }
                            }
                            Text(model.subtitle)
                                .font(T.sans(12))
                                .foregroundColor(T.ink2)
                                .fixedSize(horizontal: false, vertical: true)
                            Text(model.repoID)
                                .font(T.mono(9.5))
                                .foregroundColor(T.ink3)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }

                        Spacer(minLength: 4)

                        ZStack {
                            Circle()
                                .stroke(isSelected ? T.accent : T.rule2, lineWidth: 2)
                                .frame(width: 22, height: 22)
                            if isSelected {
                                Circle()
                                    .fill(T.accent)
                                    .frame(width: 12, height: 12)
                            }
                        }
                        .padding(.top, 2)
                    }

                    KFlowLayout(horizontalSpacing: 6, verticalSpacing: 6) {
                        assistantFormatPill(text: model.runtime.label)
                        ForEach(model.displayCapabilities, id: \.rawValue) { cap in
                            KCapabilityPill(capability: cap, size: .compact)
                        }
                        if footprint > 0 {
                            assistantRamPill(bytes: footprint, verdict: verdict)
                        }
                        if let compatibility = model.platformCompatibility {
                            assistantCompatibilityPill(compatibility)
                        }
                        if let tps = avgTPS, tps > 0 {
                            assistantPerfPill(tokensPerSecond: tps)
                        }
                        ForEach(model.tags.filter { tag in
                            !model.displayCapabilities.contains { capability in
                                capability.label.caseInsensitiveCompare(tag) == .orderedSame
                            }
                        }, id: \.self) { tag in
                            KTag(text: tag, size: 9)
                        }
                    }
                }
                .padding(16)
                .frame(maxWidth: .infinity, alignment: .leading)
                .kGlass(cornerRadius: 22, fallbackFill: T.surface)
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(isSelected ? T.accent.opacity(0.35) : Color.clear,
                                lineWidth: 1)
                )
                .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
            }
            .buttonStyle(.plain)

            if downloadedOnly {
                Button {
                    settings.assistantModelID = model.id
                    settings.hasPickedAssistantModel = true
                    HapticManager.impact(.medium)
                    ToastCenter.shared.success(
                        "Default model updated",
                        detail: "\(model.displayName) will be used for new chats."
                    )
                } label: {
                    Label(
                        isDefault ? "Default model" : "Set as default",
                        systemImage: isDefault ? "star.fill" : "star"
                    )
                    .font(T.sans(12, .semibold))
                    .foregroundColor(isDefault ? T.accent : T.ink2)
                }
                .buttonStyle(.plain)
                .disabled(isDefault)
                .accessibilityHint("Use this model automatically for new conversations")
            }
        }
    }

    // MARK: - Badge helpers (parallel to VisualModelPickerView's)

    @ViewBuilder
    private func assistantFormatPill(text: String) -> some View {
        Text(text.uppercased())
            .font(T.mono(8.5, .semibold))
            .tracking(0.6)
            .foregroundColor(T.ink2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2.5)
            .background(Capsule().fill(T.ink2.opacity(0.10)))
            .overlay(Capsule().stroke(T.ink2.opacity(0.28), lineWidth: 0.5))
            .fixedSize(horizontal: true, vertical: false)
    }

    @ViewBuilder
    private func assistantRamPill(bytes: Int64, verdict: MemoryAdvisor.Verdict) -> some View {
        let color: Color = {
            switch verdict {
            case .fitsComfortably: return Color(red: 0.255, green: 0.722, blue: 0.392)
            case .marginal:        return Color(red: 0.961, green: 0.486, blue: 0.149)
            case .wontFit:         return T.bad
            }
        }()
        HStack(spacing: 4) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(bytes.formattedBytes)
                .font(T.mono(8.5, .semibold))
                .tracking(0.4)
                .foregroundColor(T.ink2)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2.5)
        .background(Capsule().fill(color.opacity(0.10)))
        .overlay(Capsule().stroke(color.opacity(0.32), lineWidth: 0.5))
        .fixedSize(horizontal: true, vertical: false)
    }

    @ViewBuilder
    private func assistantPerfPill(tokensPerSecond: Double) -> some View {
        let label = String(format: "%.0f tok/s", tokensPerSecond)
        HStack(spacing: 4) {
            Image(systemName: "bolt.fill")
                .font(.system(size: 8, weight: .bold))
            Text(label)
                .font(T.mono(8.5, .semibold))
                .tracking(0.4)
        }
        .foregroundColor(T.accent)
        .padding(.horizontal, 6)
        .padding(.vertical, 2.5)
        .background(Capsule().fill(T.accent.opacity(0.10)))
        .overlay(Capsule().stroke(T.accent.opacity(0.32), lineWidth: 0.5))
        .fixedSize(horizontal: true, vertical: false)
    }

    @ViewBuilder
    private func assistantCompatibilityPill(_ compatibility: ModelPlatformCompatibility) -> some View {
        HStack(spacing: 4) {
            Image(systemName: compatibility.symbol)
                .font(.system(size: 8, weight: .semibold))
            Text(compatibility.label)
                .font(T.mono(8.5, .semibold))
                .tracking(0.2)
        }
        .foregroundColor(T.ink2)
        .padding(.horizontal, 6)
        .padding(.vertical, 2.5)
        .background(Capsule().fill(T.ink2.opacity(0.08)))
        .overlay(Capsule().stroke(T.ink2.opacity(0.24), lineWidth: 0.5))
        .fixedSize(horizontal: true, vertical: false)
        .accessibilityHint(compatibility.detail)
    }

    // MARK: - Downloaded (HF Search + custom repos + local imports)

    private var downloadedSection: some View {
        modelSection(title: "Downloaded") {
            LazyVStack(spacing: 12) {
                ForEach(downloadedModels) { model in
                    row(for: assistantModel(from: model))
                }
            }
        }
    }

    /// Wraps a DownloadableModel into the AssistantModel shape the picker
    /// row expects. id prefix distinguishes downloaded entries from preset /
    /// custom-typed / imported ones so MemoryAdvisor and selection bookkeeping
    /// stay consistent.
    private func assistantModel(from m: DownloadableModel) -> AssistantModel {
        LocalModelRegistry
            .descriptor(for: m, forcedRole: .assistant)
            .assistantModel
            ?? LocalModelRegistry.customAssistantDescriptor(repoID: m.id,
                                                            origin: LocalModelRegistry.origin(for: m))
                .assistantModel!
    }

    // MARK: - Import from Files

    private var importSection: some View {
        KSection(title: "from_files") {
            Menu {
                Button("Import model folder from Files", systemImage: "folder.badge.plus") {
                    LocalModelDocumentPickerSession.shared.present(
                        importKind: .folder,
                        onPick: { url in await importLocal(url) }
                    )
                    HapticManager.impact(.light)
                }
                Button("Import complete model file from Files", systemImage: "doc.badge.plus") {
                    LocalModelDocumentPickerSession.shared.present(
                        importKind: .file,
                        onPick: { url in await importLocal(url) }
                    )
                    HapticManager.impact(.light)
                }
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "tray.and.arrow.down")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(T.good)
                        .frame(width: 32, height: 32)
                        .background(RoundedRectangle(cornerRadius: 6)
                            .fill(T.good.opacity(0.12)))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("import from files")
                            .font(T.mono(13, .semibold))
                            .foregroundColor(T.ink)
                        KMono(text: "load an mlx folder, .mlpackage, or .mlmodel",
                               size: 10, color: T.ink3)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    if isImporting {
                        ProgressView().tint(T.good).scaleEffect(0.7)
                    } else {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10))
                            .foregroundColor(T.ink3)
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .contentShape(Rectangle())
            }
            .menuStyle(.borderlessButton)
            .disabled(isImporting)
        }
    }

    private func importLocal(_ url: URL) async {
        isImporting = true
        defer { isImporting = false }
        do {
            let repoID = try await LocalModelImportService.shared.importModel(from: url)
            // Switch the assistant to the imported model automatically
            // Resolve through the newly registered catalog entry so its
            // on-disk runtime (.llamaCpp for GGUF) is preserved. Synthesizing
            // only from the opaque local/ id defaulted imported GGUFs to MLX.
            let model: AssistantModel
            if let entry = center.models.first(where: { $0.id == repoID }),
               let resolved = LocalModelRegistry
                    .descriptor(for: entry, forcedRole: .assistant, forcedOrigin: .imported)
                    .assistantModel {
                model = resolved
            } else {
                model = LocalModelRegistry
                    .customAssistantDescriptor(repoID: repoID, origin: .imported)
                    .assistantModel!
            }
            await assistant.switchTo(
                model,
                persistAsDefault: !downloadedOnly
            )
            dismiss()
        } catch {
            ToastCenter.shared.error("Import failed",
                                      detail: error.localizedDescription)
        }
    }

    // MARK: - Custom

    private var customSection: some View {
        KSection(title: "custom_repo") {
            VStack(alignment: .leading, spacing: 8) {
                KMono(text: "load any mlx-compatible model directly from huggingface.",
                       size: 10, color: T.ink3)
                    .fixedSize(horizontal: false, vertical: true)
                TextField("author/repo (e.g. mlx-community/...)",
                          text: $customRepoID)
                    .font(T.mono(11))
                    .foregroundColor(T.ink)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(T.surface2))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(T.rule, lineWidth: 1))
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                Button {
                    let trimmed = customRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty, trimmed.contains("/") else {
                        ToastCenter.shared.error("Invalid repo ID",
                                                  detail: "Format: author/repo-name")
                        return
                    }
                    let custom = LocalModelRegistry
                        .customAssistantDescriptor(repoID: trimmed, origin: .custom)
                        .assistantModel!
                    HapticManager.impact(.medium)
                    Task {
                        await assistant.switchTo(
                            custom,
                            persistAsDefault: !downloadedOnly
                        )
                        dismiss()
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.down.circle")
                            .font(.system(size: 11, weight: .medium))
                        Text("load this repo")
                            .font(T.mono(11, .semibold))
                    }
                    .foregroundColor(T.bg)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(T.ink))
                }
                .buttonStyle(.plain)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct ApplePrivateCloudPickerSection: View {
    let status: ApplePCCStatus
    let isActive: Bool
    let isDefault: Bool
    let isActivating: Bool
    let showsSetDefault: Bool
    let reasoningLevel: ApplePCCReasoningLevel
    let onSelect: () -> Void
    let onSetDefault: () -> Void
    let onReasoningChange: (ApplePCCReasoningLevel) -> Void
    let onShowLimitOptions: () -> Void

    @Environment(\.koduTheme) private var T

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                KCaption(text: "APPLE PRIVATE CLOUD")
                Rectangle().fill(T.rule).frame(height: 1)
            }

            VStack(alignment: .trailing, spacing: 8) {
                Button(action: onSelect) {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "apple.intelligence")
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundStyle(T.accent)
                            .frame(width: 48, height: 48)
                            .background(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .fill(T.accentSoft)
                            )

                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 6) {
                                KModelName(
                                    ApplePrivateCloud.displayName,
                                    font: T.display(17, .semibold),
                                    color: T.ink
                                )
                                if isActive { KActivePill(text: "active") }
                                if isDefault { KActivePill(text: "default") }
                            }
                            Text(ApplePrivateCloud.subtitle)
                                .font(T.sans(12))
                                .foregroundStyle(T.ink2)
                                .fixedSize(horizontal: false, vertical: true)
                            Label(statusTitle, systemImage: statusSymbol)
                                .font(T.mono(10, .semibold))
                                .foregroundStyle(statusColor)
                            Text(statusDetail)
                                .font(T.sans(11))
                                .foregroundStyle(T.ink3)
                                .fixedSize(horizontal: false, vertical: true)
                        }

                        Spacer(minLength: 4)

                        if isActivating {
                            ProgressView().controlSize(.small)
                        } else {
                            Image(systemName: "chevron.right")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundStyle(T.accent)
                        }
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .kGlass(cornerRadius: 22, fallbackFill: T.surface)
                    .contentShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(isActivating)
                .accessibilityHint("Reviews privacy disclosure before first use")

                if status == .limitReached {
                    Button("Show Options", action: onShowLimitOptions)
                        .font(T.sans(12, .semibold))
                        .buttonStyle(.plain)
                        .foregroundStyle(T.accent)
                }

                Picker(
                    "Reasoning",
                    selection: Binding(
                        get: { reasoningLevel },
                        set: onReasoningChange
                    )
                ) {
                    ForEach(ApplePCCReasoningLevel.allCases, id: \.rawValue) { level in
                        Text(reasoningLabel(level)).tag(level)
                    }
                }
                .pickerStyle(.menu)
                .font(T.sans(12, .semibold))

                if showsSetDefault {
                    Button(action: onSetDefault) {
                        Label(
                            isDefault ? "Default model" : "Set as default",
                            systemImage: isDefault ? "star.fill" : "star"
                        )
                        .font(T.sans(12, .semibold))
                        .foregroundStyle(isDefault ? T.accent : T.ink2)
                    }
                    .buttonStyle(.plain)
                    .disabled(isDefault || isActivating)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.top, 22)
    }

    private var statusTitle: String {
        switch status {
        case .ready: return "Available"
        case .approachingLimit: return "Available · nearing daily limit"
        case .limitReached: return "Daily limit reached"
        case .unsupportedOS: return "Requires iOS 27"
        case .unsupportedDevice: return "Device not eligible"
        case .appleIntelligenceUnavailable: return "Apple Intelligence unavailable"
        case .offline: return "Offline"
        case .temporarilyUnavailable: return "Temporarily unavailable"
        case .entitlementUnavailable: return "Not enabled for this build"
        case .unknown: return "Availability unknown"
        }
    }

    private func reasoningLabel(_ level: ApplePCCReasoningLevel) -> String {
        switch level {
        case .automatic: return "Automatic reasoning"
        case .light: return "Light reasoning"
        case .moderate: return "Moderate reasoning"
        case .deep: return "Deep reasoning"
        }
    }

    private var statusDetail: String {
        switch status {
        case .ready, .approachingLimit:
            return "Uses a network connection. No model download or local memory load."
        case .limitReached:
            return "Choose Show Options or continue with any downloaded local model."
        case .offline:
            return "Connect to the internet, then reopen this picker to refresh."
        default:
            return "Downloaded local models remain available."
        }
    }

    private var statusSymbol: String {
        switch status {
        case .ready: return "checkmark.circle.fill"
        case .approachingLimit: return "exclamationmark.circle.fill"
        case .limitReached: return "hourglass.circle.fill"
        case .offline: return "wifi.slash"
        default: return "info.circle.fill"
        }
    }

    private var statusColor: Color {
        switch status {
        case .ready: return T.good
        case .approachingLimit: return T.warn
        case .limitReached: return T.bad
        default: return T.ink3
        }
    }
}

private struct ApplePrivateCloudPrivacyDisclosureView: View {
    let onAccept: () -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.koduTheme) private var T

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Image(systemName: "lock.icloud.fill")
                        .font(.system(size: 38, weight: .semibold))
                        .foregroundStyle(T.accent)

                    Text("Use Apple Private Cloud")
                        .font(T.display(28, .semibold))
                        .foregroundStyle(T.ink)

                    Text(
                        "When selected, the conversation content needed for your request is sent to Apple Private Cloud Compute for processing."
                    )
                    .font(T.sans(15))
                    .foregroundStyle(T.ink2)

                    VStack(alignment: .leading, spacing: 12) {
                        disclosureRow(
                            symbol: "text.bubble",
                            text: "This can include recent messages, system instructions, and text extracted from files, web pages, or images."
                        )
                        disclosureRow(
                            symbol: "network",
                            text: "A network connection is required. Requests do not run entirely on this device."
                        )
                        disclosureRow(
                            symbol: "arrow.triangle.2.circlepath",
                            text: "You can switch back to a downloaded local model at any time."
                        )
                    }

                    Button {
                        onAccept()
                        dismiss()
                    } label: {
                        Text("Agree and Continue")
                            .font(T.sans(15, .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 13)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(T.accent)
                }
                .padding(22)
            }
            .background(LiquidPinkBackdrop())
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    private func disclosureRow(symbol: String, text: String) -> some View {
        Label {
            Text(text)
                .font(T.sans(13))
                .foregroundStyle(T.ink2)
        } icon: {
            Image(systemName: symbol)
                .foregroundStyle(T.accent)
        }
    }
}
