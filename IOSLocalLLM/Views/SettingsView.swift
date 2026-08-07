import SwiftUI

// MARK: - SettingsView

struct SettingsView: View {
    @ObservedObject private var settings   = AppSettings.shared
    @ObservedObject private var assistant: CodingAssistantService
    @ObservedObject private var fastVLM    = FastVLMService.shared
    @ObservedObject private var voiceServiceObs = VoiceService.shared
    @ObservedObject private var loc        = LocalizationService.shared
    @ObservedObject private var hfToken    = HFTokenStore.shared
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL
    @Environment(\.koduTheme) private var T

    @State private var showingHFTokenSheet: Bool = false
    #if !targetEnvironment(macCatalyst)
    /// nil = default monochrome LAS icon; "AppIconClassic" = previous pink mark.
    @State private var activeIconName: String? = UIApplication.shared.alternateIconName
    #endif
    /// Presents the onboarding model picker as a sheet so existing
    /// users (already past the original 3-page onboarding) can run
    /// the new picker without uninstalling.
    @State private var showingModelPickerSheet: Bool = false
    @State private var showingKnowledgeBase: Bool = false
    @ObservedObject private var knowledgeBase = KnowledgeBaseService.shared

    private var fastVLMStatus: FastVLMComponentStatus { fastVLM.componentStatus }
    private var fastVLMDebugInfo: FastVLMDebugInfo?   { fastVLM.debugInfo }
    /// Filled once OnDevice: AI Image Studio is live on the App Store.
    /// Keeping this nil makes the promo visible without exposing a dead link.
    private static let imageStudioAppStoreURL: URL? = nil
    private static let imageStudioWebsiteURL = URL(string: "https://mesut.uk/apps/ondeviceimage")!

    init(assistant: CodingAssistantService) {
        self.assistant = assistant
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    // Page header
                    VStack(alignment: .leading, spacing: 1) {
                        Text(loc.t("Preferences").uppercased())
                            .font(T.sans(13, .semibold)).tracking(0.7)
                            .foregroundColor(T.accent)
                        Text(loc.t("Settings"))
                            .font(T.display(32, .bold)).foregroundColor(T.ink)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.top, 8)
                    .padding(.bottom, 12)

                    // Privacy hero (design screen 06) — leads Settings with the
                    // on-device, no-account promise.
                    privacyHero
                        .padding(.horizontal, 16)
                        .padding(.bottom, 14)

                    // Top-level category index. Replaces the previous flat
                    // 13-section scroll, which dumped every preference,
                    // status panel, and diagnostic on the same screen and
                    // overwhelmed first-time users. Each row navigates to a
                    // focused sub-screen that still hosts the original
                    // KSection bodies — same content, less noise.
                    VStack(spacing: 8) {
                        ForEach(SettingsCategory.allCases) { cat in
                            NavigationLink {
                                categoryDestination(cat)
                            } label: {
                                categoryRow(cat)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 12)

                    imageStudioBanner
                        .padding(.horizontal, 16)
                        .padding(.top, 16)

                    aboutSection
                }
                .padding(.bottom, 32)
            }
            .background(LiquidPinkBackdrop())
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        dismiss()
                    } label: {
                        Text(loc.t("Done")).font(T.sans(15, .medium)).foregroundColor(T.accent)
                    }
                }
                ToolbarItem(placement: .principal) {
                    Text(loc.t("Settings")).font(T.sans(16, .semibold)).foregroundColor(T.ink)
                }
            }
            .toolbarBackground(.hidden, for: .navigationBar)
        }
        // SwiftUI sheets don't inherit `.preferredColorScheme` from the
        // root WindowGroup in iOS 18, so the appearance Picker's underlying
        // UISegmentedControl stayed in the OLD scheme after the user toggled
        // it — the unselected option washed out against the new theme bg
        // and looked greyed/un-tappable. Re-apply here so the sheet tracks
        // the same setting as the rest of the app.
        .preferredColorScheme(settings.resolvedColorScheme)
        .sheet(isPresented: $showingHFTokenSheet) {
            HFTokenSheet()
                .preferredColorScheme(settings.resolvedColorScheme)
        }
        .sheet(isPresented: $showingKnowledgeBase) {
            KnowledgeBaseView()
                .preferredColorScheme(settings.resolvedColorScheme)
        }
        .sheet(isPresented: $showingModelPickerSheet) {
            NavigationStack {
                OnboardingModelPickerView {
                    showingModelPickerSheet = false
                }
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Button("Close") { showingModelPickerSheet = false }
                            .font(T.sans(15))
                            .foregroundColor(T.ink2)
                    }
                }
                .toolbarBackground(.hidden, for: .navigationBar)
            }
            .preferredColorScheme(settings.resolvedColorScheme)
        }
    }

    // MARK: - Category index
    //
    // Top-level navigation pattern: six categories that route to focused
    // sub-screens. iOS Settings.app pattern (group → sub-page) preserves
    // discoverability while collapsing the at-a-glance complexity from
    // 13 sections to 6 rows. The sub-pages render the original KSection
    // bodies inline, so no content was lost — only the firehose.

    enum SettingsCategory: String, CaseIterable, Identifiable {
        case userGuide
        case capture
        case modelsAI
        case voice
        case appearance
        case system
        case privacyLegal

        var id: String { rawValue }

        var title: String {
            switch self {
            case .userGuide:    return "user guide"
            case .capture:      return "capture & lens"
            case .modelsAI:     return "models & ai"
            case .voice:        return "voice"
            case .appearance:   return "appearance"
            case .system:       return "system & diagnostics"
            case .privacyLegal: return "privacy & legal"
            }
        }

        var subtitle: String {
            switch self {
            case .userGuide:    return "learn how to use ioslocalllm & local models"
            case .capture:      return "analysis mode, camera, detection"
            case .modelsAI:     return "assistant, api keys, fastvlm pipeline"
            case .voice:        return "engine, voice, audio, behavior"
            case .appearance:   return "theme, language, interface"
            case .system:       return "status, thermal, developer tools"
            case .privacyLegal: return "data, reset, legal notices"
            }
        }

        var icon: String {
            switch self {
            case .userGuide:    return "book.closed"
            case .capture:      return "camera.viewfinder"
            case .modelsAI:     return "brain"
            case .voice:        return "waveform"
            case .appearance:   return "paintpalette"
            case .system:       return "gauge.with.dots.needle.bottom.50percent"
            case .privacyLegal: return "lock.shield"
            }
        }
    }

    // Privacy hero — quiet card with the on-device promise.
    private var privacyHero: some View {
        HStack(spacing: 13) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(T.surface2)
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(T.rule, lineWidth: 1))
                Image(systemName: "lock.fill")
                    .font(.system(size: 20, weight: .semibold)).foregroundColor(T.ink)
            }
            .frame(width: 46, height: 46)
            VStack(alignment: .leading, spacing: 2) {
                Text(loc.t("On-device & private"))
                    .font(T.sans(17, .bold)).foregroundColor(T.ink)
                Text(loc.t("No account. Your data never leaves this iPhone."))
                    .font(T.sans(13)).foregroundColor(T.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .kGlass(cornerRadius: 22, fallbackFill: T.surface, fallbackStroke: T.rule)
    }

    private func categoryRow(_ cat: SettingsCategory) -> some View {
        HStack(spacing: 13) {
            ZStack {
                RoundedRectangle(cornerRadius: 11, style: .continuous).fill(T.accentSoft)
                Image(systemName: cat.icon)
                    .font(.system(size: 17, weight: .regular))
                    .foregroundColor(T.accent)
            }
            .frame(width: 38, height: 38)
            VStack(alignment: .leading, spacing: 2) {
                Text(cat.title.capitalized)
                    .font(T.sans(16, .semibold)).foregroundColor(T.ink)
                Text(cat.subtitle)
                    .font(T.sans(12.5)).foregroundColor(T.ink3).lineLimit(1)
            }
            Spacer(minLength: 8)
            Image(systemName: "chevron.right")
                .font(.system(size: 12, weight: .semibold))
                .foregroundColor(T.ink4)
        }
        .padding(.horizontal, 15)
        .padding(.vertical, 13)
        .frame(maxWidth: .infinity, alignment: .leading)
        .kGlass(cornerRadius: 18, fallbackFill: T.surface, fallbackStroke: T.rule)
    }

    @ViewBuilder
    private func categoryDestination(_ cat: SettingsCategory) -> some View {
        ScrollView {
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 4) {
                    KCaption(text: "SETTINGS")
                    KPageTitle(title: cat.title.capitalized, size: 28)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 16)
                .padding(.top, 8)
                .padding(.bottom, 4)

                switch cat {
                case .userGuide:
                    UserGuideView()
                case .capture:
                    analysisModeSection
                case .modelsAI:
                    modelsSection
                    assistantSection
                    apiSettingsSection
                    fastvlmPipelineSection
                case .voice:
                    voiceSection
                case .appearance:
                    uiSection
                case .system:
                    systemStatusSection
                    thermalProtectionSection
                    developerSection
                case .privacyLegal:
                    privacyResetSection
                    legalSection
                }
            }
            .padding(.bottom, 32)
        }
        .background(LiquidPinkBackdrop())
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
    }

    // MARK: - Sections

    // MARK: - Analysis Mode Section

    private var analysisModeSection: some View {
        KSection(title: "analysis_mode") {
            VStack(alignment: .leading, spacing: 10) {
                Picker("Mode", selection: $settings.analysisMode) {
                    ForEach(AnalysisMode.allCases) { mode in
                        Label(mode.displayName, systemImage: mode.systemImage).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.segmented)

                let mode = AnalysisMode(rawValue: settings.analysisMode) ?? .code
                Text(mode == .code
                     ? "Extracts source code from the detected region and generates a code review."
                     : "Describes what the camera sees — useful for diagrams, whiteboards, UI screenshots, or any non-code scene.")
                    .font(T.sans(11))
                    .foregroundColor(T.ink2)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(14)
        }
    }

    private var modelsSection: some View {
        KSection(title: "models") {
            // Download center link — primary entry point
            NavigationLink {
                ModelDownloadCenterView()
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 16))
                        .foregroundColor(T.accent)
                        .frame(width: 32, height: 32)
                        .background(RoundedRectangle(cornerRadius: 6).fill(T.accentSoft))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Download models")
                            .font(T.sans(13, .semibold))
                            .foregroundColor(T.ink)
                        KMono(text: "FastVLM · Qwen3 · KittenTTS", size: 10, color: T.ink3, mono: false)
                    }
                    Spacer()
                    downloadStatusBadge
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)

            Rectangle().fill(T.rule).frame(height: 1)

            // Onboarding picker re-entry — existing users who already
            // passed the original 3-page onboarding never saw the new
            // model-picker step. This row presents it as a sheet so
            // they can pick fresh defaults without uninstalling.
            Button {
                showingModelPickerSheet = true
                HapticManager.impact(.light)
            } label: {
                HStack(spacing: 12) {
                    Image(systemName: "checkmark.circle.badge.questionmark")
                        .font(.system(size: 15))
                        .foregroundColor(T.accent)
                        .frame(width: 32, height: 32)
                        .background(RoundedRectangle(cornerRadius: 6).fill(T.accentSoft))
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Pick models again")
                            .font(T.sans(13, .semibold))
                            .foregroundColor(T.ink)
                        KMono(text: "Open the guided model picker", size: 10, color: T.ink3, mono: false)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)

            Rectangle().fill(T.rule).frame(height: 1)

            // Multi-GB model downloads on cellular can burn through a data
            // plan in minutes — let users restrict them to Wi-Fi. Kept on the
            // everyday card (it's a meaningful choice, not pipeline internals).
            KRow(label: "Wi-Fi only downloads", trailing: {
                KToggle(isOn: $settings.wifiOnlyDownloads)
            })

            // Power-user / pipeline internals collapsed behind "Advanced" so the
            // everyday card stays short. Nothing is removed — one tap away.
            KDisclosureRows(title: "Advanced") {
                Rectangle().fill(T.rule).frame(height: 1)
                KRow(label: "Text detector", trailing: {
                    KMono(text: "Apple Vision · System", size: 10, color: T.ink2, mono: false)
                })
                KRow(label: "Vision encoder", trailing: {
                    KMono(text: "468 MB · Neural Engine", size: 10, color: T.ink2, mono: false)
                })
                KRow(label: "Enable vision encoder", trailing: {
                    KToggle(isOn: $settings.fastVLMEnabled)
                })
                KRow(label: "Memory safety", trailing: {
                    KToggle(isOn: $settings.strictMemoryGate)
                })
                KRow(label: "MLX low-cache / GGUF paging", trailing: {
                    KToggle(isOn: $settings.largeModelLowMemoryEnabled)
                })
                KRow(label: "Load timeout", trailing: {
                    HStack(spacing: 8) {
                        KMono(text: settings.modelLoadTimeoutSeconds == 0
                                ? "Off"
                                : "\(settings.modelLoadTimeoutSeconds / 60) min",
                              size: 10, color: T.ink2, mono: false)
                        Stepper("", value: $settings.modelLoadTimeoutSeconds,
                                in: 0...1800, step: 60)
                            .labelsHidden()
                    }
                }, last: true)
            }
            Rectangle().fill(T.rule).frame(height: 1)
            // Active assistant model + picker — tap to open the Models tab
            // (the unified hub now owns this flow; the legacy picker sheet
            // is still mounted below for share-extension entry points).
            Button {
                dismiss()
                AppBridge.shared.requestTab(.models)
                HapticManager.impact(.light)
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "brain")
                        .font(.system(size: 13))
                        .foregroundColor(T.accent)
                        .frame(width: 28, height: 28)
                        .background(RoundedRectangle(cornerRadius: 6).fill(T.accentSoft))
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            KMono(
                                text: "Default assistant model",
                                size: 11,
                                color: T.ink3,
                                mono: false
                            )
                            // Surface the selected model's top marker
                            // (Recommended / Best / New) — guidance that
                            // otherwise only appears inside the picker sheet.
                            if let marker = [ModelCapability.recommended, .best, .newRelease]
                                .first(where: { AssistantModelCatalog.currentSelection().capabilities.contains($0) }) {
                                KCapabilityPill(capability: marker, size: .compact)
                            }
                        }
                        // MarqueeText keeps the row height fixed regardless
                        // of which model is selected. The settings sheet
                        // gets cramped when this row grows to two lines
                        // (it shifts the status badge + chevron and forces
                        // the row above to recalculate), so for long HF
                        // ids we slide rather than wrap.
                        MarqueeText(
                            text: AssistantModelCatalog.currentSelection().displayName,
                            font: T.mono(13, .semibold),
                            color: T.ink
                        )
                        .frame(height: 18)
                    }
                    Spacer()
                    assistantStatusBadge
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
            }
            .buttonStyle(.plain)

            // Voice model picker — parallel to assistant + visual model
            // pickers. Lets the user pick the TTS engine (System / Kitten)
            // from the same Settings → Models hub, with download and
            // health-state surfaced inline.
            Rectangle().fill(T.rule).frame(height: 1)
            Button {
                showVoiceModelPicker = true
                HapticManager.impact(.light)
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: "waveform")
                        .font(.system(size: 13))
                        .foregroundColor(T.accent)
                        .frame(width: 28, height: 28)
                        .background(RoundedRectangle(cornerRadius: 6).fill(T.accentSoft))
                    VStack(alignment: .leading, spacing: 2) {
                        KMono(text: "Voice model", size: 11, color: T.ink3, mono: false)
                        MarqueeText(
                            text: voiceModelDisplayName,
                            font: T.sans(13, .semibold),
                            color: T.ink
                        )
                        .frame(height: 18)
                    }
                    Spacer()
                    voiceModelStatusBadge
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
            }
            .buttonStyle(.plain)

            // Persona quick-switcher
            Rectangle().fill(T.rule).frame(height: 1)
            Button {
                showPersonaPicker = true
                HapticManager.impact(.light)
            } label: {
                HStack(spacing: 10) {
                    Image(systemName: PersonaStore.shared.active.icon)
                        .font(.system(size: 13))
                        .foregroundColor(PersonaStore.shared.active.accent)
                        .frame(width: 28, height: 28)
                        .background(RoundedRectangle(cornerRadius: 6)
                            .fill(PersonaStore.shared.active.accent.opacity(0.12)))
                    VStack(alignment: .leading, spacing: 2) {
                        KMono(text: "Persona", size: 11, color: T.ink3, mono: false)
                        KMono(text: PersonaStore.shared.active.name,
                               size: 13, weight: .semibold, color: T.ink, mono: false)
                    }
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 11)
            }
            .buttonStyle(.plain)

            // Conditional action below the rows
            if case .ready = assistant.state {
                VStack(spacing: 0) {
                    Rectangle().fill(T.rule).frame(height: 1)
                    KSecondaryButton(label: "Unload model (free RAM)",
                                     systemImage: "memorychip",
                                     trailing: nil,
                                     destructive: true) { assistant.unload() }
                        .padding(10)
                }
            } else if case .failed(let msg) = assistant.state {
                VStack(spacing: 0) {
                    Rectangle().fill(T.rule).frame(height: 1)
                    VStack(alignment: .leading, spacing: 8) {
                        Text(msg)
                            .font(T.sans(10))
                            .foregroundColor(T.bad)
                            .fixedSize(horizontal: false, vertical: true)
                        KSecondaryButton(label: "Retry loading model",
                                         systemImage: "arrow.clockwise",
                                         trailing: nil) {
                            Task { await assistant.load() }
                        }
                    }
                    .padding(10)
                }
            }
        }
        .sheet(isPresented: $showAssistantModelPicker) {
            AssistantModelPickerView()
        }
        .sheet(isPresented: $showVoiceModelPicker) {
            VoiceModelPickerView()
        }
        .sheet(isPresented: $showPersonaPicker) {
            PersonaPickerView()
        }
    }

    /// Display name for the currently selected voice engine — used by the
    /// "voice model" row in the models section. Parallels the assistant
    /// model name pattern.
    private var voiceModelDisplayName: String {
        let kind = VoiceEngineKind(rawValue: settings.voiceEngine) ?? .appleSystem
        switch kind {
        case .appleSystem: return "Apple System Voice"
        case .kittenTTS:   return "KittenTTS (neural)"
        case .kokoro:      return "Kokoro (neural)"
        }
    }

    /// Status badge for the voice model row. Mirrors the visual model
    /// picker's status semantics: green = ready, amber = loading, red =
    /// failed. System voice is always ready, so we read the kitten state
    /// only when the preferred engine is Kitten.
    @ViewBuilder
    private var voiceModelStatusBadge: some View {
        let kind = VoiceEngineKind(rawValue: settings.voiceEngine) ?? .appleSystem
        switch kind {
        case .appleSystem:
            KStatusBadge(glyph: .ready, label: "ready", color: T.good)
        case .kittenTTS:
            switch voiceServiceObs.kittenState {
            case .ready:    KStatusBadge(glyph: .ready, label: "ready", color: T.good)
            case .loading:  KStatusBadge(glyph: .streaming, label: "loading", color: T.warn)
            case .failed:   KStatusBadge(glyph: .remote, label: "error", color: T.bad)
            case .unloaded: KStatusBadge(glyph: .remote, label: "off", color: T.ink3)
            }
        case .kokoro:
            switch voiceServiceObs.kokoroState {
            case .ready:    KStatusBadge(glyph: .ready, label: "ready", color: T.good)
            case .loading:  KStatusBadge(glyph: .streaming, label: "loading", color: T.warn)
            case .failed:   KStatusBadge(glyph: .remote, label: "error", color: T.bad)
            case .unloaded: KStatusBadge(glyph: .remote, label: "off", color: T.ink3)
            }
        }
    }

    @ViewBuilder
    private var downloadStatusBadge: some View {
        let center = ModelDownloadCenter.shared
        let readyCount = center.models.filter { $0.isReady }.count
        let total = center.models.count
        let anyActive = center.models.contains { $0.state.isActive }

        if anyActive {
            KStatusBadge(glyph: .download, label: "downloading", color: T.warn)
        } else if readyCount == total {
            KStatusBadge(glyph: .ready, label: "all ready", color: T.good)
        } else {
            KMono(text: "\(readyCount)/\(total) ready", size: 10, color: T.ink3, mono: false)
        }
    }

    // MARK: - System Status Section

    @State private var showSystemStatus = false
    @State private var showDownloadCenter = false
    @State private var showAssistantModelPicker = false
    @State private var showVoiceModelPicker = false
    @State private var showPersonaPicker = false
    @State private var showSnippetEditor = false

    private var systemStatusSection: some View {
        KCollapsibleSection(title: "system_status", defaultExpanded: false) {
            VStack(spacing: 0) {
                HStack(spacing: 10) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(T.good.opacity(0.12))
                            .frame(width: 28, height: 28)
                        Image(systemName: "waveform.path.ecg")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundColor(T.good)
                    }
                    statusSummary
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 12)

                Rectangle().fill(T.rule).frame(height: 1)

                Button {
                    showSystemStatus = true
                    HapticManager.impact(.light)
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "chart.bar.xaxis")
                            .font(.system(size: 12))
                            .foregroundColor(T.accent)
                        KMono(text: "Open full diagnostics", size: 11.5, color: T.accent, mono: false)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10))
                            .foregroundColor(T.ink3)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 11)
                }
                .buttonStyle(.plain)
            }
        }
        .sheet(isPresented: $showSystemStatus) { SystemStatusView() }
    }

    @State private var settingsStatusSnap: SystemStatusService.Snapshot = .empty

    @ViewBuilder
    private var statusSummary: some View {
        // Static snapshot — refreshed on view appear, not via observation.
        // This prevents the 2s timer from re-rendering all of Settings.
        let snap = settingsStatusSnap
        HStack(spacing: 6) {
            KMono(text: snap.totalRAM.formattedBytes + " RAM", size: 10, color: T.ink3, mono: false)
            KMono(text: "·", size: 10, color: T.ink4, mono: false)
            KMono(text: snap.supportsNeuralEngine ? "ANE available" : "No ANE",
                   size: 10, color: snap.supportsNeuralEngine ? T.good : T.ink3, mono: false)
            KMono(text: "·", size: 10, color: T.ink4, mono: false)
            KMono(text: snap.modelStorageUsed.formattedBytes + " models",
                   size: 10, color: T.ink3, mono: false)
        }
        .onAppear {
            SystemStatusService.shared.refresh()
            settingsStatusSnap = SystemStatusService.shared.snapshot
        }
    }

    // MARK: - API Settings Section
    //
    // Hugging Face token surface. Mirrors the FastVLM section's
    // typographic rhythm — KSection caption + KRow body + a stack
    // row for the trailing button so the description has room to
    // breathe under the title.

    private var apiSettingsSection: some View {
        KSection(title: "api_settings") {
            // Token row — description + Set/Update button. The trailing
            // button label flips based on whether a token is already
            // saved, so the row carries that state without a separate
            // status pill cluttering the right edge.
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .top, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        KMono(text: "Hugging Face token", size: 11.5, color: T.ink, mono: false)
                        Text("Required for gated repos (google/gemma-*, meta-llama/*, mistralai/Ministral-*). Stored in Keychain.")
                            .font(T.sans(11))
                            .foregroundColor(T.ink3)
                            .fixedSize(horizontal: false, vertical: true)
                        if let masked = hfToken.maskedPreview {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.seal.fill")
                                    .font(.system(size: 9))
                                    .foregroundColor(T.good)
                                Text(masked)
                                    .font(T.mono(10, .semibold))
                                    .foregroundColor(T.good)
                            }
                            .padding(.top, 2)
                        }
                    }
                    Spacer(minLength: 6)
                    Button {
                        showingHFTokenSheet = true
                        HapticManager.impact(.light)
                    } label: {
                        Text(hfToken.hasToken ? "Update" : "Set token")
                            .font(T.sans(11, .semibold))
                            .tracking(0.4)
                            .foregroundColor(T.accent)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 7)
                            .background(Capsule().fill(T.accentSoft))
                            .overlay(Capsule().stroke(T.accent.opacity(0.4), lineWidth: 0.5))
                    }
                    .buttonStyle(.plain)
                    .fixedSize()
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)

            Rectangle().fill(T.rule).frame(height: 1)

            // Use token toggle — gates the Authorization header
            // application across all HF requests. Disabled when no
            // token is stored (nothing to toggle).
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        KMono(text: "Use token for downloads", size: 11.5, color: T.ink, mono: false)
                        Text("Sends your token with HF requests. Turn off temporarily to debug 401s without removing the token.")
                            .font(T.sans(10.5))
                            .foregroundColor(T.ink3)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Spacer()
                    KToggle(isOn: $settings.useHFToken)
                        .opacity(hfToken.hasToken ? 1 : 0.4)
                        .disabled(!hfToken.hasToken)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
        }
    }

    // MARK: - FastVLM Pipeline Section

    private var fastvlmPipelineSection: some View {
        KCollapsibleSection(title: "fastvlm_pipeline", defaultExpanded: false) {
            componentStatusRow("Encoder (FastViT-HD)", state: fastVLMStatus.encoder)
            componentStatusRow("Projector (linear+GELU)", state: fastVLMStatus.projector)
            componentStatusRow("Decoder (Qwen2-0.5B)", state: fastVLMStatus.decoder)
            componentStatusRow("Tokenizer", state: fastVLMStatus.tokenizer, last: true)

            Rectangle().fill(T.rule).frame(height: 1)

            KRow(label: "Use OCR fallback", trailing: {
                KToggle(isOn: $settings.useOCRFallback)
            })

            KRow(label: "Temperature", trailing: {
                KMono(text: String(format: "%.2f", settings.fastvlmTemperature),
                       color: T.ink2, mono: false)
            }, stack: true)
            VStack(alignment: .leading, spacing: 4) {
                Slider(value: $settings.fastvlmTemperature, in: 0.0...1.0, step: 0.05)
                    .tint(T.accent)
                KMono(text: "0 = greedy. Recommended 0.2 for extraction.",
                       size: 10, color: T.ink3, mono: false)
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 10)
            Rectangle().fill(T.rule).frame(height: 1)

            KRow(label: "Max tokens", trailing: {
                Stepper("\(settings.fastvlmMaxTokens)",
                        value: $settings.fastvlmMaxTokens, in: 64...1024, step: 64)
                    .labelsHidden()
            }, last: fastVLMStatus.canGenerate || !LocalModelRegistry.isDefaultVisionSelection(
                AppSettings.shared.cameraVisualModelID
            ))

            // FastVLM "missing components" panel — only relevant when the
            // user is actually relying on FastVLM. If they've picked an MLX
            // VLM (SmolVLM etc.) the lens won't touch FastVLM, so nagging
            // them about its components here is just noise. The panel also
            // stayed visible after a successful download because
            // canGenerate requires the model to be LOADED into memory, not
            // just present on disk — copy now makes that distinction clear.
            let usingFastVLM = LocalModelRegistry.isDefaultVisionSelection(
                AppSettings.shared.cameraVisualModelID
            )
            if usingFastVLM, !fastVLMStatus.canGenerate {
                let filesOnDisk = ModelDownloadCenter.shared.fastvlmModel?.isReady == true
                Rectangle().fill(T.rule).frame(height: 1)
                VStack(alignment: .leading, spacing: 8) {
                    KCaption(
                        text: filesOnDisk ? "DOWNLOADED — NOT YET LOADED" : "MISSING COMPONENTS",
                        color: T.warn
                    )
                    KMono(text: filesOnDisk
                          ? "Weights are on disk. Tap below to load them into memory (first capture also triggers a load)."
                          : "Open Download Models → FastVLM MLX Weights to fetch the missing files from HuggingFace.",
                           size: 11, color: T.ink2, mono: false)
                        .fixedSize(horizontal: false, vertical: true)
                    Button {
                        HapticManager.impact(.light)
                        if filesOnDisk {
                            // Files exist — trigger the load directly so
                            // canGenerate flips to true without bouncing
                            // through the Models tab.
                            Task { await FastVLMService.shared.load() }
                        } else {
                            dismiss()
                            AppBridge.shared.requestTab(.models)
                        }
                    } label: {
                        HStack(spacing: 5) {
                            Image(systemName: filesOnDisk
                                  ? "arrow.triangle.2.circlepath"
                                  : "arrow.down.circle")
                                .font(.system(size: 11, weight: .medium))
                            Text(filesOnDisk ? "Load now" : "Download models")
                                .font(T.sans(11, .semibold))
                        }
                        .foregroundColor(T.bg)
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .background(RoundedRectangle(cornerRadius: 5).fill(T.ink))
                    }
                    .buttonStyle(.plain)
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .sheet(isPresented: $showDownloadCenter) { ModelDownloadCenterView() }
    }

    // MARK: - Developer Section
    //
    // Path/URL escape hatches + diagnostic overlays. Collapsed by default so
    // they don't compete with everyday preferences. README §Settings: "Power-
    // user escape hatches don't live in the main flow."

    private var developerSection: some View {
        KCollapsibleSection(title: "developer", tinted: true, defaultExpanded: false) {
            // Live diagnostics: crash trail, logs, memory/thermal, MetricKit.
            NavigationLink {
                DiagnosticsView()
            } label: {
                HStack {
                    Image(systemName: "stethoscope")
                        .font(.system(size: 13))
                        .foregroundColor(T.ink)
                    KMono(text: "Diagnostics & crash log", size: 11.5, color: T.ink, mono: false)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)

            Rectangle().fill(T.rule).frame(height: 1)

            // Editable FastVLM repo ID — lets users switch to a working
            // mirror if the default 404/401s.
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    KMono(text: "FastVLM repo", size: 11, color: T.ink3, mono: false)
                    Spacer()
                    Button {
                        settings.fastVLMRepoID = "apple/FastVLM-0.5B-MLX"
                        HapticManager.impact(.light)
                    } label: {
                        Text("Reset")
                            .font(T.sans(10))
                            .foregroundColor(T.accent)
                    }
                }
                TextField("author/repo", text: $settings.fastVLMRepoID)
                    .font(T.mono(11))
                    .foregroundColor(T.ink)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(T.surface2))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(T.rule, lineWidth: 1))
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                KMono(text: "Hugging Face repo for the VLM weights. Takes effect on next app launch.",
                       size: 9, color: T.ink3, mono: false)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            Rectangle().fill(T.rule).frame(height: 1)
            KRow(label: "Show debug model shapes", trailing: {
                KToggle(isOn: $settings.showDebugModelShapes)
            })

            if settings.showDebugModelShapes, let debug = fastVLMDebugInfo {
                VStack(alignment: .leading, spacing: 4) {
                    if let shape = debug.encoderOutputShape {
                        KMono(text: "encoder out: \(shape.map(String.init).joined(separator: "×"))",
                               size: 10, color: T.ink2)
                    }
                    if let shape = debug.projectorOutputShape {
                        KMono(text: "projector out: \(shape.map(String.init).joined(separator: "×"))",
                               size: 10, color: T.ink2)
                    }
                    if let tps = debug.lastTokensPerSecond {
                        KMono(text: String(format: "speed: %.1f tok/s", tps),
                               size: 10, color: T.ink2)
                    }
                    if let err = debug.lastGenerationError {
                        KMono(text: "error: \(err)", size: 10, color: T.bad)
                    }
                }
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                Rectangle().fill(T.rule).frame(height: 1)
            }

            // Diagnostic toggle. When on, the lens shows a small
            // thumbnail of the exact image MLX received last (post-
            // orientation, post-resize, post-everything). Lets you
            // diagnose accuracy issues at a glance.
            KRow(label: "show model input debug", trailing: {
                KToggle(isOn: $settings.showModelInputDebug)
            }, last: !settings.showModelInputDebug)

            if settings.showModelInputDebug {
                Button {
                    if let url = MLXVisionService.shared.saveLastModelInputToDocuments() {
                        ToastCenter.shared.success("Saved model input",
                                                    detail: url.lastPathComponent)
                    } else {
                        ToastCenter.shared.error("No model input to save",
                                                  detail: "Run a lens capture first.")
                    }
                    HapticManager.impact(.light)
                } label: {
                    HStack {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 12))
                            .foregroundColor(T.ink)
                        KMono(text: "Save last model input PNG", size: 11.5, color: T.ink, mono: false)
                        Spacer()
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder
    private func componentStatusRow(_ label: String, state: FastVLMModelState, last: Bool = false) -> some View {
        KRow(label: label, trailing: {
            HStack(spacing: 6) {
                Text(stateGlyph(for: state))
                    .font(T.mono(10))
                    .foregroundColor(statusColor(for: state))
                KMono(text: state.statusLabel.lowercased(), size: 10, color: T.ink3, mono: false)
            }
        }, last: last)
    }

    private func stateGlyph(for state: FastVLMModelState) -> String {
        switch state {
        case .ready:    return "●"
        case .loading:  return "◐"
        case .failed:   return "✕"
        case .unloaded: return "○"
        }
    }

    private func statusColor(for state: FastVLMModelState) -> Color {
        switch state {
        case .ready:   return T.good
        case .loading: return T.warn
        case .failed:  return T.bad
        case .unloaded: return T.ink3
        }
    }

    private var assistantSection: some View {
        KSection(title: "assistant") {
            KRow(label: "thinking mode", trailing: {
                KToggle(isOn: $settings.assistantThinking)
            })

            Rectangle().fill(T.rule).frame(height: 1)
            // Lets the assistant emit `tool` JSON blocks (calculator,
            // datetime, web_search, file_read, describe_image) and have
            // them executed on-device. Off → the system prompt drops
            // the tool addendum, so the model never mentions tools at
            // all. Defaults on.
            KRow(label: "let model call tools", trailing: {
                KToggle(isOn: $settings.toolsEnabled)
            })

            Rectangle().fill(T.rule).frame(height: 1)
            VStack(alignment: .leading, spacing: 6) {
                KMono(text: "Response style", size: 11, color: T.ink3, mono: false)
                Picker("", selection: Binding(
                    get: { ResponseStyle.nearest(settings.assistantTemperature) },
                    set: { settings.assistantTemperature = $0.temperature }
                )) {
                    ForEach(ResponseStyle.allCases, id: \.self) {
                        Text($0.rawValue.lowercased()).tag($0)
                    }
                }
                .pickerStyle(.segmented)
                KMono(text: ResponseStyle.nearest(settings.assistantTemperature).hint, size: 10, color: T.ink3, mono: false)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)

            Rectangle().fill(T.rule).frame(height: 1)
            VStack(alignment: .leading, spacing: 6) {
                KMono(text: "Response length", size: 11, color: T.ink3, mono: false)
                Picker("", selection: Binding(
                    get: { ResponseLength.nearest(settings.assistantMaxTokens) },
                    set: {
                        settings.assistantMaxTokens = $0.maxTokens
                        settings.hasPickedResponseLength = true
                    }
                )) {
                    ForEach(ResponseLength.allCases, id: \.self) {
                        Text($0.rawValue.lowercased()).tag($0)
                    }
                }
                .pickerStyle(.segmented)
                KMono(text: ResponseLength.nearest(settings.assistantMaxTokens).hint, size: 10, color: T.ink3, mono: false)
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 10)

            Rectangle().fill(T.rule).frame(height: 1)
            // On-device RAG over the user's own files — chat with your
            // documents/codebase, fully offline.
            Button { showingKnowledgeBase = true } label: {
                HStack {
                    Image(systemName: "books.vertical")
                        .font(.system(size: 13))
                        .foregroundColor(T.ink)
                    KMono(text: "Knowledge base (RAG)", size: 11.5, color: T.ink, mono: false)
                    Spacer()
                    KMono(text: knowledgeBase.documents.isEmpty
                            ? "off"
                            : "\(knowledgeBase.documents.count) docs · \(knowledgeBase.isEnabled ? "on" : "off")",
                           size: 10, color: T.ink3)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Voice Section

    private var voiceSection: some View {
        KSection(title: "voice") {
            NavigationLink {
                VoiceSettingsView()
            } label: {
                HStack {
                    Image(systemName: "speaker.wave.2")
                        .font(.system(size: 13))
                        .foregroundColor(T.ink)
                    KMono(text: "Voice settings", size: 11.5, color: T.ink, mono: false)
                    Spacer()
                    KMono(text: currentEngineLabel.lowercased(), size: 10, color: T.ink3, mono: false)
                    Image(systemName: "chevron.right")
                        .font(.system(size: 10))
                        .foregroundColor(T.ink3)
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
            Rectangle().fill(T.rule).frame(height: 1)

            KRow(label: "auto-read results", trailing: {
                KToggle(isOn: $settings.voiceAutoRead)
            })

            Rectangle().fill(T.rule).frame(height: 1)

            // Speech-to-text provider picker. Whisper option is disabled
            // when no whisper model is installed on disk so users can't
            // toggle into a broken state — tapping the segment when
            // disabled would silently fall back to system anyway, which
            // is confusing. The row owns its own download affordance so
            // users can fetch Whisper inline without leaving Settings.
            WhisperSTTBlock()

            if VoiceService.shared.isPlaying {
                Rectangle().fill(T.rule).frame(height: 1)
                KSecondaryButton(label: "Stop speaking",
                                 systemImage: "stop.circle",
                                 trailing: nil,
                                 destructive: true) {
                    VoiceService.shared.stop()
                }
                .padding(10)
            }
        }
    }

    private var currentEngineLabel: String {
        (VoiceEngineKind(rawValue: settings.voiceEngine) ?? .appleSystem).shortName
    }

    // MARK: - App Icon Picker

    #if !targetEnvironment(macCatalyst)
    /// Default "Neon" icon + the previous "Classic" mark as an alternate.
    /// Declared via CFBundleAlternateIcons in Info.plist; the classic asset
    /// compiles because of INCLUDE_ALL_APPICON_ASSETS.
    private var appIconPickerRow: some View {
        VStack(alignment: .leading, spacing: 8) {
            KMono(text: "app icon", size: 11.5, color: T.ink, mono: false)
            HStack(spacing: 14) {
                appIconOption(name: nil, title: "Monochrome", preview: "AppIconPreview")
                appIconOption(name: "AppIconClassic", title: "Classic", preview: "AppIconClassicPreview")
                Spacer()
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func appIconOption(name: String?, title: String, preview: String) -> some View {
        let selected = activeIconName == name
        return Button {
            guard !selected else { return }
            HapticManager.impact(.light)
            activeIconName = name
            UIApplication.shared.setAlternateIconName(name) { error in
                guard let error else { return }
                Task { @MainActor in
                    ToastCenter.shared.info(
                        "Couldn't change app icon",
                        detail: error.localizedDescription
                    )
                }
            }
        } label: {
            VStack(spacing: 5) {
                Image(preview)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 52, height: 52)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(selected ? T.accent : T.rule,
                                          lineWidth: selected ? 2 : 1)
                    )
                KMono(text: title, size: 9.5,
                      color: selected ? T.accent : T.ink3, mono: false)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(title) app icon")
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
    #endif

    private var uiSection: some View {
        KSection(title: loc.t("INTERFACE").lowercased(), tinted: true) {
            KRow(label: loc.t("appearance"), trailing: {
                Picker("", selection: $settings.appearance) {
                    Text(loc.t("light")).tag("light")
                    Text(loc.t("dark")).tag("dark")
                    Text(loc.t("OLED")).tag("oled")
                }
                .pickerStyle(.segmented)
                .frame(width: 170)
            })
            #if !targetEnvironment(macCatalyst)
            if UIApplication.shared.supportsAlternateIcons {
                appIconPickerRow
            }
            #endif
            KRow(label: loc.t("language"), trailing: {
                Picker("", selection: Binding(
                    get: { AppLanguage(rawValue: settings.uiLanguage) ?? .system },
                    set: { newValue in
                        settings.uiLanguage = newValue.rawValue
                        loc.setLanguage(newValue)
                    }
                )) {
                    ForEach(AppLanguage.allCases) { lang in
                        Text(lang.nativeName).tag(lang)
                    }
                }
                .pickerStyle(.menu)
                .tint(T.ink)
            })
            KRow(label: loc.t("haptic feedback"), trailing: {
                KToggle(isOn: $settings.hapticsEnabled)
            })
            KRow(label: "show fps counter", trailing: {
                KToggle(isOn: $settings.showFPSCounter)
            })
            Rectangle().fill(T.rule).frame(height: 1)
            Button {
                settings.hasSeenOnboarding = false
                dismiss()
            } label: {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                        .font(.system(size: 12))
                        .foregroundColor(T.ink)
                    KMono(text: "Show onboarding again", size: 11.5, color: T.ink, mono: false)
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)

            Rectangle().fill(T.rule).frame(height: 1)
            Button {
                TipsManager.shared.resetAll()
                HapticManager.impact(.light)
                ToastCenter.shared.info("Tips reset",
                                          detail: "First-use hints will show again.")
            } label: {
                HStack {
                    Image(systemName: "lightbulb")
                        .font(.system(size: 12))
                        .foregroundColor(T.ink)
                    KMono(text: "Reset onboarding tips", size: 11.5, color: T.ink, mono: false)
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
            }
            .buttonStyle(.plain)
        }
    }

    // MARK: - Thermal protection

    @ObservedObject private var safetyMonitor = DeviceSafetyMonitor.shared

    private var thermalProtectionSection: some View {
        KCollapsibleSection(title: "thermal_protection", defaultExpanded: false) {
            VStack(spacing: 0) {
                // Live state — both the raw iOS reading AND our debounced
                // "effective" value. Useful when the user thinks the warning
                // is a false positive.
                KSpecTable(rows: [
                    ("raw state",       label(for: safetyMonitor.thermalState)),
                    ("effective state", label(for: safetyMonitor.effectiveThermalState)),
                    ("low-power mode",  safetyMonitor.lowPowerMode ? "on" : "off"),
                    ("warnings",        settings.thermalWarningsEnabled ? "enabled" : "disabled"),
                ], keyWidth: 110)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)

                Rectangle().fill(T.rule).frame(height: 1)

                // Toggle — for users who know their device is fine and want
                // to silence the pill / lift the max-token cap.
                Toggle(isOn: $settings.thermalWarningsEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        KMono(text: "Thermal warnings", size: 11.5, color: T.ink, mono: false)
                        KMono(
                            text: "off → hide pill, no auto-cap on max tokens",
                            size: 10, color: T.ink3
                        )
                    }
                }
                .tint(T.accent)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)

                Rectangle().fill(T.rule).frame(height: 1)

                // Manual override: force-clear a stuck warning. iOS sometimes
                // reports .serious after a heavy first-load and takes 30s+ to
                // drop back to .nominal even when the device feels cool.
                Button {
                    safetyMonitor.clearThermalWarning()
                    HapticManager.impact(.light)
                    ToastCenter.shared.info("Cleared until iOS updates the reading")
                } label: {
                    HStack {
                        Image(systemName: "thermometer.snowflake")
                            .font(.system(size: 12))
                            .foregroundColor(T.accent)
                        KMono(text: "Clear current warning", size: 11.5, color: T.accent, mono: false)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10))
                            .foregroundColor(T.ink3)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func label(for s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "?"
        }
    }

    // MARK: - Privacy / reset

    @State private var showWipeConfirm = false
    @State private var lastWipeReceipt: WipeAllDataService.Receipt?

    private var privacyResetSection: some View {
        KSection(title: "privacy_and_reset") {
            VStack(spacing: 0) {
                Button(role: .destructive) {
                    showWipeConfirm = true
                    HapticManager.impact(.medium)
                } label: {
                    HStack {
                        Image(systemName: "trash.slash")
                            .font(.system(size: 12))
                            .foregroundColor(T.bad)
                        KMono(text: "Wipe all on-device data", size: 11.5, color: T.bad, mono: false)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10))
                            .foregroundColor(T.ink3)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.plain)
                if let r = lastWipeReceipt {
                    Rectangle().fill(T.rule).frame(height: 1)
                    VStack(alignment: .leading, spacing: 4) {
                        KMono(text: "wiped", size: 9, color: T.ink3, mono: false)
                        Text("\(r.modelsDeleted) models · \(r.conversationsDeleted) conversations · \(r.snippetsDeleted) snippets · \(r.memoriesDeleted) memories · \(r.keychainItemsCleared) credentials · freed \(r.bytesFreed.formattedBytes)")
                            .font(T.mono(11))
                            .foregroundColor(T.ink2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 14).padding(.vertical, 10)
                }
            }
        }
        .confirmationDialog(
            "Wipe all on-device data?",
            isPresented: $showWipeConfirm,
            titleVisibility: .visible
        ) {
            Button("Wipe everything", role: .destructive) {
                lastWipeReceipt = WipeAllDataService.wipeAll()
                ToastCenter.shared.success(
                    "Wiped",
                    detail: "Restart the app for a completely fresh state."
                )
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Deletes downloaded models, conversations, snippets, memories, saved API keys, Mac Bridge pairings, cache, and resets onboarding. Cannot be undone.")
        }
    }

    // MARK: - Legal Section

    @State private var showPrivacy = false
    @State private var showEULA = false
    @State private var showDisclaimer = false
    @State private var showAttributions = false

    @State private var showDeviceSafety = false

    private var legalSection: some View {
        KSection(title: "legal") {
            legalRow(icon: "lock.shield", label: "privacy policy") { showPrivacy = true }
            legalRow(icon: "doc.text", label: "open-source license & safety") { showEULA = true }
            legalRow(icon: "exclamationmark.triangle", label: "ai output disclaimer") { showDisclaimer = true }
            legalRow(icon: "thermometer.medium", label: "device safety notice") { showDeviceSafety = true }
            legalRow(icon: "heart.text.square", label: "open-source & attributions",
                     last: URL(string: LegalDocuments.supportURL) == nil) {
                showAttributions = true
            }
            // Acceptance audit row — small, ink3 — so users (and us) can see
            // when they accepted.
            if let when = LegalAcceptanceManager.shared.acceptedAtDescription {
                Rectangle().fill(T.rule).frame(height: 1)
                HStack {
                    Image(systemName: "checkmark.seal")
                        .font(.system(size: 11))
                        .foregroundColor(T.ink3)
                    KMono(text: "accepted v\(LegalDocuments.currentVersion) on \(when)",
                          size: 10, color: T.ink3)
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
            }
            if let supportURL = URL(string: LegalDocuments.supportURL) {
                Link(destination: supportURL) {
                    HStack {
                        Image(systemName: "questionmark.circle")
                            .font(.system(size: 12))
                            .foregroundColor(T.accent)
                        KMono(text: "Support", size: 11.5, color: T.accent, mono: false)
                        Spacer()
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 10))
                            .foregroundColor(T.accent)
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                }
            }
        }
        .sheet(isPresented: $showPrivacy) {
            LegalDocumentView(title: "Privacy Policy",
                              markdown: LegalDocuments.privacyPolicy,
                              externalURL: LegalDocuments.privacyPolicyURL)
        }
        .sheet(isPresented: $showEULA) {
            LegalDocumentView(title: "License & Safety",
                              markdown: LegalDocuments.eula,
                              externalURL: LegalDocuments.eulaURL)
        }
        .sheet(isPresented: $showDisclaimer) {
            LegalDocumentView(title: "AI Disclaimer",
                              markdown: LegalDocuments.aiDisclaimer)
        }
        .sheet(isPresented: $showDeviceSafety) {
            LegalDocumentView(title: "Device Safety",
                              markdown: LegalDocuments.deviceSafetyNotice)
        }
        .sheet(isPresented: $showAttributions) { AttributionsView() }
    }

    @ViewBuilder
    private func legalRow(icon: String, label: String, last: Bool = false, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(T.ink)
                KMono(text: label, size: 11.5, color: T.ink, mono: false)
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 10))
                    .foregroundColor(T.ink3)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .overlay(alignment: .bottom) {
                if !last {
                    Rectangle().fill(T.rule).frame(height: 1)
                }
            }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Response style / length helpers (used by assistantSection pickers)

    private var imageStudioBanner: some View {
        Button {
            openURL(Self.imageStudioWebsiteURL)
            HapticManager.impact(.light)
        } label: {
            VStack(alignment: .leading, spacing: 13) {
                HStack(alignment: .center, spacing: 12) {
                    Image("image_studio_app_icon")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 58, height: 58)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(T.rule, lineWidth: 1)
                        )

                    VStack(alignment: .leading, spacing: 4) {
                        KCaption(text: "also by mesut")
                        Text("OnDevice: AI Image Studio")
                            .font(T.sans(18, .bold))
                            .foregroundColor(T.ink)
                            .lineLimit(1)
                            .minimumScaleFactor(0.82)
                        Text("Create private AI images on-device, even when you're offline.")
                            .font(T.sans(12.5))
                            .foregroundColor(T.ink2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .layoutPriority(1)

                    Spacer(minLength: 0)
                }

                HStack(spacing: 8) {
                    promoFeaturePill(icon: "lock.shield", text: "private")
                    promoFeaturePill(icon: "iphone", text: "on-device")
                    Spacer(minLength: 8)
                    Text("coming soon")
                        .font(T.mono(10, .semibold))
                        .foregroundColor(T.warn)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(Capsule().fill(T.warn.opacity(0.10)))
                        .overlay(Capsule().stroke(T.warn.opacity(0.32), lineWidth: 0.5))
                }

                HStack(spacing: 8) {
                    Text("See the preview and follow launch updates.")
                        .font(T.sans(11.5))
                        .foregroundColor(T.ink3)
                        .lineLimit(1)
                        .minimumScaleFactor(0.8)
                    Spacer(minLength: 8)
                    HStack(spacing: 5) {
                        Text("Website")
                            .font(T.mono(11, .semibold))
                            .lineLimit(1)
                        Image(systemName: "chevron.right")
                            .font(.system(size: 9, weight: .bold))
                    }
                    .foregroundColor(T.ink)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Capsule().fill(T.surface2))
                    .overlay(Capsule().stroke(T.rule, lineWidth: 1))
                }
            }
            .padding(15)
            .frame(maxWidth: .infinity, alignment: .leading)
            .kGlass(cornerRadius: 20, fallbackFill: T.surface, fallbackStroke: T.rule)
        }
        .buttonStyle(KTactileButtonStyle())
        .accessibilityLabel("Open the OnDevice AI Image Studio preview website. App Store release coming soon.")
    }

    private func promoFeaturePill(icon: String, text: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
            Text(text)
                .font(T.mono(10, .semibold))
                .lineLimit(1)
        }
        .foregroundColor(T.ink2)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(Capsule().fill(T.surface2))
        .overlay(Capsule().stroke(T.rule, lineWidth: 0.5))
    }

    private var appVersionLabel: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(version) (\(build))"
    }

    private var aboutSection: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                KCaption(text: "about")
                Rectangle().fill(T.rule).frame(height: 1)
            }
            .padding(.bottom, 8)
            KSpecTable(rows: [
                ("version",   appVersionLabel),
                ("models",    "apple vision · fastvit-hd · qwen2.5"),
                ("framework", "core ml · mlx swift · vision"),
                ("runtime",   "mlx-swift-examples 2.21"),
                ("ios",       "18.0+"),
                ("privacy",   "100% on-device"),
            ], keyWidth: 80)
        }
        .padding(.horizontal, 16)
        .padding(.top, 18)
    }

    @ViewBuilder
    private var assistantStatusBadge: some View {
        switch assistant.state {
        case .ready:      KStatusBadge(glyph: .ready, label: "ready", color: T.good)
        case .loading:    KStatusBadge(glyph: .streaming, label: "loading", color: T.warn)
        case .generating: KStatusBadge(glyph: .streaming, label: "busy", color: T.accent)
        case .unloaded:   KStatusBadge(glyph: .remote, label: "off", color: T.ink3)
        case .failed:     KStatusBadge(glyph: .remote, label: "error", color: T.bad)
        }
    }
}

// MARK: - Response style presets

private enum ResponseStyle: String, CaseIterable, Hashable {
    case precise  = "Precise"
    case balanced = "Balanced"
    case creative = "Creative"

    var temperature: Double {
        switch self { case .precise: 0.2; case .balanced: 0.6; case .creative: 0.9 }
    }

    var hint: String {
        switch self {
        case .precise:  return "consistent · best for code"
        case .balanced: return "good for most questions"
        case .creative: return "more varied · best for writing"
        }
    }

    static func nearest(_ t: Double) -> Self {
        if t <= 0.4 { return .precise }
        if t <= 0.75 { return .balanced }
        return .creative
    }
}

// MARK: - Response length presets

private enum ResponseLength: String, CaseIterable, Hashable {
    case short    = "Short"
    case normal   = "Normal"
    case detailed = "Detailed"
    case full     = "Full"

    var maxTokens: Int {
        switch self { case .short: 512; case .normal: 1024; case .detailed: 2048; case .full: 4096 }
    }

    var hint: String {
        switch self {
        case .short:    return "quick answers · faster"
        case .normal:   return "good for most questions"
        case .detailed: return "thorough explanations"
        case .full:     return "longest possible reply"
        }
    }

    static func nearest(_ tokens: Int) -> Self {
        if tokens <= 512  { return .short }
        if tokens <= 768  { return .normal }
        if tokens <= 1280 { return .detailed }
        return .full
    }
}

// MARK: - Whisper STT row
//
// Speech-to-text segmented picker (system / whisper) with an inline
// Download Whisper button that takes the user from "greyed-out" to
// "fully installed" without leaving Settings. Previously the row
// pointed at the Models tab with a static hint — a two-screen
// detour for a single 142 MB file.
//
// Drives off the existing `whisper-base-en` catalog entry in
// ModelDownloadCenter, so the download itself is identical to the
// one a user would kick off from the Download Center. The Voice
// section just gets a more proximate trigger.

private struct WhisperSTTBlock: View {
    @ObservedObject private var settings = AppSettings.shared
    @ObservedObject private var voiceServiceObs = VoiceService.shared
    @StateObject private var observer: DownloadObserver
    @Environment(\.koduTheme) private var T

    private let whisperModel: DownloadableModel?

    init() {
        let m = ModelDownloadCenter.shared.models.first { $0.id == "whisper-base-en" }
        self.whisperModel = m
        // ModelDownloadCenter.buildCatalog runs in init and always
        // appends the whisper-base-en entry, so the optional should
        // never be nil at runtime. Crash early if catalog wiring
        // changes — DownloadObserver requires a real model.
        _observer = StateObject(wrappedValue: DownloadObserver(
            model: m ?? ModelDownloadCenter.shared.models.first!
        ))
    }

    /// Reactive install check — disk presence OR observer reporting
    /// .ready. We can't rely on `WhisperModelCatalog.isInstalled` alone
    /// because that's a static fileExists probe; the Picker stays
    /// disabled across a successful in-place download without the
    /// observer half of the check.
    private var isInstalled: Bool {
        observer.state == .ready || WhisperModelCatalog.isInstalled
    }

    var body: some View {
        let speaking = voiceServiceObs.isPlaying
        VStack(spacing: 0) {
            KRow(label: "speech-to-text", trailing: {
                Picker("", selection: $settings.sttProvider) {
                    Text("system").tag("system")
                    Text("whisper").tag("whisper")
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
                .disabled(!isInstalled)
            }, last: !speaking && isInstalled && !observer.isActive)

            if !isInstalled {
                Rectangle().fill(T.rule).frame(height: 1)
                downloadCard
            }
        }
    }

    @ViewBuilder
    private var downloadCard: some View {
        switch observer.state {
        case .downloading, .enumerating:
            progressCard
        case .failed(let msg):
            failedCard(message: msg)
        case .idle, .ready:
            idleCard
        }
    }

    private var idleCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.system(size: 11))
                    .foregroundColor(T.ink3)
                KMono(text: "Install Whisper for on-device, multilingual STT.", size: 10, color: T.ink3, mono: false)
                .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }
            KSecondaryButton(
                label: "Download Whisper base.en (~142 MB)",
                systemImage: "arrow.down.circle",
                trailing: nil
            ) {
                whisperModel?.start()
                HapticManager.impact(.medium)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private var progressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                ProgressView()
                    .tint(T.accent)
                    .scaleEffect(0.7)
                KMono(
                    text: observer.state == .enumerating
                        ? "checking files…"
                        : "downloading whisper… \(Int(observer.progress * 100))%",
                    size: 11, color: T.ink2
                )
                Spacer()
                Button {
                    whisperModel?.cancel()
                    HapticManager.impact(.light)
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundColor(T.ink3)
                }
                .buttonStyle(.plain)
            }
            // Thin progress bar mirroring the Download Center.
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Rectangle().fill(T.surface2).frame(height: 2)
                    Rectangle().fill(T.accent)
                        .frame(width: geo.size.width * max(0, min(1, observer.progress)),
                               height: 2)
                        .animation(.linear(duration: 0.3), value: observer.progress)
                }
            }
            .frame(height: 2)
            if observer.totalBytes > 0 {
                KMono(
                    text: "\(observer.downloadedBytes.formattedBytes) / \(observer.totalBytes.formattedBytes)",
                    size: 9, color: T.ink3
                )
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }

    private func failedCard(message: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.system(size: 11))
                    .foregroundColor(T.bad)
                KMono(text: "Download failed: \(message)", size: 10, color: T.bad, mono: false)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer()
            }
            KSecondaryButton(
                label: "Retry download",
                systemImage: "arrow.clockwise",
                trailing: nil
            ) {
                whisperModel?.start()
                HapticManager.impact(.medium)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}
