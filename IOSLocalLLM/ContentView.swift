import SwiftUI
import UIKit

struct LocalAPIServerView: View {
    @ObservedObject private var manager = LocalAPIManager.shared
    @ObservedObject private var modelService = CodingAssistantService.shared
    @ObservedObject private var systemStatus = SystemStatusService.shared
    @ObservedObject private var safetyMonitor = DeviceSafetyMonitor.shared
    @Environment(\.scenePhase) private var scenePhase

    @AppStorage("localAPIEnabled") private var serverEnabled = true
    @AppStorage("localAPIPort") private var port = 11_434
    @AppStorage("localAPIKeepScreenAwake") private var keepScreenAwake = true
    @AppStorage("localAPIAutoLoadModel") private var autoLoadModel = false
    @AppStorage("localAPIToolCallingEnabled") private var toolCallingEnabled = true
    @AppStorage("localAPIReasoningEnabled") private var reasoningEnabled = false
    @AppStorage("localAPIParallelToolCallsLimit") private var parallelToolCallsLimit = 2
    @AppStorage("localAPIStrictToolSchemasEnabled") private var strictToolSchemasEnabled = true

    @State private var portText = "11434"
    @State private var apiKeyVisible = false
    @State private var copyMessage: String?
    @State private var showingKeyRotationConfirmation = false
    @State private var showingModelPicker = false
    @State private var showingDownloads = false
    @State private var lastLoggedModelState: CodingAssistantService.ServiceState?

    var body: some View {
        NavigationStack {
            ZStack {
                LASPageBackground()

                VStack(spacing: 0) {
                    LASHomeHeader()

                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                        LASNetworkBanner(
                            state: manager.state,
                            modelState: modelService.state,
                            address: manager.addresses.first,
                            port: port
                        )

                        LASHealthDashboard(
                            snapshot: systemStatus.snapshot,
                            thermalState: safetyMonitor.effectiveThermalState,
                            lowPowerMode: safetyMonitor.lowPowerMode
                        )

                        LASLocalServerParserCard(
                            serverState: manager.state,
                            modelState: modelService.state,
                            modelName: modelService.activeDisplayName,
                            tokenRate: modelService.tokenRate
                        )

#if CORE_AI_SERVER_APP
                        CoreAIModelStatusCard()
#endif

                        LASHomeServerCard(
                            state: manager.state,
                            serverEnabled: $serverEnabled,
                            keepScreenAwake: $keepScreenAwake,
                            autoLoadModel: $autoLoadModel,
                            toolCallingEnabled: $toolCallingEnabled,
                            reasoningEnabled: $reasoningEnabled,
                            parallelToolCallsLimit: $parallelToolCallsLimit,
                            strictToolSchemasEnabled: $strictToolSchemasEnabled,
                            portText: $portText,
                            modelName: modelService.activeDisplayName,
                            modelID: modelService.activeModel.id,
                            repoID: modelService.activeModel.repoID,
                            capabilityProfile: ModelCapabilityProfile.resolve(
                                for: modelService.activeModel
                            ),
                            modelState: modelService.state,
                            addresses: manager.addresses,
                            port: port,
                            apiKey: manager.apiKey,
                            apiKeyVisible: $apiKeyVisible,
                            copyMessage: copyMessage,
                            onRestart: restartServer,
                            onApplyPort: applyPort,
                            onLoadModel: loadModel,
                            onChooseModel: { showingModelPicker = true },
                            onStopModel: modelService.unload,
                            onCopyURL: copyURL,
                            onCopyKey: copyKey,
                            onRotateKey: { showingKeyRotationConfirmation = true },
                            onCopySetup: copySetupCommands
                        )

                        LASDeveloperEndpointsCard(
                            address: manager.addresses.first,
                            port: port,
                            onCopyURL: copyURL
                        )
                    }
                        .padding(.horizontal, LASDesignTokens.pageInset)
                        .padding(.top, 12)
                        .padding(.bottom, 28)
                    }
                    .scrollIndicators(.hidden)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(isPresented: $showingDownloads) {
                LASModelDownloadsView()
            }
            .task {
                portText = String(port)
                systemStatus.startObserving()
                if serverEnabled {
                    await manager.start()
                }
                if autoLoadModel {
                    loadModel()
                }
            }
            .onChange(of: serverEnabled) { _, enabled in
                Task {
                    if enabled {
                        await manager.start()
                    } else {
                        await manager.stop()
                    }
                }
            }
            .onChange(of: port) { _, newPort in
                portText = String(newPort)
            }
            .onChange(of: keepScreenAwake) { _, _ in
                manager.refreshIdleTimerPolicy()
            }
            .onChange(of: autoLoadModel) { _, enabled in
                if enabled {
                    loadModel()
                }
            }
            .onChange(of: scenePhase) { _, phase in
                if phase != .active {
                    // Do not leave a bearer credential visible in the app
                    // switcher snapshot or on a locked screen.
                    apiKeyVisible = false
                }
            }
            .onReceive(modelService.$state.removeDuplicates()) { state in
                guard state != lastLoggedModelState else { return }
                lastLoggedModelState = state
                RuntimeLogCenter.shared.append(
                    modelStateMessage(state),
                    subsystem: "model"
                )
            }
            .onDisappear {
                apiKeyVisible = false
                systemStatus.stopObserving()
            }
            .sheet(isPresented: $showingModelPicker) {
                LASModelPickerView(
                    currentModel: modelService.activeModel,
                    onSelect: handleModelSelection,
                    onOpenModels: {
                        showingModelPicker = false
                        showingDownloads = true
                    }
                )
            }
            .alert("Rotate API key?", isPresented: $showingKeyRotationConfirmation) {
                Button("Rotate", role: .destructive) {
                    manager.rotateKey()
                    copyMessage = "New key generated"
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Existing clients will stop working until they use the new key.")
            }
        }
    }

    private func restartServer() {
        Task { await manager.restart() }
    }

    private func loadModel() {
        if openDownloadsIfNeeded(for: modelService.activeModel) {
            return
        }
        modelService.startLoad()
    }

    private func handleModelSelection(_ model: AssistantModel) {
        if openDownloadsIfNeeded(for: model) {
            showingModelPicker = false
            return
        }

        showingModelPicker = false
        modelService.startSwitchTo(model)
    }

    private func openDownloadsIfNeeded(for model: AssistantModel) -> Bool {
        let center = ModelDownloadCenter.shared
        center.refreshAllStates()

        guard let downloadable = center.models.first(where: {
            $0.id == model.id || $0.sourceRepoID == model.repoID
        }), !downloadable.isReady else {
            return false
        }

        // A cold selection needs the download surface so the user can see
        // resumable file progress while the model is fetched. The local
        // server parser stays on Home because it represents runtime activity,
        // not file-transfer progress.
        downloadable.start()
        showingDownloads = true
        return true
    }

    private func applyPort() {
        let normalized = portText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let candidate = Int(normalized),
              LocalAPIValidation.validPort(candidate) != nil else {
            copyMessage = "Port must be 1024–65535"
            return
        }
        port = candidate
        copyMessage = "Port saved"
        guard serverEnabled else { return }
        Task { await manager.restart() }
    }

    private func copyURL(_ value: String) {
        UIPasteboard.general.string = value
        copyMessage = "URL copied"
    }

    private func copyKey() {
        UIPasteboard.general.string = manager.apiKey
        copyMessage = "API key copied"
    }

    private func copySetupCommands() {
        let base = manager.addresses.first.map { "http://\($0):\(port)" } ?? "http://<DEVICE_IP>:\(port)"
        UIPasteboard.general.string = """
        export OPENAI_BASE_URL=\(base)/v1
        export OPENAI_API_KEY=\(manager.apiKey)
        curl "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
        """
        copyMessage = "Setup commands copied with API key"
    }

    private func modelStateMessage(_ state: CodingAssistantService.ServiceState) -> String {
        switch state {
        case .unloaded:
            return "Model unloaded"
        case .loading(let message):
            return "Model loading: \(message)"
        case .ready:
            return "Model ready: \(modelService.activeDisplayName)"
        case .generating:
            return "Model serving an API request"
        case .failed(let message):
            return "Model failed: \(message)"
        }
    }
}

private struct LASHomeHeader: View {
    var body: some View {
        HStack(spacing: LASDesignTokens.row) {
            Text("On Device : LAS")
                .font(.title3.weight(.bold))
                .lineLimit(1)

            Spacer(minLength: LASDesignTokens.tight)

            HStack(spacing: 0) {
                NavigationLink {
                    LASThemeSettingsView()
                } label: {
                    Image(systemName: "paintpalette")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Theme")

                LASHomeHeaderDivider()

                NavigationLink {
                    LASModelDownloadsView()
                } label: {
                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Model downloads")

                LASHomeHeaderDivider()

                NavigationLink {
                    LASOnDeviceDebuggerView()
                } label: {
                    Image(systemName: "ladybug")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("On-device debugger")

                LASHomeHeaderDivider()

                NavigationLink {
                    LASTerminalView()
                } label: {
                    Image(systemName: "terminal")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Verbose terminal")
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 4)
            .glassSurface(.capsule)
            .overlay {
                Capsule().stroke(LASDesignTokens.hairline, lineWidth: 1)
            }
        }
        .padding(.horizontal, LASDesignTokens.pageInset)
        .frame(minHeight: 64)
        .background(.ultraThinMaterial.opacity(0.55))
    }
}

private struct LASHomeHeaderDivider: View {
    var body: some View {
        Rectangle()
            .fill(LASDesignTokens.hairline)
            .frame(width: 1, height: 20)
    }
}

private struct LASHealthDashboard: View {
    let snapshot: SystemStatusService.Snapshot
    let thermalState: ProcessInfo.ThermalState
    let lowPowerMode: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            LASSectionLabel(
                title: "DEVICE HEALTH",
                trailing: lowPowerMode ? "LOW POWER" : thermalLabel.uppercased()
            )

            HStack(spacing: 0) {
                LASCompactHealthMetric(
                    title: "Thermal",
                    value: thermalLabel,
                    symbol: "thermometer.medium"
                )
                LASCompactHealthDivider()
                LASCompactHealthMetric(
                    title: "Available",
                    value: MemoryAdvisor.availableRAM.formattedBytes,
                    symbol: "memorychip"
                )
                LASCompactHealthDivider()
                LASCompactHealthMetric(
                    title: "Headroom",
                    value: snapshot.availableForML.formattedBytes,
                    symbol: "gauge.with.dots.needle.67percent"
                )
            }
            .padding(.vertical, LASDesignTokens.tight)
            .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 18))

            HStack(spacing: LASDesignTokens.component) {
                Label("App \(snapshot.usedByApp.formattedBytes)", systemImage: "chart.bar.fill")
                Spacer(minLength: 4)
                Label("Free \(snapshot.diskFree.formattedBytes)", systemImage: "internaldrive")
            }
            .font(.caption.monospaced())
            .foregroundStyle(.secondary)
        }
        .lasCard(radius: LASDesignTokens.cardRadius, padding: LASDesignTokens.component)
        .accessibilityElement(children: .contain)
    }

    private var thermalLabel: String {
        switch thermalState {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Warm"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

}

private struct LASCompactHealthMetric: View {
    let title: String
    let value: String
    let symbol: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Label(title, systemImage: symbol)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.78)
            Text(verbatim: value)
                .font(.caption.weight(.semibold).monospaced())
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
        .frame(maxWidth: .infinity, minHeight: 54, alignment: .leading)
        .padding(.horizontal, LASDesignTokens.tight)
    }
}

private struct LASCompactHealthDivider: View {
    var body: some View {
        Rectangle()
            .fill(LASDesignTokens.hairline)
            .frame(width: 1, height: 38)
    }
}

/// A single parser for the model that serves the local API. Download rows have
/// their own file-progress dots; this card deliberately reflects runtime work
/// instead: loading, waiting for a request, or generating a response.
private struct LASLocalServerParserCard: View {
    let serverState: LocalAPIManager.State
    let modelState: CodingAssistantService.ServiceState
    let modelName: String
    let tokenRate: Double

    var body: some View {
        let activity = LASLocalServerActivity(
            serverState: serverState,
            modelState: modelState
        )
        LiveParserBar(
            isGenerating: activity == .generating,
            status: activity.parserStatus,
            badge: activity.badgeLabel,
            tokenRate: tokenRate
        )
        .accessibilityHint(activity.detail(modelName: modelName))
    }
}

private enum LASLocalServerActivity: Equatable {
    case offline
    case connecting
    case listening
    case ready
    case preparing
    case generating
    case issue

    init(
        serverState: LocalAPIManager.State,
        modelState: CodingAssistantService.ServiceState
    ) {
        switch serverState {
        case .failed:
            self = .issue
        case .starting:
            self = .connecting
        case .stopped:
            self = .offline
        case .running:
            switch modelState {
            case .generating: self = .generating
            case .loading: self = .preparing
            case .failed: self = .issue
            case .unloaded: self = .listening
            case .ready: self = .ready
            }
        }
    }

    var parserStatus: String {
        switch self {
        case .offline: return "Offline"
        case .connecting: return "Connecting"
        case .listening: return "Listening"
        case .ready: return "Ready"
        case .preparing: return "Preparing"
        case .generating: return "Parsing"
        case .issue: return "Attention"
        }
    }

    var badgeLabel: String {
        switch self {
        case .offline: return "OFFLINE"
        case .connecting: return "STARTING"
        case .listening: return "NO MODEL"
        case .ready: return "READY"
        case .preparing: return "LOADING"
        case .generating: return "ACTIVE"
        case .issue: return "ISSUE"
        }
    }

    func detail(modelName: String) -> String {
        switch self {
        case .offline:
            return "Start the local server to accept requests"
        case .connecting:
            return "Opening local endpoint"
        case .listening:
            return "Load a model to serve requests"
        case .ready:
            return "Waiting for local requests"
        case .preparing:
            return "Loading selected model"
        case .generating:
            return "Streaming local response"
        case .issue:
            return "Open the debugger for details"
        }
    }
}
private struct LASNetworkBanner: View {
    let state: LocalAPIManager.State
    let modelState: CodingAssistantService.ServiceState
    let address: String?
    let port: Int

    var body: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(statusColor)
                .frame(width: 10, height: 10)
            Text(verbatim: endpointText)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.72)
                .allowsTightening(true)
            Spacer(minLength: 4)
            Image(systemName: "network")
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .glassSurface(.card, cornerRadius: 17)
        .overlay {
            RoundedRectangle(cornerRadius: 17, style: .continuous)
                .stroke(LASDesignTokens.hairline, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
    }

    private var endpointText: String {
        address.map { "Local API server · \($0):" + String(port) }
            ?? "Local API server · waiting for LAN"
    }

    private var statusColor: Color {
        switch state {
        case .running:
            switch modelState {
            case .ready, .generating: return .green
            case .loading: return .blue
            case .unloaded: return .orange
            case .failed: return .red
            }
        case .starting: return .blue
        case .failed: return .red
        case .stopped: return .gray
        }
    }
}

private struct LASModelCapabilitySection: View {
    let profile: ModelCapabilityProfile

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            LASSectionLabel(title: "MODEL PROFILE")

            valueRow("Family", profile.family.displayName)
            valueRow("Context length", profile.configuredContextLength.formatted() + " tokens")
            valueRow("KV-cache capacity", profile.maximumKVCacheTokens.formatted() + " tokens")
            valueRow("Maximum output", profile.maximumOutputTokens.formatted() + " tokens")
            valueRow("Estimated full KV", profile.estimatedKVCacheBytes.formattedBytes)
            valueRow("Tool parser", profile.toolParser.displayName)
            valueRow("Reasoning parser", profile.reasoningParser.displayName)
            valueRow("Conversation memory", "Automatic at 72%")

            if let warning = profile.hermesWarning {
                Label(warning, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(LASDesignTokens.component)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))
            }
        }
        .accessibilityElement(children: .contain)
    }

    private func valueRow(_ label: String, _ value: String) -> some View {
        LabeledContent(label) {
            Text(value)
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}

private struct LASHomeServerCard: View {
    let state: LocalAPIManager.State
    @Binding var serverEnabled: Bool
    @Binding var keepScreenAwake: Bool
    @Binding var autoLoadModel: Bool
    @Binding var toolCallingEnabled: Bool
    @Binding var reasoningEnabled: Bool
    @Binding var parallelToolCallsLimit: Int
    @Binding var strictToolSchemasEnabled: Bool
    @Binding var portText: String

    let modelName: String
    let modelID: String
    let repoID: String
    let capabilityProfile: ModelCapabilityProfile
    let modelState: CodingAssistantService.ServiceState
    let addresses: [String]
    let port: Int
    let apiKey: String
    @Binding var apiKeyVisible: Bool
    let copyMessage: String?
    let onRestart: () -> Void
    let onApplyPort: () -> Void
    let onLoadModel: () -> Void
    let onChooseModel: () -> Void
    let onStopModel: () -> Void
    let onCopyURL: (String) -> Void
    let onCopyKey: () -> Void
    let onRotateKey: () -> Void
    let onCopySetup: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.component) {
            VStack(alignment: .leading, spacing: LASDesignTokens.component) {
                intro
                LASCardDivider()
                serverControls
            }
            .lasCard(radius: LASDesignTokens.majorRadius)

            modelSection
                .lasCard()

            LASModelCapabilitySection(profile: capabilityProfile)
                .lasCard()

            settingsSection
                .lasCard()

            connectionSection
                .lasCard()

            VStack(alignment: .leading, spacing: LASDesignTokens.section) {
                apiKeySection
                LASCardDivider()
                quickStartSection
                LASCardDivider()
                securityNote
            }
            .lasCard()
        }
    }

    private var intro: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "globe.americas.fill")
                .font(.system(size: 30))
                .foregroundStyle(.primary)
                .frame(width: 42)
            VStack(alignment: .leading, spacing: 4) {
                Text("Local API Server")
                    .font(.title3.weight(.bold))
                Text("OpenAI + Anthropic + Ollama · local network only")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var serverControls: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Enable Local API Server")
                    .font(.headline)
                Spacer()
                Toggle("Enable Local API Server", isOn: $serverEnabled)
                    .labelsHidden()
                    .tint(.black)
                    .accessibilityLabel("Enable Local API Server")
            }

            HStack(spacing: 10) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(statusText)
                    .font(.system(.subheadline, design: .monospaced))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Restart", systemImage: "arrow.clockwise", action: onRestart)
                    .buttonStyle(LASSecondaryButtonStyle())
                    .disabled(!serverEnabled)
            }
        }
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.component) {
            LASSectionLabel(
                title: "MODEL RUNTIME",
                trailing: modelStatusLabel,
                trailingColor: modelStatusColor
            )

            HStack(alignment: .center, spacing: LASDesignTokens.row) {
                ZStack {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.primary.opacity(0.055))
                    Image(systemName: modelStatusSymbol)
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundStyle(modelStatusColor)
                }
                .frame(width: 52, height: 52)
                .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 4) {
                    Text(modelName)
                        .font(.headline)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(modelStateTitle)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(modelStatusColor)
                        .lineLimit(2)
                    Text(repoID)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                        .textSelection(.enabled)
                }
                Spacer(minLength: 4)
                Button {
                    UIPasteboard.general.string = modelID
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 15, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .background(Color.primary.opacity(0.055), in: Circle())
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Copy model ID")
            }

            HStack(spacing: LASDesignTokens.tight) {
                modelActionButton
                chooseModelButton
            }
        }
    }

    @ViewBuilder
    private var modelActionButton: some View {
        if isModelReady {
            Button("Unload", systemImage: "eject", action: onStopModel)
                .buttonStyle(LASModelSecondaryButtonStyle())
                .accessibilityHint("Remove the selected model from memory")
        } else if isModelLoading {
            Button("Stop loading", systemImage: "xmark", action: onStopModel)
                .buttonStyle(LASModelSecondaryButtonStyle(foreground: .orange))
        } else {
            Button("Load model", systemImage: "play.fill", action: onLoadModel)
                .buttonStyle(LASModelPrimaryButtonStyle())
        }
    }

    private var chooseModelButton: some View {
        Button("Choose", systemImage: "arrow.triangle.2.circlepath", action: onChooseModel)
            .buttonStyle(LASModelSecondaryButtonStyle())
    }

    private var parallelToolCallsExplanation: String {
        guard toolCallingEnabled else {
            return "Enable tool calling to use this setting."
        }

        let setting = LocalAPIParallelToolCallsSetting(rawValue: parallelToolCallsLimit)
            ?? .automaticTwo
        if setting == .sequential {
            return "Tool calls run one at a time."
        }

        let modelLimit = capabilityProfile.parallelTools.maximumCalls
        if modelLimit < setting.maximumCalls {
            return "This model profile limits parallel calls to \(modelLimit)."
        }
        return "The active model may emit up to \(setting.maximumCalls) independent calls."
    }

    private var selectedParallelToolCalls: LocalAPIParallelToolCallsSetting {
        LocalAPIParallelToolCallsSetting(rawValue: parallelToolCallsLimit) ?? .automaticTwo
    }

    private var settingsSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            LASSectionLabel(title: "COMPATIBILITY")

            Toggle("Keep screen awake while server runs", isOn: $keepScreenAwake)
                .frame(minHeight: 44)
            Toggle("Load selected model when app opens", isOn: $autoLoadModel)
                .frame(minHeight: 44)
            Toggle("Tool calling", isOn: $toolCallingEnabled)
                .frame(minHeight: 44)
            Toggle("Allow model reasoning", isOn: $reasoningEnabled)
                .frame(minHeight: 44)
            Menu {
                ForEach(LocalAPIParallelToolCallsSetting.allCases) { setting in
                    Button {
                        parallelToolCallsLimit = setting.rawValue
                    } label: {
                        if setting.rawValue == parallelToolCallsLimit {
                            Label(setting.displayName, systemImage: "checkmark")
                        } else {
                            Text(setting.displayName)
                        }
                    }
                }
            } label: {
                HStack(spacing: LASDesignTokens.tight) {
                    Text("Parallel tool calls")
                    Spacer(minLength: LASDesignTokens.tight)
                    Text(selectedParallelToolCalls.displayName)
                        .foregroundStyle(.secondary)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                .contentShape(Rectangle())
            }
            .disabled(!toolCallingEnabled)
            .accessibilityLabel("Parallel tool calls")
            .accessibilityValue(selectedParallelToolCalls.displayName)
            .accessibilityHint("Choose the maximum number of independent tool calls the active model may emit")

            Text(parallelToolCallsExplanation)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Toggle("Strict tool-schema validation", isOn: $strictToolSchemasEnabled)
                .disabled(!toolCallingEnabled)
                .frame(minHeight: 44)

            Text("Reasoning is parsed separately from final content. Tool calls are validated after parsing and malformed calls receive one bounded repair pass.")
                .font(.caption)
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: LASDesignTokens.tight) {
                Text("Port")
                    .font(.subheadline.weight(.semibold))
                HStack(spacing: LASDesignTokens.tight) {
                    TextField("11434", text: $portText)
                        .keyboardType(.numberPad)
                        .font(.body.monospaced())
                        .textFieldStyle(.roundedBorder)
                        .frame(minHeight: 44)
                        .accessibilityLabel("Local API server port")
                    Button("Apply", action: onApplyPort)
                        .buttonStyle(LASSecondaryButtonStyle())
                        .disabled(!canApplyPort)
                }
                if !portText.isEmpty, Int(portText) == nil {
                    Text("Enter a numeric port from 1024 through 65535.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }
        }
        .toggleStyle(.switch)
        .tint(.black)
    }

    private var connectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            LASSectionLabel(title: "CONNECTION URLS")

            if addresses.isEmpty {
                Text("Waiting for a Wi‑Fi or LAN address…")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(connectionEntries) { entry in
                    LASURLRow(label: entry.label, value: entry.value) {
                        onCopyURL(entry.value)
                    }
                }
            }
        }
    }

    private var connectionEntries: [LASConnectionEntry] {
        var seen = Set<String>()
        var entries: [LASConnectionEntry] = []
        for address in addresses {
            let openAI = "http://\(address):\(port)/v1"
            let localBase = "http://\(address):\(port)"
            if seen.insert(openAI).inserted {
                entries.append(LASConnectionEntry(label: "OpenAI", value: openAI))
            }
            // Ollama and Anthropic use the same local base URL. Keep one
            // copyable row instead of presenting the identical URL twice.
            if seen.insert(localBase).inserted {
                entries.append(LASConnectionEntry(label: "Ollama · Anthropic", value: localBase))
            }
        }
        return entries
    }

    private var apiKeySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            LASSectionLabel(title: "API KEY")

            HStack(spacing: 10) {
                Text(apiKeyVisible ? apiKey : maskedKey)
                    .font(.footnote.monospaced())
                    .lineLimit(1)
                    .minimumScaleFactor(0.65)
                    .textSelection(.enabled)
                    .privacySensitive()
                Spacer(minLength: 4)
                Button(apiKeyVisible ? "Hide" : "Reveal") {
                    apiKeyVisible.toggle()
                }
                .buttonStyle(.borderless)
                Button("Copy", action: onCopyKey)
                    .buttonStyle(.borderless)
                Button("Rotate", action: onRotateKey)
                    .buttonStyle(.borderless)
            }

            if let copyMessage {
                Text(copyMessage)
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        }
    }

    private var quickStartSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "LINUX QUICK START")

            Text(setupCommands)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)

            ViewThatFits {
                HStack(spacing: LASDesignTokens.tight) {
                    setupCopyButton
                    setupShareButton
                }
                VStack(alignment: .leading, spacing: LASDesignTokens.tight) {
                    setupCopyButton
                    setupShareButton
                }
            }

            Text("Copy and Share include the current API key. Send them only to a device you trust.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("For Ollama SDKs, use the Ollama URL above and send the same Authorization bearer header.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var securityNote: some View {
        Label {
            Text("HTTP traffic is not encrypted. Enable this only on a trusted LAN. The server stops when iOS backgrounds the app.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } icon: {
            Image(systemName: "exclamationmark.shield.fill")
                .foregroundStyle(.orange)
        }
    }

    private var isModelReady: Bool {
        switch modelState {
        case .ready, .generating: return true
        case .unloaded, .loading, .failed: return false
        }
    }

    private var isModelLoading: Bool {
        if case .loading = modelState { return true }
        return false
    }

    private var modelStateTitle: String {
        switch modelState {
        case .unloaded: return "Load model for API requests"
        case .loading(let message): return message
        case .ready: return "Ready for API requests"
        case .generating: return "Serving API request"
        case .failed(let message): return message
        }
    }

    private var modelStatusLabel: String {
        if isModelReady { return "RESIDENT" }
        if isModelLoading { return "LOADING" }
        if case .failed = modelState { return "ISSUE" }
        return "IDLE"
    }

    private var modelStatusSymbol: String {
        if isModelReady { return "checkmark" }
        if isModelLoading { return "ellipsis" }
        if case .failed = modelState { return "exclamationmark" }
        return "cpu"
    }

    private var modelStatusColor: Color {
        if isModelReady { return .green }
        if isModelLoading { return .blue }
        if case .failed = modelState { return .red }
        return .secondary
    }

    private var statusText: String {
        switch state {
        case .running(let port):
            return isModelReady
                ? "ready on port \(String(port))"
                : "listening on port \(String(port)) · no model loaded"
        case .starting: return "starting server"
        case .failed(let message): return message
        case .stopped: return "server is stopped"
        }
    }

    private var statusColor: Color {
        switch state {
        case .running:
            if isModelReady { return .green }
            if isModelLoading { return .blue }
            return .orange
        case .starting: return .blue
        case .failed: return .red
        case .stopped: return .secondary
        }
    }

    private var maskedKey: String {
        String(repeating: "•", count: max(12, min(apiKey.count, 24)))
    }

    private var canApplyPort: Bool {
        guard let candidate = Int(portText), (1024...65535).contains(candidate) else {
            return false
        }
        return candidate != port
    }

    private var setupCopyButton: some View {
        Button("Copy setup", systemImage: "doc.on.doc", action: onCopySetup)
            .buttonStyle(LASSecondaryButtonStyle())
    }

    private var setupShareButton: some View {
        ShareLink(
            item: setupCommandsWithAPIKey,
            subject: Text("On Device LAS setup commands")
        ) {
            Label("Share setup", systemImage: "square.and.arrow.up")
        }
        .buttonStyle(LASSecondaryButtonStyle())
    }

    private var setupCommands: String {
        let base = addresses.first.map { "http://\($0):\(port)" } ?? "http://<DEVICE_IP>:\(port)"
        return "export OPENAI_BASE_URL=\(base)/v1\nexport OPENAI_API_KEY=<API_KEY>\ncurl \"$OPENAI_BASE_URL/models\" -H \"Authorization: Bearer $OPENAI_API_KEY\""
    }

    private var setupCommandsWithAPIKey: String {
        let base = addresses.first.map { "http://\($0):\(port)" } ?? "http://<DEVICE_IP>:\(port)"
        return "export OPENAI_BASE_URL=\(base)/v1\nexport OPENAI_API_KEY=\(apiKey)\ncurl \"$OPENAI_BASE_URL/models\" -H \"Authorization: Bearer $OPENAI_API_KEY\""
    }
}

private struct LASModelPrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        LASThemedPrimaryLabel(configuration: configuration)
    }

    private struct LASThemedPrimaryLabel: View {
        let configuration: Configuration
        @Environment(\.koduTheme) private var T

        var body: some View {
            configuration.label
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, minHeight: 48)
                .background(
                    T.accentStrong.opacity(configuration.isPressed ? 0.76 : 1),
                    in: RoundedRectangle(cornerRadius: 16, style: .continuous)
                )
                .scaleEffect(configuration.isPressed ? 0.985 : 1)
        }
    }
}

private struct LASModelSecondaryButtonStyle: ButtonStyle {
    var foreground: Color = .primary

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(foreground)
            .frame(maxWidth: .infinity, minHeight: 48)
            .background(
                Color.primary.opacity(configuration.isPressed ? 0.11 : 0.055),
                in: RoundedRectangle(cornerRadius: 16, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(LASDesignTokens.hairline, lineWidth: 1)
            }
    }
}

private struct LASConnectionEntry: Identifiable {
    let label: String
    let value: String

    var id: String { "\(label)|\(value)" }
}

private struct LASModelPickerView: View {
    let currentModel: AssistantModel
    let onSelect: (AssistantModel) -> Void
    let onOpenModels: () -> Void
    @ObservedObject private var modelCenter = ModelDownloadCenter.shared
    @Environment(\.dismiss) private var dismiss

    /// Only models whose complete files are already on-device belong in the
    /// load picker. The catalog is intentionally not a fallback here: its
    /// entries describe models that *can* be downloaded, not models that the
    /// local API can load right now. Imported models use the same
    /// `DownloadableModel` path after registration, so they appear alongside
    /// downloaded catalog models once their validator reports `.ready`.
    private var models: [AssistantModel] {
        var result: [AssistantModel] = []
        var seenRepoIDs = Set<String>()

        for downloadable in modelCenter.models
            where downloadable.isReady
                // The picker forces the text-runtime descriptor below. Do
                // not rely on the catalog's primary category here: some
                // text models expose a generic *ForConditionalGeneration
                // architecture and are classified as VLM during disk scan.
                // Voice and image-only packages are the only categories that
                // cannot be API assistant candidates.
                && downloadable.category != .voice
                && downloadable.category != .imageGen {
            guard let model = LocalModelRegistry
                .descriptor(for: downloadable, forcedRole: .assistant)
                .assistantModel else { continue }

            let repoKey = model.repoID.lowercased()
            guard seenRepoIDs.insert(repoKey).inserted else { continue }
            result.append(model)
        }

        return result.sorted { lhs, rhs in
            let lhsIsCurrent = lhs.id == currentModel.id
            let rhsIsCurrent = rhs.id == currentModel.id
            if lhsIsCurrent != rhsIsCurrent { return lhsIsCurrent }
            return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName)
                == .orderedAscending
        }
    }

    var body: some View {
        NavigationStack {
            List {
                if models.isEmpty {
                    VStack(spacing: 12) {
                        ContentUnavailableView(
                            "No ready models",
                            systemImage: "internaldrive",
                            description: Text(
                                "Download or import a model from the Models tab before loading the local API."
                            )
                        )
                        Button("Open Models", systemImage: "arrow.down.circle", action: onOpenModels)
                            .buttonStyle(.borderedProminent)
                    }
                    .frame(maxWidth: .infinity)
                    .listRowSeparator(.hidden)
                } else {
                    ForEach(models) { model in
                        Button {
                            onSelect(model)
                        } label: {
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(model.displayName)
                                        .font(.body.weight(.semibold))
                                        .foregroundStyle(.primary)
                                    Text(model.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if model.id == currentModel.id {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(.green)
                                }
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .navigationTitle("Choose API model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .task {
                modelCenter.refreshAllStates()
            }
        }
    }
}

private struct LASURLRow: View {
    let label: String
    let value: String
    let action: () -> Void

    var body: some View {
        ViewThatFits {
            HStack(spacing: LASDesignTokens.row) {
                Text(label)
                    .font(.subheadline.weight(.semibold))
                Text(value)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
                copyButton
            }
            VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
                HStack {
                    Text(label)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    copyButton
                }
                Text(value)
                    .font(.footnote.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
        }
        .frame(minHeight: 48)
    }

    private var copyButton: some View {
        Button(action: action) {
            Image(systemName: "doc.on.doc")
                .foregroundStyle(.secondary)
                .frame(width: 44, height: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .accessibilityLabel("Copy \(label) URL")
    }
}

private struct LASDeveloperEndpoint: Identifiable {
    let method: String
    let path: String

    var id: String { "\(method) \(path)" }
}

private struct LASDeveloperEndpointsCard: View {
    let address: String?
    let port: Int
    let onCopyURL: (String) -> Void

    private static let openAIEndpoints = [
        LASDeveloperEndpoint(method: "GET", path: "/v1"),
        LASDeveloperEndpoint(method: "GET", path: "/v1/models"),
        LASDeveloperEndpoint(method: "GET", path: "/v1/models/{model}"),
        LASDeveloperEndpoint(method: "POST", path: "/v1/chat/completions"),
        LASDeveloperEndpoint(method: "POST", path: "/v1/responses")
    ]
    private static let anthropicEndpoints = [
        LASDeveloperEndpoint(method: "POST", path: "/v1/messages")
    ]
    private static let ollamaEndpoints = [
        LASDeveloperEndpoint(method: "GET", path: "/api/tags"),
        LASDeveloperEndpoint(method: "GET", path: "/api/ps"),
        LASDeveloperEndpoint(method: "GET", path: "/api/version"),
        LASDeveloperEndpoint(method: "POST", path: "/api/show"),
        LASDeveloperEndpoint(method: "POST", path: "/api/chat"),
        LASDeveloperEndpoint(method: "POST", path: "/api/generate")
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: "chevron.left.forwardslash.chevron.right")
                    .font(.title2.weight(.semibold))
                    .frame(width: 34)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Developer endpoints")
                        .font(.headline)
                    Text("Complete LAN URLs for OpenCode, OpenAI, Anthropic, and Ollama clients.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            LASDeveloperEndpointSection(
                title: "OPENAI · OPENCODE",
                baseURL: baseURL,
                endpoints: Self.openAIEndpoints,
                onCopyURL: onCopyURL
            )

            LASDeveloperEndpointSection(
                title: "ANTHROPIC",
                baseURL: baseURL,
                endpoints: Self.anthropicEndpoints,
                onCopyURL: onCopyURL
            )

            LASDeveloperEndpointSection(
                title: "OLLAMA",
                baseURL: baseURL,
                endpoints: Self.ollamaEndpoints,
                onCopyURL: onCopyURL
            )

            Label(
                "All routes require the API key through Authorization: Bearer or X-API-Key.",
                systemImage: "key.horizontal"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .lasCard(radius: 24, padding: 20)
    }

    private var baseURL: String {
        "http://\(address ?? "<DEVICE_IP>"):\(port)"
    }
}

private struct LASDeveloperEndpointSection: View {
    let title: LocalizedStringResource
    let baseURL: String
    let endpoints: [LASDeveloperEndpoint]
    let onCopyURL: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.caption.weight(.bold).monospaced())
                .foregroundStyle(.secondary)

            ForEach(endpoints) { endpoint in
                LASDeveloperEndpointRow(
                    method: endpoint.method,
                    url: baseURL + endpoint.path,
                    onCopyURL: onCopyURL
                )
            }
        }
    }
}

private struct LASDeveloperEndpointRow: View {
    let method: String
    let url: String
    let onCopyURL: (String) -> Void

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            Text(method)
                .font(.caption2.weight(.bold).monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 34, alignment: .leading)

            Text(url)
                .font(.caption.monospaced())
                .foregroundStyle(.primary)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)

            Spacer(minLength: 0)

            Button {
                onCopyURL(url)
            } label: {
                Image(systemName: "doc.on.doc")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Copy \(method) endpoint URL")
        }
    }
}

private struct LASCardDivider: View {
    var body: some View {
        Divider()
            .overlay(LASDesignTokens.hairline)
    }
}
