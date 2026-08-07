import SwiftUI

/// Server dashboard for the Core AI product — tabbed operator surface with
/// Overview, Models, Server, and Diagnostics.
struct CoreAIDashboardView: View {
    @ObservedObject private var manager = LocalAPIManager.shared
    @ObservedObject private var assistant = CodingAssistantService.shared
    @ObservedObject private var store = CoreAIModelStore.shared
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.koduTheme) private var T

    @State private var tab: Tab = .overview
    @State private var keyVisible = false
    @State private var copiedFeedback: String?
    @State private var portText = ""
    @State private var sharePayload: SharePayload?

    private enum Tab: Hashable {
        case overview, models, server, diagnostics
    }

    /// Items handed to the system share sheet (Messages / AirDrop / Notes).
    private struct SharePayload: Identifiable {
        let id = UUID()
        let subject: String
        let body: String
    }

    var body: some View {
        TabView(selection: $tab) {
            overviewTab
                .tabItem { Label("Overview", systemImage: "dot.radiowaves.left.and.right") }
                .tag(Tab.overview)

            NavigationStack {
                CoreAIModelManagementView()
            }
            .tabItem { Label("Models", systemImage: "square.stack.3d.up.fill") }
            .tag(Tab.models)

            serverTab
                .tabItem { Label("Server", systemImage: "server.rack") }
                .tag(Tab.server)

            NavigationStack {
                LASTerminalView()
            }
            .tabItem { Label("Diagnostics", systemImage: "terminal") }
            .tag(Tab.diagnostics)
        }
        .tint(T.accentStrong)
        .task {
            portText = String(settings.localAPIPort)
            if settings.localAPIEnabled {
                await manager.start()
            }
            if settings.localAPIAutoLoadModel {
                assistant.startLoad()
            }
            manager.refreshIdleTimerPolicy()
        }
        .onChange(of: settings.localAPIKeepScreenAwake) { _, _ in
            manager.refreshIdleTimerPolicy()
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active {
                keyVisible = false
            }
        }
    }

    private var overviewTab: some View {
        NavigationStack {
            ZStack {
                LASPageBackground()
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        header
                        networkBanner
                        LiveParserBar(
                            isGenerating: assistant.state == .generating,
                            status: parserStatus,
                            badge: parserBadge,
                            tokenRate: assistant.tokenRate
                        )
                        .frame(maxWidth: .infinity)
                        modelQuickCard
                        serverQuickCard
                        lifecycleNotice
                    }
                    .padding(.horizontal, LASDesignTokens.pageInset)
                    .padding(.bottom, 32)
                }
                .scrollIndicators(.hidden)
            }
            .navigationBarHidden(true)
            .sheet(item: $sharePayload) { payload in
                CoreAIActivityShareSheet(
                    items: [payload.body],
                    subject: payload.subject
                )
            }
        }
    }

    private var serverTab: some View {
        NavigationStack {
            ZStack {
                LASPageBackground()
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        serverCard
                        shareCard
                        controlsCard
                        compatibilityCard
                        endpointsCard
                    }
                    .padding(.horizontal, LASDesignTokens.pageInset)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle("Server")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(item: $sharePayload) { payload in
                CoreAIActivityShareSheet(
                    items: [payload.body],
                    subject: payload.subject
                )
            }
        }
    }

    private var header: some View {
        HStack(spacing: LASDesignTokens.row) {
            Image(systemName: "cpu.fill")
                .font(.title2.weight(.semibold))
                .foregroundStyle(.white)
                .frame(width: 44, height: 44)
                .background(
                    LinearGradient(
                        colors: [Color(red: 0.08, green: 0.35, blue: 0.22), .black],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: 13, style: .continuous)
                )

            VStack(alignment: .leading, spacing: 2) {
                Text("Core AI: LAS")
                    .font(.title3.weight(.bold))
                Text("LOCAL API · iOS 27")
                    .font(.caption2.weight(.bold).monospaced())
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: LASDesignTokens.tight)

            NavigationLink {
                LASOnDeviceDebuggerView()
            } label: {
                Image(systemName: "ladybug")
                    .font(.system(size: 18, weight: .semibold))
                    .frame(width: 44, height: 44)
            }
            .buttonStyle(LASHeaderIconButtonStyle(dark: false))
            .accessibilityLabel("On-device debugger")

            NavigationLink {
                LASThemeSettingsView()
            } label: {
                Image(systemName: "paintpalette")
                    .font(.system(size: 18, weight: .semibold))
                    .frame(width: 44, height: 44)
            }
            .buttonStyle(LASHeaderIconButtonStyle(dark: false))
            .accessibilityLabel("Theme")

            Button {
                tab = .diagnostics
            } label: {
                Image(systemName: "terminal")
                    .font(.system(size: 18, weight: .semibold))
                    .frame(width: 44, height: 44)
            }
            .buttonStyle(LASHeaderIconButtonStyle(dark: false))
            .accessibilityLabel("Verbose terminal")
        }
        .padding(.top, 8)
        .frame(minHeight: 56)
    }

    private var networkBanner: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 10, height: 10)
                Text(endpointText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.72)
                Spacer(minLength: 4)
                Image(systemName: "network")
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                Button {
                    copyString(primaryIP, feedback: "IP copied")
                } label: {
                    Label("Copy IP", systemImage: "doc.on.doc")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASSecondaryButtonStyle())
                .disabled(manager.addresses.isEmpty)

                Button {
                    copyString(openaiBaseURL, feedback: "API URL copied")
                } label: {
                    Label("Copy API", systemImage: "link")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASSecondaryButtonStyle())

                Button {
                    copyString(manager.apiKey, feedback: "API key copied")
                } label: {
                    Label("Copy key", systemImage: "key")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(LASSecondaryButtonStyle())
            }

            Button {
                presentShare(
                    subject: "Core AI: LAS connection",
                    body: connectionShareText
                )
            } label: {
                Label("Share connection + API key", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(LASPrimaryButtonStyle())
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 13)
        .glassSurface(.card, cornerRadius: 17)
        .overlay {
            RoundedRectangle(cornerRadius: 17, style: .continuous)
                .stroke(LASDesignTokens.hairline, lineWidth: 1)
        }
    }

    private var modelQuickCard: some View {
        Button { tab = .models } label: {
            VStack(alignment: .leading, spacing: LASDesignTokens.row) {
                HStack {
                    LASSectionLabel(title: "MODEL")
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundStyle(.secondary)
                }
                Text(assistant.activeDisplayName)
                    .font(.headline)
                    .foregroundStyle(.primary)
                Text(modelStateText)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                HStack(spacing: 10) {
                    switch assistant.state {
                    case .unloaded, .failed:
                        Button("Load model") { assistant.startLoad() }
                            .buttonStyle(LASPrimaryButtonStyle())
                    case .loading:
                        ProgressView().controlSize(.small)
                    case .ready, .generating:
                        Button("Unload") {
                            Task { await assistant.unloadAndWaitForCleanup() }
                        }
                        .buttonStyle(LASSecondaryButtonStyle())
                    }
                    Spacer(minLength: 0)
                }
            }
        }
        .buttonStyle(.plain)
        .lasCard()
    }

    private var serverQuickCard: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            LASSectionLabel(title: "SERVER", trailing: serverStateText.uppercased())
            Toggle(isOn: Binding(
                get: { settings.localAPIEnabled },
                set: { enabled in
                    settings.localAPIEnabled = enabled
                    Task {
                        if enabled { await manager.start() }
                        else { await manager.stop() }
                    }
                }
            )) {
                Label(serverStateText, systemImage: serverSymbol)
                    .font(.headline)
            }
            Button("Open server settings") { tab = .server }
                .buttonStyle(LASSecondaryButtonStyle())
        }
        .lasCard()
    }

    private var serverCard: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            LASSectionLabel(title: "LISTENER", trailing: serverStateText.uppercased())

            Toggle(isOn: Binding(
                get: { settings.localAPIEnabled },
                set: { enabled in
                    settings.localAPIEnabled = enabled
                    Task {
                        if enabled { await manager.start() }
                        else { await manager.stop() }
                    }
                }
            )) {
                Label(serverStateText, systemImage: serverSymbol)
                    .font(.headline)
            }

            HStack {
                Text("Port")
                Spacer()
                TextField("11434", text: $portText)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 88)
                    .onSubmit { applyPort() }
                Button("Apply") { applyPort() }
                    .buttonStyle(LASSecondaryButtonStyle())
            }

            if !manager.addresses.isEmpty {
                ForEach(manager.addresses, id: \.self) { address in
                    let url = "http://\(address):\(settings.localAPIPort)"
                    Button {
                        copyString(url, feedback: "Copied \(url)")
                    } label: {
                        HStack {
                            Text(url)
                                .font(.caption.monospaced())
                                .foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: "doc.on.doc")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }

            apiKeyRow

            HStack(spacing: 10) {
                Button("Restart") {
                    Task { await manager.restart() }
                }
                .buttonStyle(LASPrimaryButtonStyle())

                Button("Rotate API key", systemImage: "arrow.clockwise") {
                    manager.rotateKey()
                    keyVisible = true
                    copyString(manager.apiKey, feedback: "New API key copied")
                }
                .buttonStyle(LASSecondaryButtonStyle())
            }

            if let copiedFeedback {
                Text(copiedFeedback)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Toggle("Auto-load model on foreground", isOn: $settings.localAPIAutoLoadModel)
                .font(.subheadline)
            Toggle("Keep screen awake while serving", isOn: $settings.localAPIKeepScreenAwake)
                .font(.subheadline)

            Text("Trusted LAN only · plain HTTP bearer auth. Do not expose this port to the public internet.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .lasCard()
    }

    private var shareCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "SHARE CONNECTION")

            Text(quickSharePreview)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

            Button {
                presentShare(
                    subject: "Core AI: LAS connection",
                    body: connectionShareText
                )
            } label: {
                Label("Share connection + API key", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(LASPrimaryButtonStyle())

            Button {
                copyString(connectionShareText, feedback: "Connection + API key copied")
            } label: {
                Label("Copy connection + API key", systemImage: "doc.on.doc")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(LASSecondaryButtonStyle())

            ViewThatFits {
                HStack(spacing: 8) {
                    shareIPButton
                    shareAPIButton
                }
                VStack(spacing: 8) {
                    shareIPButton
                    shareAPIButton
                }
            }

            Text("Share connection includes the API key, so clients can authenticate immediately. Only send it to a device you trust.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .lasCard()
    }

    private var apiKeyRow: some View {
        VStack(alignment: .leading, spacing: 6) {
            LASSectionLabel(title: "API KEY")
            HStack(spacing: 8) {
                Group {
                    if keyVisible {
                        Text(manager.apiKey)
                            .textSelection(.enabled)
                    } else {
                        Text(String(repeating: "•", count: 24))
                    }
                }
                .font(.footnote.monospaced())
                .foregroundStyle(.primary)
                .lineLimit(1)
                .minimumScaleFactor(0.55)
                .privacySensitive()
                Spacer(minLength: 4)
                Button(keyVisible ? "Hide" : "Reveal") {
                    keyVisible.toggle()
                }
                .buttonStyle(LASSecondaryButtonStyle())
                Button("Copy") {
                    copyString(manager.apiKey, feedback: "API key copied")
                }
                .buttonStyle(LASSecondaryButtonStyle())
            }
        }
    }

    private var shareIPButton: some View {
        Button {
            presentShare(subject: "Core AI: LAS IP", body: primaryIP)
        } label: {
            Label("Share IP", systemImage: "network")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(LASSecondaryButtonStyle())
        .disabled(manager.addresses.isEmpty)
    }

    private var shareAPIButton: some View {
        Button {
            presentShare(subject: "Core AI: LAS API", body: openaiBaseURL)
        } label: {
            Label("Share API URL", systemImage: "link")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(LASSecondaryButtonStyle())
    }

    private var controlsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "GENERATION")
            Toggle("Tool calling", isOn: $settings.localAPIToolCallingEnabled)
            Toggle("Reasoning", isOn: $settings.localAPIReasoningEnabled)
            Toggle("Parallel tool calls", isOn: $settings.localAPIParallelToolCallsEnabled)
            Toggle("Strict tool schemas", isOn: $settings.localAPIStrictToolSchemasEnabled)
            Stepper(
                "Max output tokens · \(settings.assistantMaxTokens)",
                value: $settings.assistantMaxTokens,
                in: 256...8_192,
                step: 256
            )
            .font(.subheadline)
        }
        .lasCard()
    }

    private var compatibilityCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "COMPATIBILITY")
            ForEach(
                [
                    "OpenAI · /v1/chat/completions",
                    "Responses · /v1/responses",
                    "Anthropic · /v1/messages",
                    "Ollama · /api/chat"
                ],
                id: \.self
            ) { route in
                Label(route, systemImage: "checkmark.circle")
                    .font(.caption.monospaced())
            }
            Text("Unsupported Core AI capabilities return explicit protocol errors — never silent remote fallback.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .lasCard()
    }

    private var endpointsCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            LASSectionLabel(title: "QUICK START")
            Text("curl \(openaiBaseURL)/models -H \"Authorization: Bearer \(keyVisible ? manager.apiKey : "…")\"")
                .font(.caption2.monospaced())
                .textSelection(.enabled)
            Button("Copy setup commands") {
                copyString(setupCommandsWithAPIKey, feedback: "Setup commands copied")
            }
            .buttonStyle(LASSecondaryButtonStyle())
        }
        .lasCard()
    }

    private var lifecycleNotice: some View {
        Label(
            "Inference stops when iOS suspends the app and recovers after foregrounding.",
            systemImage: "pause.circle"
        )
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 4)
    }

    private func applyPort() {
        guard let port = UInt16(portText), port >= 1_024 else {
            portText = String(settings.localAPIPort)
            return
        }
        settings.localAPIPort = Int(port)
        Task { await manager.restart() }
    }

    private func copyString(_ value: String, feedback: String) {
        UIPasteboard.general.string = value
        copiedFeedback = feedback
        ToastCenter.shared.success(feedback)
    }

    private func presentShare(subject: String, body: String) {
        sharePayload = SharePayload(subject: subject, body: body)
    }

    private var primaryIP: String {
        manager.addresses.first ?? "<device-ip>"
    }

    private var httpBaseURL: String {
        "http://\(primaryIP):\(settings.localAPIPort)"
    }

    private var openaiBaseURL: String {
        "\(httpBaseURL)/v1"
    }

    /// Functional connection block — includes the bearer key so a shared
    /// message can authenticate without a second round-trip.
    private var connectionShareText: String {
        """
        Core AI: LAS — on-device AI API
        IP: \(primaryIP)
        Port: \(settings.localAPIPort)
        OpenAI base: \(openaiBaseURL)
        Ollama base: \(httpBaseURL)
        API key: \(manager.apiKey)

        curl \(openaiBaseURL)/models -H "Authorization: Bearer \(manager.apiKey)"
        """
    }

    private var quickSharePreview: String {
        """
        IP   \(primaryIP)
        API  \(openaiBaseURL)
        KEY  \(keyVisible ? manager.apiKey : String(repeating: "•", count: 18))
        """
    }

    private var setupCommandsWithAPIKey: String {
        """
        export OPENAI_BASE_URL=\(openaiBaseURL)
        export OPENAI_API_KEY=\(manager.apiKey)
        curl "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
        """
    }

    private var endpointText: String {
        manager.addresses.first.map { "Local API · \($0):\(settings.localAPIPort)" }
            ?? "Local API · waiting for LAN"
    }

    private var statusColor: Color {
        switch manager.state {
        case .running:
            switch assistant.state {
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

    private var serverStateText: String {
        switch manager.state {
        case .stopped: return "Stopped"
        case .starting: return "Starting…"
        case .running(let port): return "Listening on \(port)"
        case .failed(let message): return "Failed · \(message)"
        }
    }

    private var serverSymbol: String {
        if case .running = manager.state { return "checkmark.circle.fill" }
        return "circle"
    }

    private var modelStateText: String {
        switch assistant.state {
        case .unloaded: return storeStatusText
        case .loading(let message): return message
        case .ready: return "Loaded and ready for Core AI generation"
        case .generating: return "Generating on device"
        case .failed(let message): return message
        }
    }

    private var storeStatusText: String {
        switch store.state {
        case .missing: return "No model resource directory installed"
        case .downloading: return "Downloading model resources"
        case .validating: return "Validating model resources"
        case .ready: return "Installed · open Models to manage"
        case .unavailable(let message), .failed(let message): return message
        }
    }

    private var parserStatus: String {
        switch assistant.state {
        case .generating: return "STREAMING"
        case .ready: return "READY"
        case .loading: return "LOADING"
        case .failed: return "FAILED"
        case .unloaded: return "IDLE"
        }
    }

    private var parserBadge: String {
        switch assistant.state {
        case .generating: return "CORE AI"
        case .ready: return "ON DEVICE"
        default: return "WAITING"
        }
    }
}

/// Thin UIKit share sheet wrapper so Overview/Server can hand off IP + API
/// strings to AirDrop / Messages without nesting `ShareLink` in menus.
private struct CoreAIActivityShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    var subject: String?

    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        if let subject {
            controller.setValue(subject, forKey: "subject")
        }
        return controller
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct CoreAIOnboardingView: View {
    @State private var page = 0

    private let pages = [
        (
            "LOCAL-FIRST INFERENCE",
            "Your model.\nYour endpoint.",
            "Run a compatible Core AI resource directory on your iPhone and expose it only to trusted devices on your local network."
        ),
        (
            "ZOO + HUGGING FACE",
            "Install the\nfull model pack.",
            "Browse curated Core AI zoo packs, search Hugging Face, pause and resume downloads, then load the validated resource folder."
        ),
        (
            "CLIENT COMPATIBILITY",
            "Familiar APIs.\nHonest limits.",
            "OpenAI, Responses, Anthropic, and Ollama-compatible routes remain available. Background suspension and unsupported capabilities are reported explicitly."
        )
    ]

    var body: some View {
        ZStack {
            LASPageBackground()

            VStack(spacing: 0) {
                HStack {
                    Image(systemName: "cpu.fill")
                    Text("CORE AI : LAS")
                        .font(.headline.weight(.semibold))
                    Spacer()
                    Text("iOS 27")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                .padding(20)

                TabView(selection: $page) {
                    ForEach(pages.indices, id: \.self) { index in
                        VStack(alignment: .leading, spacing: 20) {
                            Spacer()
                            Text(pages[index].0)
                                .font(.caption.weight(.bold).monospaced())
                                .foregroundStyle(.secondary)
                            Text(pages[index].1)
                                .font(.system(size: 36, weight: .bold, design: .rounded))
                            Text(pages[index].2)
                                .font(.body)
                                .foregroundStyle(.secondary)
                                .lineSpacing(5)
                            Spacer()
                        }
                        .padding(24)
                        .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .always))

                Button(page == pages.count - 1 ? "Enter Core AI LAS" : "Next") {
                    if page == pages.count - 1 {
                        AppSettings.shared.hasSeenOnboarding = true
                    } else {
                        page += 1
                    }
                }
                .buttonStyle(LASPrimaryButtonStyle())
                .controlSize(.large)
                .frame(maxWidth: .infinity)
                .padding(20)
            }
        }
    }
}
