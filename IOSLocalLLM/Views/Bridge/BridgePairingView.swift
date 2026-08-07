import SwiftUI

// MARK: - BridgePairingView
//
// New pairing flow (Mac shows QR → iPhone scans it):
//   1. Mac app generates a pairing QR and displays it on screen.
//   2. User taps "Scan Mac QR" here, points the iPhone at the Mac screen.
//   3. iPhone reads the QR, starts its local inference server, and POSTs the
//      iPhone's connection details to the Mac's pairing endpoint.
//   4. Mac stores the details and can now forward inference requests to this
//      iPhone via the bearer token included in the POST.
//
// Protocol (Mac QR JSON):
//   { "v": 1, "macHost": "192.168.x.x", "pairingPort": 8444,
//     "nonce": "<uuid>", "macName": "MacBook Pro" }

struct BridgePairingView: View {

    @ObservedObject private var manager   = BridgeManager.shared
    @ObservedObject private var assistant = CodingAssistantService.shared
    @ObservedObject private var settings  = AppSettings.shared
    @Environment(\.koduTheme) private var T

    @State private var showScanner = false

    /// Copy for the orange banner shown when the assistant isn't
    /// ready to serve Mac-bound inference. Per-state so users get an
    /// accurate diagnosis instead of the catch-all "load a model
    /// first" line, which used to fire mid-generation and looked like
    /// a bug.
    private var bannerCopy: String {
        switch assistant.state {
        case .unloaded:
            return "Load a model first — the Mac needs a model resident in memory to run inference."
        case .loading(let msg):
            return "Model loading… \(msg). The Mac can pair now but inference will queue until ready."
        case .failed(let err):
            return "Model failed to load: \(err). Open the Assistant tab to retry."
        case .ready, .generating:
            // Should never display in these states — isModelLoaded
            // guards the banner — but kept for exhaustiveness so a
            // future state addition is a compile error here too.
            return ""
        }
    }

    var body: some View {
        // NavigationStack so the new ArgentRemoteView push works.
        // Previously the Mac tab was a flat ScrollView; wrapping
        // doesn't affect existing layout — the inner ScrollView
        // still owns its scroll axis.
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    header
                    serverStatusRow
                    LocalAPIServerCard()
                    macBridgeIntegrationCard
                    pairingCard
                    insecurePairingCard
                    pairedClientsCard
                    // Phase-1 agent surface. Only renders when at least
                    // one paired Mac has issued an agent bearer to us —
                    // legacy pairings without `macBearerToken` get nothing
                    // here so the UI doesn't promise something the wire
                    // can't deliver. Re-pairing fixes it.
                    if manager.pairedClients.contains(where: { $0.macBearerToken != nil }) {
                        BridgeAgentCard()
                        argentRemoteRow
                    }
                }
                .padding(20)
            }
            .background(LiquidPinkBackdrop())
            // Start the iPhone's inference server as soon as the tab opens so it's
            // ready by the time the QR is scanned and the Mac tries to connect.
            .task {
                if manager.serverState == .stopped { await manager.start() }
            }
            .sheet(isPresented: $showScanner) {
                ScannerSheet(showScanner: $showScanner)
            }
        }
    }

    // MARK: - Insecure pairing opt-in
    //
    // A Mac that doesn't advertise a TLS cert fingerprint pairs over plaintext
    // http, so the nonce + bearer travel in the clear on the LAN. pairWithMac()
    // refuses such a Mac unless this is on; surfaced here so the opt-in the
    // failure message points to actually exists and is discoverable.

    private var insecurePairingCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: $settings.allowInsecureBridgePairing) {
                HStack(spacing: 10) {
                    Image(systemName: "lock.open.trianglebadge.exclamationmark")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundColor(settings.allowInsecureBridgePairing ? T.warn : T.ink3)
                    Text("Allow insecure pairing")
                        .font(T.sans(15, .semibold))
                        .foregroundColor(T.ink)
                }
            }
            .tint(T.accent)

            Text("Only for older Mac apps without encrypted pairing. When on, the pairing handshake (including the access token) is sent over plaintext on your local network. Leave off unless a Mac fails to pair.")
                .font(T.sans(12))
                .foregroundColor(T.ink3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .kGlass(cornerRadius: 16, fallbackFill: T.surface)
    }

    // MARK: - Mac bridge integration guide
    //
    // The public source distribution documents the protocol instead of
    // directing contributors to a binary companion whose source and release
    // provenance are not part of this repository.

    private var macBridgeIntegrationCard: some View {
        Button {
            if let url = URL(string: "https://github.com/Mesutcydev/ios-local-llm/blob/main/Docs/AGENT_INTEGRATION.md") {
                UIApplication.shared.open(url)
            }
            HapticManager.impact(.light)
        } label: {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(
                            LinearGradient(
                                colors: [T.accent.opacity(0.22),
                                         T.accent.opacity(0.08)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    Image(systemName: "book.closed.fill")
                        .font(.system(size: 22, weight: .semibold))
                        .foregroundColor(T.accent)
                }
                .frame(width: 48, height: 48)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text("Mac bridge integration")
                            .font(T.mono(13, .semibold))
                            .foregroundColor(T.ink)
                        Text("OPEN GUIDE")
                            .font(T.mono(9, .semibold))
                            .foregroundColor(T.ink3)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                Capsule().fill(T.surface3)
                            )
                    }
                    Text("Review the public pairing protocol, security boundaries, and client implementation notes.")
                        .font(T.mono(10.5))
                        .foregroundColor(T.ink3)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Image(systemName: "arrow.up.right.square")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(T.ink3)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .kGlass(cornerRadius: 14, fallbackFill: T.surface)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Open Mac bridge integration guide")
        .accessibilityHint("Opens the public protocol documentation on GitHub.")
    }

    // MARK: - Argent Remote launcher
    //
    // Pushes ArgentRemoteView when tapped. Only shown when at least
    // one paired Mac exists with an agent bearer — same gating as
    // BridgeAgentCard above. The destination view does its own
    // argent.status probe on appear, so we don't preflight here;
    // tapping when argent isn't installed on the Mac shows the
    // "not installed" card with install instructions.

    private var argentRemoteRow: some View {
        NavigationLink {
            ArgentRemoteView()
        } label: {
            HStack(spacing: 10) {
                Image(systemName: "iphone.gen3.radiowaves.left.and.right")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundColor(T.accent)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Argent Remote")
                        .font(T.mono(13, .semibold))
                        .foregroundColor(T.ink)
                    Text("One-tap simulator control — screenshot, tap, swipe, launch app, …")
                        .font(T.mono(10.5))
                        .foregroundColor(T.ink3)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(T.ink3)
            }
            .padding(16)
            .frame(maxWidth: .infinity)
            .kGlass(cornerRadius: 14, fallbackFill: T.surface)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Header

    // Design-language hero (matches Home / the local LLM foundation handoff). The
    // Mac tab isn't in the design files, so this is built from the same
    // primitives: pink gradient, white type, a live status dot, soft bloom.
    private var header: some View {
        let running: Bool = { if case .running = manager.serverState { return true } else { return false } }()
        return VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Circle()
                    .fill(running ? T.good : Color.white.opacity(0.7))
                    .frame(width: 7, height: 7)
                    .shadow(color: running ? T.good.opacity(0.9) : .clear, radius: 4)
                Text("MAC BRIDGE")
                    .font(T.sans(12, .bold)).tracking(0.8)
                    .foregroundColor(.white)
                Spacer()
                Image(systemName: "desktopcomputer")
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundColor(.white.opacity(0.92))
            }
            Text("Control from your Mac")
                .font(T.display(23, .bold))
                .foregroundColor(.white)
                .padding(.top, 12)
            Text("Pair a Mac to run its prompts on this iPhone — nothing leaves your devices.")
                .font(T.sans(14))
                .foregroundColor(.white.opacity(0.92))
                .padding(.top, 3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            ZStack {
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(LinearGradient(colors: [T.roseHi, T.accent, T.accentStrong],
                                         startPoint: .topLeading, endPoint: .bottomTrailing))
                Circle().fill(Color.white.opacity(0.16))
                    .frame(width: 150, height: 150).blur(radius: 6)
                    .offset(x: 120, y: -70)
            }
            .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        )
    }

    // MARK: - Server status row

    private var serverStatusRow: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusLabel)
                .font(T.mono(13, .regular))
                .foregroundColor(T.ink2)
                .lineLimit(1)
            Spacer()
        }
        .padding(12)
        .kGlass(cornerRadius: 10, fallbackFill: T.surface)
    }

    private var statusColor: Color {
        switch manager.serverState {
        case .running:  return .green
        case .starting: return .yellow
        case .failed:   return .red
        case .stopped:  return T.ink4
        }
    }

    private var statusLabel: String {
        switch manager.serverState {
        case .running(let host, let port):
            return "Mac pairing bridge (TLS) · \(host):\(port)"
        case .starting:
            return "Mac pairing bridge starting…"
        case .failed(let msg):
            return "Mac pairing bridge error: \(msg)"
        case .stopped:
            return "Mac pairing bridge stopped"
        }
    }

    // MARK: - Pairing card

    @ViewBuilder
    private var pairingCard: some View {
        VStack(spacing: 16) {
            switch manager.pairingPhase {

            case .idle, .scanning:
                idleContent

            case .pairing:
                pairingContent

            case .paired(let macName):
                pairedContent(macName: macName)

            case .failed(let msg):
                failedContent(msg: msg)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(16)
        .kGlass(cornerRadius: 14, fallbackFill: T.surface)
    }

    // ── Idle: prompt to scan ──────────────────────────────────────────────

    private var idleContent: some View {
        VStack(spacing: 14) {
            Image(systemName: "qrcode.viewfinder")
                .font(.system(size: 52, weight: .thin))
                .foregroundColor(T.ink3)

            VStack(spacing: 6) {
                Text("Pair with Mac")
                    .font(T.mono(15, .semibold))
                    .foregroundColor(T.ink)
                Text("Open iOS Local LLM Bridge on your Mac, click \"Show QR\", then scan it here.")
                    .font(T.mono(12, .regular))
                    .foregroundColor(T.ink3)
                    .multilineTextAlignment(.center)
            }

            // Banner ONLY when there's genuinely no model in memory.
            // `.generating` and `.ready` both count as "loaded" — see
            // CodingAssistantService.isModelLoaded. Previously this
            // also fired on `.generating`, `.loading`, and `.failed`,
            // which made the banner appear right after a normal
            // generation completed and confused users who could see
            // their model name in the picker. The copy is also
            // narrower now: it states the precondition (model resident
            // in memory) and the single action that fixes it.
            if !assistant.isModelLoaded {
                Label(bannerCopy,
                      systemImage: "exclamationmark.triangle")
                    .font(T.mono(11, .regular))
                    .foregroundColor(.orange)
                    .multilineTextAlignment(.center)
            }

            Button {
                if BridgeQRScannerView.isAvailable {
                    manager.pairingPhase = .scanning
                    showScanner = true
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "camera")
                    Text("Scan Mac QR")
                }
                .font(T.mono(13, .semibold))
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(RoundedRectangle(cornerRadius: 10).fill(T.accent))
            }
            .buttonStyle(.plain)
            .disabled(!BridgeQRScannerView.isAvailable)
        }
    }

    // ── Pairing in progress ───────────────────────────────────────────────

    private var pairingContent: some View {
        VStack(spacing: 12) {
            ProgressView()
                .progressViewStyle(.circular)
                .tint(T.accent)
            Text("Pairing with Mac…")
                .font(T.mono(13, .regular))
                .foregroundColor(T.ink2)
        }
        .padding(.vertical, 24)
    }

    // ── Paired ───────────────────────────────────────────────────────────

    private func pairedContent(macName: String) -> some View {
        VStack(spacing: 14) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 44))
                .foregroundColor(T.good)

            VStack(spacing: 4) {
                Text("Paired with \(macName)")
                    .font(T.mono(14, .semibold))
                    .foregroundColor(T.ink)
                Text("The Mac can now send inference requests to this iPhone.")
                    .font(T.mono(11, .regular))
                    .foregroundColor(T.ink3)
                    .multilineTextAlignment(.center)
            }

            Button("Pair Another Mac") {
                manager.pairingPhase = .idle
            }
            .font(T.mono(12, .regular))
            .foregroundColor(T.accent)
        }
        .padding(.vertical, 8)
    }

    // ── Failed ────────────────────────────────────────────────────────────

    private func failedContent(msg: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "xmark.circle")
                .font(.largeTitle)
                .foregroundColor(T.bad)

            Text(msg)
                .font(T.mono(12, .regular))
                .foregroundColor(T.ink3)
                .multilineTextAlignment(.center)

            Button("Try Again") {
                manager.pairingPhase = .idle
            }
            .font(T.mono(13, .medium))
            .foregroundColor(T.accent)
        }
        .padding(.vertical, 8)
    }

    // MARK: - Paired clients card

    @ViewBuilder
    private var pairedClientsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("PAIRED MACS")
                    .font(T.mono(11, .semibold))
                    .foregroundColor(T.ink4)
                Spacer()
                if !manager.pairedClients.isEmpty {
                    Button("Forget All") { manager.forgetAllClients() }
                        .font(T.mono(12, .regular))
                        .foregroundColor(.red)
                }
            }

            if manager.pairedClients.isEmpty {
                Text("No Macs paired yet. Scan a Mac QR above.")
                    .font(T.mono(12, .regular))
                    .foregroundColor(T.ink3)
            } else {
                ForEach(manager.pairedClients, id: \.token) { client in
                    HStack(spacing: 10) {
                        Image(systemName: "desktopcomputer")
                            .foregroundColor(T.accent)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(client.clientName)
                                .font(T.mono(13, .medium))
                                .foregroundColor(T.ink)
                            Text("Paired \(client.pairedAt.formatted(date: .abbreviated, time: .omitted))")
                                .font(T.mono(11, .regular))
                                .foregroundColor(T.ink3)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .kGlass(cornerRadius: 14, fallbackFill: T.surface)
    }
}

// MARK: - Cross-platform local API

private struct LocalAPIServerCard: View {
    @ObservedObject private var manager = LocalAPIManager.shared
    @ObservedObject private var settings = AppSettings.shared
    @ObservedObject private var assistant = CodingAssistantService.shared
    @Environment(\.koduTheme) private var theme

    @State private var portText = String(AppSettings.shared.localAPIPort)
    @State private var revealsKey = false
    @State private var showDownloadedModelPicker = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header
            Toggle("Enable Local API Server", isOn: enabledBinding)
                .font(theme.sans(14, .semibold))
                .tint(theme.accent)
            status
            modelStatus
            Toggle("Keep screen awake while server runs", isOn: keepAwakeBinding)
                .font(theme.sans(13, .medium))
                .tint(theme.accent)
            portEditor
            if case .running(let port) = manager.state {
                endpointList(port: port)
                keyControls
                setupExamples(port: port)
            }
            warning
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .kGlass(cornerRadius: 16, fallbackFill: theme.surface)
        .onAppear {
            portText = String(settings.localAPIPort)
            manager.refreshAddresses()
        }
        .sheet(isPresented: $showDownloadedModelPicker) {
            AssistantModelPickerView(downloadedOnly: true)
        }
    }

    private var keepAwakeBinding: Binding<Bool> {
        Binding(
            get: { settings.localAPIKeepScreenAwake },
            set: { enabled in
                settings.localAPIKeepScreenAwake = enabled
                manager.refreshIdleTimerPolicy()
            }
        )
    }

    private var modelStatus: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: modelStatusSymbol)
                .foregroundColor(modelStatusColor)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 2) {
                Text(assistant.activeModel.displayName)
                    .font(theme.sans(13, .semibold))
                    .foregroundColor(theme.ink)
                Text(modelStatusText)
                    .font(theme.mono(10))
                    .foregroundColor(modelStatusColor)
            }
            Spacer()
            if canLoadModel {
                Button {
                    Task { await assistant.load() }
                    HapticManager.impact(.medium)
                } label: {
                    Text("Load")
                        .font(theme.mono(10, .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Capsule().fill(theme.accent))
                }
                .buttonStyle(.plain)
            }
            Button {
                showDownloadedModelPicker = true
                HapticManager.impact(.light)
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(theme.ink)
                    .frame(width: 30, height: 30)
                    .background(Circle().fill(theme.surface2))
            }
            .buttonStyle(.plain)
            .disabled(isModelBusy)
            .accessibilityLabel("Choose downloaded model")
        }
    }

    private var canLoadModel: Bool {
        switch assistant.state {
        case .unloaded, .failed: true
        default: false
        }
    }

    private var isModelBusy: Bool {
        switch assistant.state {
        case .loading, .generating: true
        default: false
        }
    }

    private var modelStatusText: String {
        switch assistant.state {
        case .unloaded: return "Not loaded"
        case .loading(let detail): return detail.isEmpty ? "Loading…" : "Loading · \(detail)"
        case .ready: return "Ready for API requests"
        case .generating: return "Generating · server busy"
        case .failed(let message): return "Load failed · \(message)"
        }
    }

    private var modelStatusSymbol: String {
        switch assistant.state {
        case .ready: return "checkmark.circle.fill"
        case .generating: return "waveform"
        case .loading: return "arrow.triangle.2.circlepath"
        case .failed: return "exclamationmark.triangle.fill"
        case .unloaded: return "circle.dashed"
        }
    }

    private var modelStatusColor: Color {
        switch assistant.state {
        case .ready: return theme.good
        case .generating, .loading: return theme.warn
        case .failed: return theme.bad
        case .unloaded: return theme.ink3
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "network")
                .font(.system(size: 17, weight: .semibold))
                .foregroundColor(theme.accent)
            VStack(alignment: .leading, spacing: 2) {
                Text("Local API Server")
                    .font(theme.sans(16, .bold))
                    .foregroundColor(theme.ink)
                Text("OpenAI + Anthropic + Ollama · separate from the Mac pairing bridge")
                    .font(theme.mono(9.5))
                    .foregroundColor(theme.ink3)
            }
            Spacer()
        }
    }

    private var enabledBinding: Binding<Bool> {
        Binding(
            get: { settings.localAPIEnabled },
            set: { enabled in
                settings.localAPIEnabled = enabled
                Task {
                    if enabled { await manager.start() }
                    else { await manager.stop() }
                }
            }
        )
    }

    private var status: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 7, height: 7)
            Text(statusText)
                .font(theme.mono(10.5))
                .foregroundColor(theme.ink2)
                .textSelection(.enabled)
            Spacer()
            if settings.localAPIEnabled {
                Button("Restart") {
                    Task { await manager.restart() }
                }
                .font(theme.mono(10, .semibold))
                .foregroundColor(theme.accent)
            }
        }
    }

    private var statusColor: Color {
        switch manager.state {
        case .running: return theme.good
        case .starting: return theme.warn
        case .failed: return theme.bad
        case .stopped: return theme.ink4
        }
    }

    private var statusText: String {
        switch manager.state {
        case .running(let port): return "running on port \(port)"
        case .starting: return "starting…"
        case .failed(let message): return "error: \(message)"
        case .stopped: return "stopped"
        }
    }

    private var portEditor: some View {
        HStack(spacing: 10) {
            Text("Port")
                .font(theme.sans(13, .medium))
                .foregroundColor(theme.ink2)
            TextField("11434", text: $portText)
                .keyboardType(.numberPad)
                .textFieldStyle(.roundedBorder)
                .font(theme.mono(12))
            Button("Apply") {
                guard let port = Int(portText),
                      LocalAPIValidation.validPort(port) != nil else {
                    ToastCenter.shared.error("Invalid port", detail: "Use a port from 1024 through 65535.")
                    return
                }
                // Persist first so app lifecycle restarts and the listener both
                // read the same value. Previously this happened inside the
                // asynchronous restart task, making Apply appear to do nothing.
                settings.localAPIPort = port
                portText = String(port)
                if settings.localAPIEnabled {
                    Task {
                        await manager.restart()
                        if case .running(let activePort) = manager.state,
                           activePort == UInt16(port) {
                            ToastCenter.shared.success(
                                "Local API port changed",
                                detail: "Now listening on \(activePort)."
                            )
                        }
                    }
                } else {
                    ToastCenter.shared.success(
                        "Local API port saved",
                        detail: "The server will use \(port) when enabled."
                    )
                }
            }
            .font(theme.mono(10, .semibold))
            .foregroundColor(theme.accent)
        }
    }

    private func endpointList(port: UInt16) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CONNECTION URLS")
                .font(theme.mono(9, .bold))
                .tracking(0.6)
                .foregroundColor(theme.ink3)
            if manager.addresses.isEmpty {
                Text("No reachable local IPv4 address. Connect both devices to the same network.")
                    .font(theme.sans(12))
                    .foregroundColor(theme.warn)
            } else {
                ForEach(Array(connectionEntries(port: port).enumerated()), id: \.offset) { _, entry in
                    CopyableAPIEndpoint(
                        label: entry.label,
                        value: entry.value,
                        theme: theme
                    )
                }
            }
        }
    }

    private func connectionEntries(port: UInt16) -> [(label: String, value: String)] {
        var seen = Set<String>()
        var entries: [(label: String, value: String)] = []
        for address in manager.addresses {
            let openAI = "http://\(address):\(port)/v1"
            let localBase = "http://\(address):\(port)"
            if seen.insert(openAI).inserted {
                entries.append(("OpenAI", openAI))
            }
            if seen.insert(localBase).inserted {
                entries.append(("Ollama · Anthropic", localBase))
            }
        }
        return entries
    }

    private var keyControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("API KEY")
                .font(theme.mono(9, .bold))
                .tracking(0.6)
                .foregroundColor(theme.ink3)
            HStack(spacing: 8) {
                Text(revealsKey ? manager.apiKey : String(repeating: "•", count: 24))
                    .font(theme.mono(10))
                    .foregroundColor(theme.ink2)
                    .lineLimit(1)
                    .textSelection(.enabled)
                Spacer()
                Button(revealsKey ? "Hide" : "Reveal") { revealsKey.toggle() }
                Button("Copy") {
                    UIPasteboard.general.string = manager.apiKey
                    ToastCenter.shared.success("API key copied")
                }
                Button("Rotate") {
                    manager.rotateKey()
                    ToastCenter.shared.info("API key rotated", detail: "Update clients before their next request.")
                }
            }
            .font(theme.mono(9.5, .semibold))
            .foregroundColor(theme.accent)
        }
    }

    private func setupExamples(port: UInt16) -> some View {
        let host = manager.addresses.first ?? "IPHONE_IP"
        let command = """
        export OPENAI_BASE_URL=http://\(host):\(port)/v1
        export OPENAI_API_KEY=\(manager.apiKey)
        curl "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
        """
        return VStack(alignment: .leading, spacing: 7) {
            Text("LINUX QUICK START")
                .font(theme.mono(9, .bold))
                .tracking(0.6)
                .foregroundColor(theme.ink3)
            Text(command)
                .font(theme.mono(9.5))
                .foregroundColor(theme.ink2)
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(theme.surface3, in: RoundedRectangle(cornerRadius: 8))
            Button("Copy setup commands") {
                UIPasteboard.general.string = command
                ToastCenter.shared.success("Linux setup copied")
            }
            .font(theme.mono(10, .semibold))
            .foregroundColor(theme.accent)
            Text("For Ollama SDKs, use the Ollama URL above as the host and send the same Authorization bearer header.")
                .font(theme.sans(11))
                .foregroundColor(theme.ink3)
        }
    }

    private var warning: some View {
        Label {
            Text("HTTP traffic is not encrypted. Enable this only on a trusted LAN. The server stops when iOS backgrounds the app.")
                .font(theme.sans(11))
                .foregroundColor(theme.ink3)
        } icon: {
            Image(systemName: "exclamationmark.shield")
                .foregroundColor(theme.warn)
        }
    }
}

private struct CopyableAPIEndpoint: View {
    let label: String
    let value: String
    let theme: KoduTheme

    var body: some View {
        Button {
            UIPasteboard.general.string = value
            ToastCenter.shared.success("\(label) URL copied")
        } label: {
            HStack(spacing: 8) {
                Text(label)
                    .font(theme.mono(9, .bold))
                    .foregroundColor(theme.accent)
                    .frame(width: 45, alignment: .leading)
                Text(value)
                    .font(theme.mono(9.5))
                    .foregroundColor(theme.ink2)
                    .lineLimit(1)
                Spacer()
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(theme.ink3)
            }
            .padding(9)
            .background(theme.surface3, in: RoundedRectangle(cornerRadius: 8))
        }
        .buttonStyle(.plain)
    }
}

// MARK: - ScannerSheet

private struct ScannerSheet: View {
    @Binding var showScanner: Bool
    @ObservedObject private var manager = BridgeManager.shared
    @Environment(\.koduTheme) private var T

    var body: some View {
        ZStack(alignment: .top) {
            if BridgeQRScannerView.isAvailable {
                BridgeQRScannerView { payload in
                    showScanner = false
                    Task { await manager.pairWithMac(qrJSON: payload) }
                }
                .ignoresSafeArea()
                // Belt-and-suspenders: if dismantleUIViewController doesn't
                // fire (e.g. SwiftUI keeps the VC alive across a sheet
                // animation), the .id() forces a fresh instance every time
                // the sheet opens so a leaked previous instance can't bleed
                // its overlay into the next presentation.
                .id(showScanner ? "scanner-on" : "scanner-off")
            } else {
                LiquidPinkBackdrop()
                Text("Camera scanner not available on this device.")
                    .font(T.mono(14, .regular))
                    .foregroundColor(T.ink2)
                    .multilineTextAlignment(.center)
                    .padding()
            }

            // Dismiss button
            HStack {
                Spacer()
                Button {
                    showScanner = false
                    manager.pairingPhase = .idle
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 28))
                        .foregroundColor(.white.opacity(0.85))
                        .shadow(radius: 4)
                }
                .buttonStyle(.plain)
                .padding(20)
            }

            // Instruction overlay
            VStack {
                Spacer()
                Text("Point at the QR code on your Mac")
                    .font(T.mono(13, .semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.55).clipShape(Capsule()))
                    .padding(.bottom, 60)
            }
        }
    }
}
