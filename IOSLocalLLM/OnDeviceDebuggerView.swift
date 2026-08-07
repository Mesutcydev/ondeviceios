import SwiftUI
import UIKit

/// The server-only app's live load inspector. It combines the model loader
/// trace, current memory/thermal admission figures, and the safe runtime log
/// so a native MLX load never looks like an unexplained frozen screen.
struct LASOnDeviceDebuggerView: View {
    @ObservedObject private var modelService = CodingAssistantService.shared
    @ObservedObject private var logCenter = RuntimeLogCenter.shared
    @ObservedObject private var systemStatus = SystemStatusService.shared
    @ObservedObject private var safetyMonitor = DeviceSafetyMonitor.shared

    @State private var now = Date()
    @State private var diagnosticEntries: [DiagEntry] = []

    private let refreshTimer = Timer.publish(
        every: 1,
        on: .main,
        in: .common
    ).autoconnect()

    var body: some View {
        VStack(spacing: 0) {
            LASDetailHeader("On-device debugger") {
                ShareLink(item: debugReport) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(LASHeaderIconButtonStyle(dark: false))
                .accessibilityLabel("Share diagnostics")
            }

            ScrollView {
                VStack(alignment: .leading, spacing: LASDesignTokens.component) {
                    if let crash = CrashReporter.shared.lastCrash {
                        LASDebuggerCrashBreadcrumbCard(crash: crash)
                    }

                    LASDebuggerLoadCard(
                        snapshot: modelService.loadDebug,
                        serviceState: modelService.state,
                        fallbackName: modelService.activeDisplayName,
                        now: now,
                        onStop: { modelService.unload() },
                        onRetry: {
                            // A previous Stop/timeout may still be draining the
                            // native loader. Wait for that boundary before
                            // starting a new multi-GB load so Retry cannot create
                            // an overlapping MLX allocation.
                            Task { @MainActor in
                                await modelService.unloadAndWaitForCleanup()
                                modelService.startLoad()
                            }
                        }
                    )

                    LASDebuggerRuntimeCard(
                        snapshot: systemStatus.snapshot,
                        thermalState: safetyMonitor.effectiveThermalState,
                        memoryWarningCount: safetyMonitor.memoryWarningCount,
                        processCeiling: MemoryAdvisor.processMemoryCeiling
                    )

                    LASDebuggerEventsCard(
                        events: mergedEvents,
                        report: debugReport,
                        onCopy: {
                            UIPasteboard.general.string = debugReport
                        },
                        onClear: clearLogs
                    )
                }
                .padding(.horizontal, LASDesignTokens.pageInset)
                .padding(.bottom, 32)
            }
        }
        .background(Color(uiColor: .systemGroupedBackground))
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            systemStatus.startObserving()
            refreshDiagnostics()
        }
        .onDisappear {
            systemStatus.stopObserving()
        }
        .onReceive(refreshTimer) { tick in
            now = tick
            refreshDiagnostics()
        }
    }

    private var mergedEvents: [LASDebuggerEvent] {
        let runtime = logCenter.entries.map { entry in
            LASDebuggerEvent(
                id: entry.id,
                timestamp: entry.timestamp,
                level: entry.level.rawValue,
                subsystem: entry.subsystem,
                message: entry.message
            )
        }
        let diagnostics = diagnosticEntries.map { entry in
            LASDebuggerEvent(
                id: entry.id,
                timestamp: entry.timestamp,
                level: entry.level.label,
                subsystem: entry.category,
                message: entry.message
            )
        }
        return Array(
            (runtime + diagnostics)
                .sorted { $0.timestamp > $1.timestamp }
                .prefix(120)
        )
    }

    private func refreshDiagnostics() {
        diagnosticEntries = Diagnostics.shared.recentEntries(minLevel: .debug)
    }

    private var debugReport: String {
        let load = modelService.loadDebug
        let loadHeader = """
        === ACTIVE LOAD SNAPSHOT ===
        model: \(load.displayName.isEmpty ? modelService.activeDisplayName : load.displayName)
        repo: \(load.repoID.isEmpty ? "unknown" : load.repoID)
        phase: \(load.phase.rawValue)
        operation: \(load.operation)
        progress: \(load.progress.map { "\(Int($0 * 100))%" } ?? "none")
        detail: \(load.detail)
        process footprint: \(MemoryAdvisor.physFootprint.formattedBytes)
        available for model: \(MemoryAdvisor.availableMemoryForModel.formattedBytes)
        process ceiling: \(MemoryAdvisor.processMemoryCeiling.formattedBytes)
        high-memory entitlement: \(MemoryAdvisor.hasIncreasedMemoryLimitEntitlement)
        """
        let live = mergedEvents.reversed().map { event in
            let timestamp = event.timestamp.formatted(
                .dateTime
                    .year()
                    .month(.twoDigits)
                    .day(.twoDigits)
                    .hour(.twoDigits(amPM: .omitted))
                    .minute(.twoDigits)
                    .second(.twoDigits)
            )
            return "\(timestamp) \(event.level) \(event.subsystem.uppercased()) \(event.message)"
        }.joined(separator: "\n")
        return "\(loadHeader)\n\n\(Diagnostics.shared.exportText())\n\n=== LIVE EVENTS ===\n\(live)"
    }

    private func clearLogs() {
        logCenter.clear()
        Diagnostics.shared.clear()
        refreshDiagnostics()
    }
}

private struct LASDebuggerCrashBreadcrumbCard: View {
    let crash: CrashReporter.CrashInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "PREVIOUS SESSION")

            Label(crash.kind.uppercased(), systemImage: "exclamationmark.triangle")
                .font(.subheadline.monospaced().weight(.semibold))
            Text(crash.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if !crash.trail.isEmpty {
                Divider()
                Text("LAST SAFE BREADCRUMBS")
                    .font(.caption2.monospaced().weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(Array(crash.trail.suffix(8).enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .lasCard()
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Previous crash breadcrumbs")
    }
}

private struct LASDebuggerLoadCard: View {
    let snapshot: CodingAssistantService.LoadDebugSnapshot
    let serviceState: CodingAssistantService.ServiceState
    let fallbackName: String
    let now: Date
    let onStop: () -> Void
    let onRetry: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            LASSectionLabel(title: "MODEL LOAD")

            HStack(alignment: .top, spacing: 12) {
                Image(systemName: snapshot.phase == .ready
                      ? "checkmark.circle.fill"
                      : snapshot.isStalled
                        ? "exclamationmark.triangle.fill"
                        : "circle.dotted")
                    .font(.system(size: 28))
                    .foregroundStyle(statusColor)

                VStack(alignment: .leading, spacing: 4) {
                    Text(snapshot.displayName.isEmpty ? fallbackName : snapshot.displayName)
                        .font(.headline)
                    Text(snapshot.phase.rawValue)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(statusColor)
                    if !snapshot.repoID.isEmpty {
                        Text(snapshot.repoID)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }

                Spacer(minLength: 4)
                Text(elapsedText)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(snapshot.operation)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    if let progress = snapshot.progress {
                        Text("\(Int(progress * 100))%")
                            .font(.caption.monospaced().weight(.bold))
                    }
                }

                if let progress = snapshot.progress {
                    ProgressView(value: progress)
                        .tint(statusColor)
                } else if isLoading {
                    ProgressView()
                        .tint(statusColor)
                }

                Text(snapshot.detail.isEmpty ? stateDescription : snapshot.detail)
                    .font(.caption)
                    .foregroundStyle(snapshot.isStalled ? .orange : .secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if let lastProgressAt = snapshot.lastProgressAt, isLoading {
                    Text("Last loader callback \(timeSince(lastProgressAt)) ago")
                        .font(.caption2.monospaced())
                        .foregroundStyle(snapshot.isStalled ? .orange : .secondary)
                }
            }

            if snapshot.isStalled {
                Label(
                    "The native loader has not reported progress. This can be model parsing, a stalled Hub download, memory pressure, or a runtime deadlock.",
                    systemImage: "info.circle"
                )
                .font(.caption)
                .foregroundStyle(.orange)
            }

            HStack(spacing: 10) {
                Button("Stop loading", systemImage: "xmark.circle", action: onStop)
                    .buttonStyle(LASSecondaryButtonStyle())
                    .disabled(!isLoading)
                Button("Retry", systemImage: "arrow.clockwise", action: onRetry)
                    .buttonStyle(LASPrimaryButtonStyle())
                    .disabled(isLoading)
            }
        }
        .lasCard()
        .overlay {
            RoundedRectangle(cornerRadius: LASDesignTokens.cardRadius)
                .stroke(statusColor.opacity(0.22), lineWidth: 1)
        }
    }

    private var isLoading: Bool {
        if case .loading = serviceState { return true }
        return false
    }

    private var statusColor: Color {
        switch snapshot.phase {
        case .ready: return .green
        case .failed: return .red
        case .cancelled: return .orange
        case .downloading: return .blue
        case .preparing, .loading: return .purple
        case .idle: return .secondary
        }
    }

    private var stateDescription: String {
        switch serviceState {
        case .unloaded: return "No model is resident."
        case .loading(let message): return message
        case .ready: return "The model is resident and ready for API requests."
        case .generating: return "The model is serving an API request."
        case .failed(let message): return message
        }
    }

    private var elapsedText: String {
        guard let startedAt = snapshot.startedAt else { return "—" }
        let end = snapshot.phase == .ready
            || snapshot.phase == .failed
            || snapshot.phase == .cancelled
            ? snapshot.lastEventAt
            : now
        return elapsed(end.timeIntervalSince(startedAt))
    }

    private func timeSince(_ date: Date) -> String {
        elapsed(max(0, now.timeIntervalSince(date)))
    }

    private func elapsed(_ seconds: TimeInterval) -> String {
        let total = Int(seconds)
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

private struct LASDebuggerRuntimeCard: View {
    let snapshot: SystemStatusService.Snapshot
    let thermalState: ProcessInfo.ThermalState
    let memoryWarningCount: Int
    let processCeiling: Int64

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "ADMISSION SNAPSHOT")

            LASDebuggerMetricRow(
                label: "System-reported available memory",
                value: MemoryAdvisor.availableRAM.formattedBytes
            )
            LASDebuggerMetricRow(
                label: "Estimated process budget remaining",
                value: snapshot.availableForML.formattedBytes
            )
            LASDebuggerMetricRow(label: "Current app footprint", value: snapshot.usedByApp.formattedBytes)
            LASDebuggerMetricRow(label: "Process ceiling", value: processCeiling.formattedBytes)
            LASDebuggerMetricRow(
                label: "Kernel headroom",
                value: snapshot.freeRightNow > 0
                    ? snapshot.freeRightNow.formattedBytes
                    : "Unavailable"
            )
            LASDebuggerMetricRow(
                label: "High-memory entitlement",
                value: snapshot.hasIncreasedMemoryLimitEntitlement ? "active" : "missing"
            )
            LASDebuggerMetricRow(
                label: "Low Power Mode",
                value: snapshot.lowPowerModeEnabled ? "on" : "off"
            )
            LASDebuggerMetricRow(label: "Free storage", value: snapshot.diskFree.formattedBytes)
            LASDebuggerMetricRow(label: "Thermal state", value: thermalLabel)
            LASDebuggerMetricRow(label: "Memory warnings", value: "\(memoryWarningCount)")
            LASDebuggerMetricRow(label: "Device / OS", value: "\(snapshot.device) · \(snapshot.os)")

            LASDebuggerEntitlementNotice(
                isEntitled: snapshot.hasIncreasedMemoryLimitEntitlement
            )
        }
        .lasCard()
    }

    private var thermalLabel: String {
        switch thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

private struct LASDebuggerEntitlementNotice: View {
    let isEntitled: Bool

    var body: some View {
        if !isEntitled {
            Label(
                "The installed signature does not grant increased memory. Large MLX loads are blocked to prevent iOS from terminating the app.",
                systemImage: "signature"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        }
    }
}

private struct LASDebuggerEventsCard: View {
    let events: [LASDebuggerEvent]
    let report: String
    let onCopy: () -> Void
    let onClear: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            LASSectionLabel(title: "LIVE EVENTS", foreground: .white.opacity(0.48))
            ViewThatFits {
                HStack(spacing: LASDesignTokens.row) {
                    eventActions
                }
                VStack(alignment: .leading, spacing: LASDesignTokens.tight) {
                    eventActions
                }
            }

            if events.isEmpty {
                Text("Waiting for model, memory, and server events…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(events) { event in
                        LASDebuggerEventRow(event: event)
                    }
                }
            }
        }
        .padding(LASDesignTokens.cardPadding)
        .background(LASDesignTokens.terminalBackground,
                    in: RoundedRectangle(cornerRadius: LASDesignTokens.cardRadius))
        .overlay {
            RoundedRectangle(cornerRadius: LASDesignTokens.cardRadius)
                .stroke(.white.opacity(0.14), lineWidth: 1)
        }
    }

    @ViewBuilder
    private var eventActions: some View {
        Button(action: onCopy) {
            Label("Copy", systemImage: "doc.on.doc")
                .frame(minWidth: 64, minHeight: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .font(.caption.weight(.semibold))
        .foregroundStyle(.cyan)
        .disabled(events.isEmpty)

        ShareLink(item: report) {
            Label("Share", systemImage: "square.and.arrow.up")
                .frame(minWidth: 64, minHeight: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .font(.caption.weight(.semibold))
        .foregroundStyle(.cyan)
        .disabled(events.isEmpty)

        Button(action: onClear) {
            Label("Clear", systemImage: "trash")
                .frame(minWidth: 64, minHeight: 44)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .font(.caption.weight(.semibold))
        .foregroundStyle(.cyan)
        .disabled(events.isEmpty)
    }
}

private struct LASDebuggerEventRow: View {
    let event: LASDebuggerEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 7) {
                Text(event.timestamp, format: .dateTime.hour().minute().second())
                    .foregroundStyle(.white.opacity(0.42))
                Text(event.level)
                    .foregroundStyle(levelColor)
                Text(event.subsystem.uppercased())
                    .foregroundStyle(.white.opacity(0.5))
            }
            Text(verbatim: event.message)
                .foregroundStyle(.white.opacity(0.92))
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.system(.caption, design: .monospaced))
        .textSelection(.enabled)
    }

    private var levelColor: Color {
        switch event.level {
        case "ERROR", "FAULT": return .red
        case "WARN", "WARNING": return .yellow
        case "NOTICE": return .teal
        default: return .cyan
        }
    }
}

private struct LASDebuggerMetricRow: View {
    let label: String
    let value: String

    var body: some View {
        ViewThatFits {
            HStack(spacing: LASDesignTokens.row) {
                labelText
                Spacer()
                valueText
            }
            VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
                labelText
                valueText
            }
        }
        .frame(minHeight: 44)
    }

    private var labelText: some View {
        Text(label)
            .font(.subheadline)
            .foregroundStyle(.secondary)
    }

    private var valueText: some View {
        Text(verbatim: value)
            .font(.subheadline.monospaced().weight(.semibold))
            .multilineTextAlignment(.trailing)
            .textSelection(.enabled)
    }
}

private struct LASDebuggerEvent: Identifiable {
    let id: UUID
    let timestamp: Date
    let level: String
    let subsystem: String
    let message: String
}
