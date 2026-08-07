import SwiftUI

/// A compact live terminal for diagnosing server, model, and download work
/// without exposing credentials, prompts, or generated content.
struct LASTerminalView: View {
    @ObservedObject private var logCenter = RuntimeLogCenter.shared
    @State private var followsOutput = true

    var body: some View {
        VStack(spacing: 0) {
            LASDetailHeader("Verbose terminal", dark: true) {
                Button {
                    logCenter.clear()
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                }
                .buttonStyle(LASHeaderIconButtonStyle(dark: true))
                .disabled(logCenter.entries.isEmpty)
                .accessibilityLabel("Clear terminal output")
            }

            terminalControlBar
            Divider().overlay(Color.white.opacity(0.12))

            if logCenter.entries.isEmpty {
                ContentUnavailableView(
                    "Waiting for events",
                    systemImage: "terminal",
                    description: Text("Server and model activity will appear here live.")
                )
                .foregroundStyle(.white.opacity(0.8))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(terminalBackground)
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(logCenter.entries) { entry in
                                LASTerminalLine(entry: entry)
                                    .id(entry.id)
                            }
                        }
                        .padding(.horizontal, LASDesignTokens.component)
                        .padding(.vertical, LASDesignTokens.row)
                    }
                    .background(terminalBackground)
                    .scrollIndicators(.hidden)
                    .onChange(of: logCenter.entries.count) { _, _ in
                        guard followsOutput, let last = logCenter.entries.last else { return }
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                    .task {
                        guard followsOutput, let last = logCenter.entries.last else { return }
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .background(LASDesignTokens.terminalBackground.ignoresSafeArea())
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            if logCenter.entries.isEmpty {
                logCenter.append("Terminal attached", subsystem: "terminal")
            }
        }
    }

    private var terminalControlBar: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(.green)
                .frame(width: 8, height: 8)
            Text("LIVE")
                .font(.caption.weight(.bold).monospaced())
                .foregroundStyle(.green)
            Text("\(logCenter.entries.count) events")
                .font(.caption.monospaced())
                .foregroundStyle(.white.opacity(0.55))
            Spacer()
            Button(followsOutput ? "Following" : "Follow") {
                followsOutput.toggle()
            }
            .font(.caption.weight(.semibold))
            .foregroundStyle(.white.opacity(0.88))
            .padding(.horizontal, 12)
            .frame(minHeight: 44)
            .background(Color.white.opacity(0.11), in: Capsule())
            .contentShape(Capsule())
            .accessibilityValue(followsOutput ? "On" : "Off")
        }
        .padding(.horizontal, LASDesignTokens.pageInset)
        .frame(minHeight: 48)
        .background(LASDesignTokens.terminalBackground)
    }

    private var terminalBackground: Color {
        LASDesignTokens.terminalBackground
    }
}

private struct LASTerminalLine: View {
    let entry: RuntimeLogCenter.Entry

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 7) {
                Text(timestamp(entry.timestamp))
                    .foregroundStyle(.white.opacity(0.42))
                Text(entry.level.symbol)
                    .foregroundStyle(color(for: entry.level))
                Text(entry.subsystem.uppercased())
                    .foregroundStyle(.white.opacity(0.55))
            }
            Text(entry.message)
                .foregroundStyle(.white.opacity(0.92))
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.system(.caption, design: .monospaced))
        .textSelection(.enabled)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func timestamp(_ date: Date) -> String {
        date.formatted(
            .dateTime
                .hour(.twoDigits(amPM: .omitted))
                .minute(.twoDigits)
                .second(.twoDigits)
        )
    }

    private func color(for level: RuntimeLogCenter.Level) -> Color {
        switch level {
        case .debug: return .white.opacity(0.5)
        case .info: return .cyan
        case .warning: return .yellow
        case .error: return .red
        }
    }

}
