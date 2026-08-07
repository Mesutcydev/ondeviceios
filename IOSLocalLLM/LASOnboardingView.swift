import SwiftUI

/// First-launch orientation adapted from the reference CodeLens editorial
/// onboarding. The content is intentionally scoped to this server-only build.
struct LASOnboardingView: View {
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.koduTheme) private var T
    @State private var page = 0

    private let pages = LASOnboardingPage.content

    private var isLastPage: Bool { page == pages.count - 1 }
    private var appVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }

    var body: some View {
        ZStack {
            LASPageBackground()

            VStack(spacing: 0) {
                header

                TabView(selection: $page) {
                    ForEach(pages) { item in
                        LASOnboardingStoryPage(item: item)
                            .tag(item.id)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))

                footer
            }
        }
    }

    private var header: some View {
        HStack(spacing: LASDesignTokens.row) {
            Image("app_logo_small")
                .resizable()
                .scaledToFit()
                .frame(width: 38, height: 38)
                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

            VStack(alignment: .leading, spacing: 2) {
                Text("On Device : LAS")
                    .font(.headline.weight(.semibold))
                Text("LOCAL API SERVER")
                    .font(.caption2.weight(.bold).monospaced())
                    .tracking(0.8)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(verbatim: "v\(appVersion)")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)

            if !isLastPage {
                Button("Skip") {
                    complete()
                }
                .font(.subheadline.weight(.semibold))
                .frame(minWidth: 52, minHeight: 44)
                .contentShape(Rectangle())
            }
        }
        .padding(.horizontal, LASDesignTokens.pageInset)
        .padding(.top, LASDesignTokens.row)
    }

    private var footer: some View {
        VStack(spacing: LASDesignTokens.row) {
            HStack(spacing: 6) {
                ForEach(pages) { item in
                    Capsule()
                        .fill(item.id == page ? Color.primary : Color.primary.opacity(0.16))
                        .frame(width: item.id == page ? 24 : 6, height: 6)
                        .animation(.spring(response: 0.32, dampingFraction: 0.82), value: page)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Page \(page + 1) of \(pages.count)")

            Button {
                if isLastPage {
                    complete()
                } else {
                    withAnimation(.easeInOut(duration: 0.22)) {
                        page += 1
                    }
                }
            } label: {
                HStack {
                    Text(isLastPage ? "Enter LAS" : "Next")
                    Spacer()
                    Image(systemName: isLastPage ? "arrow.right.circle.fill" : "arrow.right")
                }
                .font(.headline.weight(.semibold))
                .foregroundStyle(.white)
                .padding(.horizontal, LASDesignTokens.component)
                .frame(maxWidth: .infinity, minHeight: 52)
                .background(T.accentStrong, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .buttonStyle(.plain)

            Label("Models and requests stay under your control.", systemImage: "lock.shield")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, LASDesignTokens.pageInset)
        .padding(.bottom, LASDesignTokens.pageInset)
    }

    private func complete() {
        withAnimation(.easeOut(duration: 0.2)) {
            settings.hasSeenOnboarding = true
        }
    }
}

private struct LASOnboardingStoryPage: View {
    let item: LASOnboardingPage
    @Environment(\.koduTheme) private var T

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: LASDesignTokens.section) {
                hero
                capabilityMatrix
            }
            .padding(.horizontal, LASDesignTokens.pageInset)
            .padding(.top, LASDesignTokens.section)
            .padding(.bottom, LASDesignTokens.component)
        }
        .scrollIndicators(.hidden)
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: LASDesignTokens.row) {
            Text(item.caption)
                .font(.caption.weight(.bold).monospaced())
                .tracking(1.0)
                .foregroundStyle(.secondary)

            HStack(alignment: .center, spacing: LASDesignTokens.component) {
                ZStack {
                    RoundedRectangle(cornerRadius: 24, style: .continuous)
                        .fill(T.accentStrong)
                    if item.usesLogo {
                        Image("app_logo_small")
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
                    } else {
                        Image(systemName: item.symbol)
                            .font(.system(size: 38, weight: .medium))
                            .foregroundStyle(.white)
                    }
                }
                .frame(width: 92, height: 92)
                .accessibilityHidden(true)

                Text(item.title)
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .tracking(-0.8)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text(item.body)
                .font(.body)
                .foregroundStyle(.secondary)
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var capabilityMatrix: some View {
        VStack(spacing: 0) {
            ForEach(item.rows) { row in
                HStack(alignment: .firstTextBaseline, spacing: LASDesignTokens.row) {
                    Text(row.label)
                        .font(.caption.weight(.semibold).monospaced())
                        .foregroundStyle(.secondary)
                        .frame(width: 92, alignment: .leading)
                    Text(row.value)
                        .font(.caption.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.horizontal, LASDesignTokens.component)
                .padding(.vertical, LASDesignTokens.row)

                if row.id != item.rows.last?.id {
                    Divider()
                        .padding(.leading, LASDesignTokens.component)
                }
            }
        }
        .lasCard(radius: LASDesignTokens.tileRadius, padding: 0)
    }
}

private struct LASOnboardingPage: Identifiable {
    struct Row: Identifiable {
        let id: Int
        let label: String
        let value: String
    }

    let id: Int
    let caption: String
    let title: String
    let body: String
    let symbol: String
    let usesLogo: Bool
    let rows: [Row]

    static let content: [LASOnboardingPage] = [
        LASOnboardingPage(
            id: 0,
            caption: "LOCAL-FIRST INFERENCE",
            title: "Your model.\nYour endpoint.",
            body: "Run a language model on your iPhone and expose it to trusted devices on your local network—without silently falling back to a cloud model.",
            symbol: "eye",
            usesLogo: true,
            rows: [
                Row(id: 0, label: "runtime", value: "MLX + llama.cpp · on device"),
                Row(id: 1, label: "network", value: "trusted LAN · foreground only"),
                Row(id: 2, label: "fallback", value: "no remote inference"),
                Row(id: 3, label: "control", value: "load · unload · cancel")
            ]
        ),
        LASOnboardingPage(
            id: 1,
            caption: "MODEL LIBRARY",
            title: "Bring the\nmodel you need.",
            body: "Download from the curated catalog, resume interrupted transfers, add a Hugging Face token, or import a compatible model from Files.",
            symbol: "shippingbox",
            usesLogo: false,
            rows: [
                Row(id: 0, label: "download", value: "pause · resume · cancel"),
                Row(id: 1, label: "import", value: "model folders and files"),
                Row(id: 2, label: "export", value: "visible in the Files app"),
                Row(id: 3, label: "storage", value: "app Documents sandbox")
            ]
        ),
        LASOnboardingPage(
            id: 2,
            caption: "CLIENT COMPATIBILITY",
            title: "One server.\nFamiliar APIs.",
            body: "Connect agent tools and local clients through OpenAI, Anthropic, and Ollama-compatible routes. Share setup commands directly from LAS.",
            symbol: "point.3.connected.trianglepath.dotted",
            usesLogo: false,
            rows: [
                Row(id: 0, label: "OpenAI", value: "/v1/chat/completions · /v1/responses"),
                Row(id: 1, label: "Anthropic", value: "/v1/messages"),
                Row(id: 2, label: "Ollama", value: "/api/chat · /api/generate"),
                Row(id: 3, label: "agents", value: "tools · reasoning · streaming")
            ]
        ),
        LASOnboardingPage(
            id: 3,
            caption: "VISIBLE OPERATIONS",
            title: "See what the\nruntime is doing.",
            body: "The live parser, device dashboard, download progress, verbose terminal, and on-device debugger keep model and server activity understandable.",
            symbol: "waveform.path.ecg.rectangle",
            usesLogo: false,
            rows: [
                Row(id: 0, label: "health", value: "memory · thermal · storage"),
                Row(id: 1, label: "parser", value: "connecting · ready · generating"),
                Row(id: 2, label: "terminal", value: "safe live runtime events"),
                Row(id: 3, label: "debugger", value: "loader progress and recovery")
            ]
        ),
        LASOnboardingPage(
            id: 4,
            caption: "READY TO SERVE",
            title: "Four steps\nto local AI.",
            body: "Open Models, download or import a compatible model, load it from Home, then share the connection setup with your client.",
            symbol: "checkmark.seal",
            usesLogo: false,
            rows: [
                Row(id: 0, label: "01", value: "Open Models"),
                Row(id: 1, label: "02", value: "Download or import"),
                Row(id: 2, label: "03", value: "Load the model"),
                Row(id: 3, label: "04", value: "Share setup commands")
            ]
        )
    ]
}
