import SwiftUI

enum LASDesignTokens {
    static let micro: CGFloat = 4
    static let tight: CGFloat = 8
    static let row: CGFloat = 12
    static let component: CGFloat = 16
    static let pageInset: CGFloat = 20
    static let cardPadding: CGFloat = 20
    static let section: CGFloat = 24

    static let majorRadius: CGFloat = 28
    static let cardRadius: CGFloat = 24
    static let tileRadius: CGFloat = 20
    static let controlRadius: CGFloat = 15

    static let hairline = Color.primary.opacity(0.09)
    static let pageTop = Color(red: 0.94, green: 0.95, blue: 0.97)
    static let pageBottom = Color(red: 0.98, green: 0.99, blue: 1.0)
    static let terminalBackground = Color(red: 0.055, green: 0.065, blue: 0.085)
}

/// Soft ambient sample field so clear Liquid Glass has live pixels to refract.
struct LASPageBackground: View {
    var body: some View {
        LiquidPinkBackdrop()
    }
}

struct LASCardModifier: ViewModifier {
    let radius: CGFloat
    let padding: CGFloat

    @Environment(\.koduTheme) private var T

    func body(content: Content) -> some View {
        content
            .padding(padding)
            // CodeLens-parity clear glass: iOS 26+ Liquid Glass, otherwise
            // ultra-thin material so the ambient backdrop still reads through.
            .glassSurface(.card, cornerRadius: radius)
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(T.rule.opacity(0.85), lineWidth: 1)
            }
    }
}

extension View {
    func lasCard(
        radius: CGFloat = LASDesignTokens.cardRadius,
        padding: CGFloat = LASDesignTokens.cardPadding
    ) -> some View {
        modifier(LASCardModifier(radius: radius, padding: padding))
    }
}

struct LASSectionLabel: View {
    let title: LocalizedStringKey
    var trailing: String?
    var trailingColor: Color = .secondary
    var foreground: Color = .secondary

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(title)
                .font(.caption.weight(.bold).monospaced())
                .tracking(0.6)
                .foregroundStyle(foreground)
            Spacer(minLength: LASDesignTokens.tight)
            if let trailing {
                Text(verbatim: trailing)
                    .font(.caption2.weight(.semibold).monospaced())
                    .foregroundStyle(trailingColor)
            }
        }
    }
}

struct LASDetailHeader<Trailing: View>: View {
    let title: LocalizedStringKey
    let dark: Bool
    private let trailing: Trailing

    @Environment(\.dismiss) private var dismiss

    init(
        _ title: LocalizedStringKey,
        dark: Bool = false,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.title = title
        self.dark = dark
        self.trailing = trailing()
    }

    var body: some View {
        ZStack {
            Text(title)
                .font(.title3.weight(.semibold))
                .foregroundStyle(dark ? Color.white : Color.primary)
                .lineLimit(1)

            HStack {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .contentShape(Circle())
                }
                .buttonStyle(LASHeaderIconButtonStyle(dark: dark))
                .accessibilityLabel("Back")

                Spacer()

                trailing
                    .frame(width: 44, height: 44)
            }
        }
        .frame(minHeight: 56)
        .padding(.horizontal, LASDesignTokens.pageInset)
    }
}

struct LASHeaderIconButtonStyle: ButtonStyle {
    let dark: Bool

    func makeBody(configuration: Configuration) -> some View {
        Group {
            if dark {
                configuration.label
                    .foregroundStyle(Color.white)
                    .background(
                        Color.white.opacity(configuration.isPressed ? 0.20 : 0.13),
                        in: Circle()
                    )
                    .overlay {
                        Circle().stroke(Color.white.opacity(0.18), lineWidth: 1)
                    }
            } else {
                configuration.label
                    .foregroundStyle(Color.primary)
                    .glassSurface(.toolbarButton, cornerRadius: 22)
                    .overlay {
                        Circle().stroke(LASDesignTokens.hairline, lineWidth: 1)
                    }
            }
        }
        .scaleEffect(configuration.isPressed ? 0.96 : 1)
        .animation(.easeOut(duration: 0.16), value: configuration.isPressed)
    }
}

struct LASEmptyHeaderSlot: View {
    var body: some View {
        Color.clear
            .frame(width: 44, height: 44)
            .accessibilityHidden(true)
    }
}

struct LASSearchField: View {
    @Binding var text: String
    let prompt: LocalizedStringKey

    var body: some View {
        HStack(spacing: LASDesignTokens.tight) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(.secondary)

            TextField(prompt, text: $text)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            if !text.isEmpty {
                Button {
                    text = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                        .frame(width: 32, height: 44)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear search")
            }
        }
        .padding(.horizontal, LASDesignTokens.component)
        .frame(minHeight: 52)
        .glassSurface(.card, cornerRadius: 20)
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(LASDesignTokens.hairline, lineWidth: 1)
        }
    }
}

struct LASPrimaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, LASDesignTokens.component)
            .frame(minHeight: 44)
            .background(
                Color.accentColor.opacity(configuration.isPressed ? 0.78 : 1),
                in: RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
    }
}

struct LASSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.tint)
            .padding(.horizontal, LASDesignTokens.component)
            .frame(minHeight: 44)
            .background(
                Color.secondary.opacity(configuration.isPressed ? 0.17 : 0.10),
                in: RoundedRectangle(cornerRadius: LASDesignTokens.controlRadius, style: .continuous)
            )
    }
}

struct LASMetricTile: View {
    let title: String
    let value: String
    let detail: String
    let symbol: String

    var body: some View {
        HStack(spacing: LASDesignTokens.row) {
            Image(systemName: symbol)
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.secondary)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: LASDesignTokens.micro) {
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(verbatim: value)
                    .font(.subheadline.weight(.semibold).monospaced())
                    .lineLimit(1)
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
        }
        .padding(LASDesignTokens.row)
        .frame(minHeight: 88)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: LASDesignTokens.tileRadius))
    }
}
