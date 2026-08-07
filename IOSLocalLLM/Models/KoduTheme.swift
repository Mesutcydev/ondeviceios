import SwiftUI
import UIKit   // UIFont probe in the font-name fallback path

// MARK: - KoduTheme
// Studio design language for IOSLocalLLM — Apple clear glass, neutral content,
// and color reserved for state or a primary action. Surface colors are
// intentionally translucent: native iOS 26 chrome gets real `.clear` glass,
// while legacy/custom fills still reveal the shared ambient backdrop.
//
// Source: Direction B from the Kodu design handoff.
//   Geist + Geist Mono · #fafaf9 background · #1d4ed8 accent · table density.
//
// Usage:
//   @Environment(\.koduTheme) var T
//   Text("hello").font(T.mono(11)).foregroundColor(T.ink2)

// MARK: - Palette

struct KoduTheme {

    // Background layers
    let bg: Color           // page background
    let surface: Color      // cards
    let surface2: Color     // secondary surface (toolbars, code blocks)
    let surface3: Color     // tertiary surface

    // Text ink (4 levels of hierarchy)
    let ink: Color          // primary
    let ink2: Color         // secondary
    let ink3: Color         // tertiary / labels
    let ink4: Color         // disabled / very subtle

    // Borders
    let rule: Color         // hairline 8% black
    let rule2: Color        // stronger 16% black

    // Brand
    let accent: Color       // #1d4ed8
    let accentSoft: Color   // rgba(29,78,216,0.10)
    let accentSofter: Color // rgba(29,78,216,0.06)

    // Semantic
    let good: Color
    let warn: Color
    let bad: Color

    // Spacing (single scale to match the prototype's pad/gap idea)
    let pad: CGFloat
    let gap: CGFloat

    let isDark: Bool
    /// True for the pure-black OLED appearance. Defaulted so existing
    /// initializers/call sites are unaffected. Decorative surfaces (the
    /// LiquidPinkBackdrop blooms/orbs) suppress themselves when this is set so
    /// the page stays true #000000.
    var isOLED: Bool = false

    // Brand-accent anchors. Stored (not computed) so `KoduTheme.make(appearance:accent:)`
    // can swap the whole brand-accent family per selected palette while leaving
    // backgrounds, ink, the sage `accent2`, and semantic good/warn/bad intact.
    // Defaults below reproduce the original Liquid Pink rose exactly.
    let roseHi: Color        // top of the primary-button gradient
    let roseDeep: Color      // inline code / deep edge of user bubble
    let accentStrong: Color  // primary CTA / active tab — deeper than `accent`

    // MARK: - Light — neutral utility
    static let light = KoduTheme(
        bg:        Color(red: 0.968, green: 0.976, blue: 0.992),  // airy pearl canvas
        surface:   Color.white.opacity(0.26),
        surface2:  Color.white.opacity(0.18),
        surface3:  Color.white.opacity(0.10),
        ink:       Color(red: 0.110, green: 0.110, blue: 0.118),  // #1C1C1E label
        ink2:      Color(red: 0.427, green: 0.427, blue: 0.447),  // #6D6D72 secondary
        ink3:      Color(red: 0.557, green: 0.557, blue: 0.576),  // #8E8E93 tertiary
        ink4:      Color(red: 0.780, green: 0.780, blue: 0.800),  // #C7C7CC quaternary
        rule:      Color.black.opacity(0.10),
        rule2:     Color.black.opacity(0.18),
        accent:    Color(red: 0.000, green: 0.478, blue: 1.000),  // system blue fallback
        accentSoft:   Color(red: 0.000, green: 0.478, blue: 1.000).opacity(0.10),
        accentSofter: Color(red: 0.000, green: 0.478, blue: 1.000).opacity(0.06),
        good:      Color(red: 0.133, green: 0.545, blue: 0.302),
        warn:      Color(red: 0.690, green: 0.424, blue: 0.047),
        bad:       Color(red: 0.784, green: 0.118, blue: 0.196),
        pad: 16, gap: 10,
        isDark: false,
        roseHi:       Color(red: 0.200, green: 0.560, blue: 1.000),
        roseDeep:     Color(red: 0.000, green: 0.330, blue: 0.780),
        accentStrong: Color(red: 0.000, green: 0.400, blue: 0.900)
    )

    // MARK: - Dark — neutral utility
    static let dark = KoduTheme(
        // Lift the canvas above black so clear glass reads as smoked crystal,
        // while keeping enough depth for bright, fully opaque foreground ink.
        bg:        Color(red: 0.140, green: 0.155, blue: 0.190),
        surface:   Color.white.opacity(0.13),
        surface2:  Color.white.opacity(0.18),
        surface3:  Color.white.opacity(0.08),
        ink:       Color(red: 0.929, green: 0.929, blue: 0.937),
        ink2:      Color(red: 0.706, green: 0.706, blue: 0.729),
        ink3:      Color(red: 0.510, green: 0.510, blue: 0.533),
        ink4:      Color(red: 0.337, green: 0.337, blue: 0.357),
        rule:      Color.white.opacity(0.15),
        rule2:     Color.white.opacity(0.25),
        accent:    Color(red: 0.220, green: 0.600, blue: 1.000),
        accentSoft:   Color(red: 0.220, green: 0.600, blue: 1.000).opacity(0.16),
        accentSofter: Color(red: 0.220, green: 0.600, blue: 1.000).opacity(0.09),
        good:      Color(red: 0.392, green: 0.784, blue: 0.533),
        warn:      Color(red: 0.898, green: 0.643, blue: 0.263),
        bad:       Color(red: 0.902, green: 0.404, blue: 0.431),
        pad: 16, gap: 10,
        isDark: true,
        roseHi:       Color(red: 0.360, green: 0.660, blue: 1.000),
        roseDeep:     Color(red: 0.040, green: 0.450, blue: 0.920),
        accentStrong: Color(red: 0.220, green: 0.600, blue: 1.000)
    )

    // MARK: - OLED Dark — Plum Dusk
    //
    // Pure #000000 page so OLED pixels switch off (deeper blacks, less battery),
    // with NEUTRAL grays for ink/surfaces instead of the warm mulberry of the
    // standard dark theme — the readable, high-contrast look of a modern code
    // editor. Surfaces are translucent whites over the black page, so cards/
    // pills read as subtly elevated panels (≈#121212 / #212121) rather than
    // pure black, keeping depth without lifting the page off true black.
    static let oled = KoduTheme(
        bg:        Color(red: 0.004, green: 0.004, blue: 0.008),  // #010102 OLED black
        surface:   Color.white.opacity(0.10),                     // clear elevated card
        surface2:  Color.white.opacity(0.15),                     // clear glass highlight
        surface3:  Color.white.opacity(0.06),                     // quiet glass layer
        ink:       Color(red: 0.929, green: 0.929, blue: 0.937),  // #EDEDEF off-white (not harsh pure white)
        ink2:      Color(red: 0.706, green: 0.706, blue: 0.729),  // #B4B4BA secondary
        ink3:      Color(red: 0.510, green: 0.510, blue: 0.533),  // #828288 tertiary
        ink4:      Color(red: 0.337, green: 0.337, blue: 0.357),  // #56565B quaternary
        rule:      Color.white.opacity(0.09),
        rule2:     Color.white.opacity(0.16),
        accent:    Color(red: 0.769, green: 0.169, blue: 0.522),  // #C42B85
        accentSoft:   Color(red: 0.643, green: 0.067, blue: 0.384).opacity(0.30),
        accentSofter: Color(red: 0.643, green: 0.067, blue: 0.384).opacity(0.16),
        good:      Color(red: 0.525, green: 1.000, blue: 0.753),  // #86FFC0
        warn:      Color(red: 0.988, green: 0.827, blue: 0.302),  // #FCD34D
        bad:       Color(red: 0.988, green: 0.647, blue: 0.647),  // #FCA5A5
        pad: 16, gap: 10,
        isDark: true,
        isOLED: true,
        roseHi:       Color(red: 0.635, green: 0.071, blue: 0.400), // #A21266
        roseDeep:     Color(red: 0.431, green: 0.043, blue: 0.275), // #6E0B46
        accentStrong: Color(red: 0.431, green: 0.043, blue: 0.275)
    )

    // MARK: - Accent palettes (theme combinations)
    //
    // Each palette swaps ONLY the brand-accent family (accent + soft/softer +
    // hi/deep/strong). Backgrounds, ink, the sage `accent2`, and good/warn/bad
    // are untouched, so layout and the semantic color language don't change.
    // `.rose` returns the original Liquid Pink theme byte-for-byte.
    enum KoduAccent: String, CaseIterable, Identifiable, Sendable {
        // `onyx` is the app's default — a near-black graphite monochrome (the
        // professional, ChatGPT-like look: near-black primary actions, color
        // reserved for status). Listed first so it leads the Settings swatch
        // row. `system` (vivid iOS blue) and the rest stay selectable.
        // rawValues are persisted, so the order may change freely.
        case onyx, system, rose, blue, violet, emerald, amber, graphite, teal, indigo, coral, magenta
        var id: String { rawValue }
        var displayName: String {
            switch self {
            case .onyx: return "Onyx"
            case .system: return "System"
            case .rose: return "Rose"
            case .blue: return "Blue"
            case .violet: return "Violet"
            case .emerald: return "Emerald"
            case .amber: return "Amber"
            case .graphite: return "Graphite"
            case .teal: return "Teal"
            case .indigo: return "Indigo"
            case .coral: return "Coral"
            case .magenta: return "Magenta"
            }
        }

        /// Representative accent color for the Settings swatch picker. `.rose`
        /// isn't in `anchors()` (make() returns the base theme for it), so it's
        /// resolved explicitly here.
        func swatchColor(dark: Bool) -> Color {
            if self == .rose {
                return dark ? Color(red: 1.0, green: 0.361, blue: 0.604)
                            : Color(red: 1.0, green: 0.180, blue: 0.494)
            }
            return anchors(dark: dark).accent
        }
        /// (accent, hi, deep, strong) anchors for this palette at the given appearance.
        fileprivate func anchors(dark: Bool) -> (accent: Color, hi: Color, deep: Color, strong: Color) {
            func c(_ r: Double, _ g: Double, _ b: Double) -> Color { Color(red: r, green: g, blue: b) }
            switch self {
            case .onyx:
                // Near-black graphite monochrome — the ChatGPT-like default.
                // Light: near-black accent/CTA (white text ~15:1). Dark: a
                // mid-graphite family kept dark enough that white button text
                // still passes everywhere (no light-on-light), while the accent
                // stays legible as ink on dark/OLED backgrounds.
                return dark
                    ? (c(0.46,0.46,0.51), c(0.55,0.55,0.60), c(0.36,0.36,0.41), c(0.42,0.42,0.47))
                    : (c(0.169,0.169,0.188), c(0.227,0.227,0.251), c(0.102,0.102,0.118), c(0.137,0.137,0.153))
            case .system:
                // iOS-native systemBlue — #007AFF (light) / a brighter blue for
                // dark. The clean, readable default: neutral page + white cards
                // + this accent reads like a stock iOS app. `hi` is kept no
                // lighter than the existing `.blue` palette's so white button
                // text holds the same contrast that already ships.
                return dark
                    ? (c(0.12,0.56,1.00), c(0.36,0.66,1.00), c(0.04,0.45,0.92), c(0.22,0.60,1.00))
                    : (c(0.00,0.478,1.00), c(0.20,0.56,1.00), c(0.00,0.33,0.78), c(0.00,0.40,0.90))
            case .rose:
                return dark
                    ? (c(1.000,0.361,0.604), c(1.000,0.561,0.694), c(1.000,0.180,0.494), c(1.000,0.302,0.553))
                    : (c(1.000,0.180,0.494), c(1.000,0.490,0.694), c(0.902,0.055,0.388), c(0.902,0.055,0.388))
            case .blue:
                return dark
                    ? (c(0.40,0.62,0.98), c(0.55,0.73,1.00), c(0.30,0.52,0.95), c(0.45,0.66,1.00))
                    : (c(0.23,0.51,0.92), c(0.36,0.62,0.97), c(0.13,0.36,0.75), c(0.18,0.44,0.84))
            case .violet:
                return dark
                    ? (c(0.66,0.52,0.95), c(0.74,0.62,1.00), c(0.56,0.42,0.92), c(0.70,0.56,1.00))
                    : (c(0.55,0.42,0.87), c(0.66,0.54,0.95), c(0.42,0.28,0.72), c(0.48,0.34,0.80))
            case .emerald:
                return dark
                    ? (c(0.30,0.80,0.58), c(0.42,0.88,0.66), c(0.22,0.70,0.50), c(0.34,0.82,0.60))
                    // Light `hi` darkened (was 0.24,0.72,0.53) so white hero
                    // text/eyebrows clear AA over the lightest gradient stop.
                    : (c(0.13,0.62,0.43), c(0.16,0.60,0.43), c(0.07,0.48,0.33), c(0.10,0.54,0.38))
            case .amber:
                return dark
                    ? (c(0.98,0.70,0.30), c(1.00,0.80,0.42), c(0.94,0.60,0.20), c(1.00,0.74,0.34))
                    // Light `hi`/`accent` darkened (was 0.86,0.53,0.13 / 0.95,
                    // 0.64,0.24) — the light orange failed AA under white text.
                    : (c(0.78,0.47,0.10), c(0.82,0.50,0.12), c(0.70,0.40,0.05), c(0.74,0.43,0.07))
            case .graphite:
                return dark
                    ? (c(0.62,0.66,0.72), c(0.74,0.78,0.84), c(0.50,0.54,0.60), c(0.70,0.74,0.80))
                    // Light `hi` darkened (was 0.48,0.52,0.58) for white-text AA.
                    : (c(0.36,0.40,0.46), c(0.40,0.44,0.50), c(0.24,0.28,0.34), c(0.30,0.34,0.40))
            case .teal:
                return dark
                    ? (c(0.20,0.78,0.74), c(0.34,0.86,0.82), c(0.12,0.68,0.64), c(0.26,0.80,0.76))
                    : (c(0.06,0.55,0.52), c(0.10,0.58,0.55), c(0.03,0.42,0.40), c(0.05,0.48,0.45))
            case .indigo:
                return dark
                    ? (c(0.48,0.52,0.96), c(0.60,0.63,1.00), c(0.40,0.44,0.92), c(0.52,0.56,1.00))
                    : (c(0.32,0.36,0.82), c(0.38,0.42,0.86), c(0.22,0.26,0.68), c(0.27,0.31,0.75))
            case .coral:
                return dark
                    ? (c(1.00,0.50,0.42), c(1.00,0.62,0.55), c(0.96,0.40,0.32), c(1.00,0.54,0.46))
                    : (c(0.88,0.30,0.22), c(0.90,0.34,0.26), c(0.74,0.20,0.14), c(0.80,0.24,0.17))
            case .magenta:
                return dark
                    ? (c(0.92,0.40,0.85), c(0.97,0.54,0.90), c(0.86,0.30,0.78), c(0.95,0.44,0.88))
                    : (c(0.78,0.16,0.66), c(0.82,0.22,0.70), c(0.64,0.08,0.54), c(0.72,0.12,0.60))
            }
        }
    }

    /// The only app-facing accent for now. Keeping the palette implementation
    /// below makes it easy to restore user-selectable colors later without
    /// allowing an old persisted choice to override the current black theme.
    static let appAccent: KoduAccent = .onyx

    /// Build a theme for an appearance + accent palette. `appearance` is the
    /// stored string — "light", "dark", or "oled". `.rose` is the untouched
    /// default; other palettes substitute only the brand-accent family.
    /// Applied at the root scene + tabs so the choice is app-wide.
    static func make(appearance: String, accent: KoduAccent) -> KoduTheme {
        let base: KoduTheme
        switch appearance {
        case "oled": base = .oled
        case "dark": base = .dark
        default:     base = .light
        }
        let dark = base.isDark
        // NOTE: every palette (including `.rose`) resolves through its own
        // anchors. The base light/dark/oled themes in this product are neutral
        // (blue/graphite), so the old `if accent == .rose { return base }`
        // shortcut made the Rose swatch paint blue. Rose now uses its pink
        // anchors to match the swatch shown in the picker.
        let a = accent.anchors(dark: dark)
        return KoduTheme(
            bg: base.bg, surface: base.surface, surface2: base.surface2, surface3: base.surface3,
            ink: base.ink, ink2: base.ink2, ink3: base.ink3, ink4: base.ink4,
            rule: base.rule, rule2: base.rule2,
            accent: a.accent,
            accentSoft: a.accent.opacity(dark ? 0.18 : 0.14),
            accentSofter: a.accent.opacity(dark ? 0.10 : 0.07),
            good: base.good, warn: base.warn, bad: base.bad,
            pad: base.pad, gap: base.gap,
            isDark: base.isDark,
            isOLED: base.isOLED,
            roseHi: a.hi, roseDeep: a.deep, accentStrong: a.strong
        )
    }

    // MARK: - Liquid Pink extras
    //
    // These extra colors aren't exposed via the original struct so existing
    // views keep their identical layout. New glass views read them via the
    // static accessors below.

    var accentStrongSoft: Color {
        accentStrong.opacity(isDark ? 0.18 : 0.12)
    }
    /// Secondary neutral accent used for quiet supporting glyphs.
    var accent2: Color {
        isDark
            ? Color(red: 0.620, green: 0.650, blue: 0.700)
            : Color(red: 0.430, green: 0.460, blue: 0.520)
    }
    /// Soft sage — fill for glyph tiles and pills that use `accent2`.
    var accent2Soft: Color {
        accent2.opacity(isDark ? 0.20 : 0.16)
    }

    // MARK: - Per-category model tints (Models hub)
    //
    // Each of the four model roles gets its own hue so cards are colour-coded
    // by type at a glance. Assistant uses the rose `accent`, Lens the sage
    // `accent2`; voice and image get the two below. Kept distinct from the
    // semantic `warn` amber so a "tight"/cleanup warning never reads as an
    // image card.

    /// Voice role — violet.
    var voiceTint: Color {
        isDark
            ? Color(red: 0.745, green: 0.624, blue: 1.000)   // #BE9FFF
            : Color(red: 0.553, green: 0.420, blue: 0.871)   // #8D6BDE
    }
    var voiceTintSoft: Color { voiceTint.opacity(isDark ? 0.20 : 0.14) }

    /// Image role — warm amber/orange (distinct from the cooler `warn`).
    var imageTint: Color {
        isDark
            ? Color(red: 1.000, green: 0.722, blue: 0.404)   // #FFB867
            : Color(red: 0.910, green: 0.561, blue: 0.180)   // #E88F2E
    }
    var imageTintSoft: Color { imageTint.opacity(isDark ? 0.20 : 0.14) }
    /// Subtle border used by legacy glass call sites.
    var glassBorder: Color {
        isDark ? Color.white.opacity(0.10) : Color.black.opacity(0.08)
    }
    /// Legacy inset color retained for compatibility with existing modifiers.
    var glassShadowInset: Color {
        isDark ? Color.black.opacity(0.20) : Color.black.opacity(0.03)
    }

    // MARK: - Typography helpers
    // Falls back to system fonts when Geist isn't installed.
    //
    // Each helper returns a Font that respects iOS Dynamic Type by mapping
    // the design's nominal size to the closest semantic TextStyle relative-
    // weight. The result still LOOKS like our design at default settings, but
    // scales when the user has Larger Text enabled in iOS Accessibility.

    /// Display / sans body. Uses Geist when the font is bundled, otherwise
    /// falls back to the system default. The Kodu prototype targets Geist
    /// specifically; this respects that intent without crashing on devices
    /// that don't have the file.
    func display(_ size: CGFloat, _ weight: Font.Weight = .semibold) -> Font {
        Self.named("Geist", size: size, weight: weight)
            ?? .system(size: size, weight: weight, design: .default)
    }

    func sans(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        Self.named("Geist", size: size, weight: weight)
            ?? .system(size: size, weight: weight, design: .default)
    }

    /// Monospaced — pervasive in this design language. Prefers Geist Mono,
    /// falls back to the system monospaced face.
    func mono(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        Self.named("Geist Mono", size: size, weight: weight)
            ?? .system(size: size, weight: weight, design: .monospaced)
    }

    /// Returns nil when the family isn't registered with UIFont — the caller
    /// then falls back to a system font. Cached per (family,weight) since
    /// CTFontManager lookups aren't free.
    private static func named(_ family: String, size: CGFloat, weight: Font.Weight) -> Font? {
        let key = "\(family)|\(weight.weightString)"
        if let cached = fontCache[key] {
            return cached.map { Font.custom($0, size: size).weight(weight) }
        }
        // Probe UIFont — cheap miss is fine, we'll cache the negative result.
        let candidate = uiFontName(family: family, weight: weight)
        if UIFont(name: candidate, size: size) != nil {
            fontCache[key] = candidate
            return Font.custom(candidate, size: size).weight(weight)
        }
        fontCache[key] = .some(nil)   // negative cache
        return nil
    }

    /// Font lookups happen on the main thread only (SwiftUI body evaluation).
    /// Marking nonisolated(unsafe) is honest about the contract and silences
    /// the Swift 6 strict-concurrency warning without forcing an actor hop.
    nonisolated(unsafe) private static var fontCache: [String: String?] = [:]

    /// Maps "Geist" + .semibold etc → "Geist-SemiBold" / "GeistMono-Regular".
    /// Foundries register weighted variants this way; an exact name avoids
    /// CoreText synthetic-bolding the regular face.
    private static func uiFontName(family: String, weight: Font.Weight) -> String {
        let base = family.replacingOccurrences(of: " ", with: "")
        return "\(base)-\(weight.weightString)"
    }
}

private extension Font.Weight {
    var weightString: String {
        switch self {
        case .ultraLight: return "UltraLight"
        case .thin:       return "Thin"
        case .light:      return "Light"
        case .regular:    return "Regular"
        case .medium:     return "Medium"
        case .semibold:   return "SemiBold"
        case .bold:       return "Bold"
        case .heavy:      return "Heavy"
        case .black:      return "Black"
        default:          return "Regular"
        }
    }
}

// MARK: - Dynamic Type modifier

extension View {
    /// Limits how aggressively a view scales with iOS Larger Text settings.
    /// We allow up to `.accessibility2` so the design doesn't blow up.
    func koduScaledType() -> some View {
        self.dynamicTypeSize(...DynamicTypeSize.accessibility2)
    }
}

// MARK: - Environment

private struct KoduThemeKey: EnvironmentKey {
    static let defaultValue: KoduTheme = .light
}

extension EnvironmentValues {
    var koduTheme: KoduTheme {
        get { self[KoduThemeKey.self] }
        set { self[KoduThemeKey.self] = newValue }
    }
}

extension View {
    /// Inject a KoduTheme into the environment.
    func koduTheme(_ theme: KoduTheme) -> some View {
        self.environment(\.koduTheme, theme)
    }
}
