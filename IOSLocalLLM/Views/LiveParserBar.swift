import SwiftUI

/// Compact, monochrome status for the model serving the local API.
///
/// A single adaptive timeline drives every animated element. Keeping the
/// animation clock at this boundary avoids installing separate display-link
/// schedules for the waveform, progress bar, dots, and shimmer.
struct LiveParserBar: View {
    let isGenerating: Bool
    let status: String
    let badge: String
    let tokenRate: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.koduTheme) private var T

    var body: some View {
        TimelineView(
            .animation(
                minimumInterval: isGenerating ? 1.0 / 60.0 : 1.0 / 20.0,
                paused: reduceMotion
            )
        ) { timeline in
            GeometryReader { proxy in
                let compact = proxy.size.width < 420
                let phase = timeline.date.timeIntervalSinceReferenceDate

                HStack(spacing: compact ? 8 : 12) {
                    LiveParserWaveIcon(
                        phase: phase,
                        isGenerating: isGenerating,
                        reduceMotion: reduceMotion,
                        tint: T.ink
                    )
                    .frame(width: 24, height: 24)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("LIVE PARSER")
                            .font(.system(size: 9, weight: .semibold))
                            .tracking(2)
                            .foregroundStyle(isGenerating ? T.ink : T.ink3)
                            .lineLimit(1)
                        Text(status)
                            .font(.system(size: 15, weight: isGenerating ? .semibold : .medium))
                            .lineLimit(1)
                    }
                    .frame(width: compact ? 84 : 92, alignment: .leading)

                    Text(verbatim: tokenRateText)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .monospacedDigit()
                        .lineLimit(1)
                        .frame(width: isGenerating ? (compact ? 48 : 54) : 0, alignment: .trailing)
                        .opacity(isGenerating ? 1 : 0)
                        .offset(x: isGenerating ? 0 : -4)

                    LiveParserProgress(
                        phase: phase,
                        isGenerating: isGenerating,
                        reduceMotion: reduceMotion,
                        fill: T.ink,
                        track: Color.primary.opacity(0.10)
                    )
                    .frame(maxWidth: .infinity)

                    LiveParserDots(
                        phase: phase,
                        isGenerating: isGenerating,
                        reduceMotion: reduceMotion,
                        tint: T.ink
                    )

                    Text(badge)
                        .font(.system(size: 9, weight: .semibold))
                        .tracking(compact ? 0.8 : 1.5)
                        .foregroundStyle(isGenerating ? T.ink : T.ink3)
                        .lineLimit(1)
                        .minimumScaleFactor(0.75)
                        .frame(width: compact ? 48 : 54, alignment: .trailing)
                }
                .padding(.horizontal, compact ? 12 : 18)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .bottomLeading) {
                    if isGenerating, !reduceMotion {
                        LiveParserShimmer(phase: phase, tint: T.ink)
                            .frame(height: 2)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                    }
                }
            }
        }
        .frame(maxWidth: 520)
        .frame(height: 68)
        .glassSurface(.card, cornerRadius: 16)
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .stroke(
                    isGenerating ? T.ink.opacity(0.85) : LASDesignTokens.hairline,
                    lineWidth: 1.5
                )
        }
        .animation(
            reduceMotion ? nil : .timingCurve(0.4, 0, 0.2, 1, duration: 0.5),
            value: isGenerating
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Live parser")
        .accessibilityValue(accessibilityValue)
    }

    private var tokenRateText: String {
        "\(Int(tokenRate.rounded())) tok/s"
    }

    private var accessibilityValue: String {
        if isGenerating, tokenRate > 0 {
            return "\(status), \(tokenRateText), \(badge)"
        }
        return "\(status), \(badge)"
    }
}

private struct LiveParserWaveIcon: View {
    let phase: TimeInterval
    let isGenerating: Bool
    let reduceMotion: Bool
    let tint: Color

    var body: some View {
        LiveParserWaveShape()
            .trim(from: 0, to: 1)
            .stroke(
                tint,
                style: StrokeStyle(
                    lineWidth: 2,
                    lineCap: .round,
                    lineJoin: .round,
                    dash: [12, 5],
                    dashPhase: dashPhase
                )
            )
            .padding(2)
            .accessibilityHidden(true)
    }

    private var dashPhase: CGFloat {
        guard isGenerating, !reduceMotion else { return 0 }
        return CGFloat(phase.truncatingRemainder(dividingBy: 0.5) / 0.5) * -17
    }
}

private struct LiveParserWaveShape: Shape {
    func path(in rect: CGRect) -> Path {
        let sx = rect.width / 18
        let sy = rect.height / 20
        var path = Path()
        path.move(to: CGPoint(x: 2 * sx, y: 10 * sy))
        path.addQuadCurve(
            to: CGPoint(x: 8 * sx, y: 10 * sy),
            control: CGPoint(x: 5 * sx, y: 4 * sy)
        )
        path.addQuadCurve(
            to: CGPoint(x: 14 * sx, y: 10 * sy),
            control: CGPoint(x: 11 * sx, y: 16 * sy)
        )
        path.addQuadCurve(
            to: CGPoint(x: 18 * sx, y: 10 * sy),
            control: CGPoint(x: 16 * sx, y: 4 * sy)
        )
        return path
    }
}

private struct LiveParserProgress: View {
    let phase: TimeInterval
    let isGenerating: Bool
    let reduceMotion: Bool
    let fill: Color
    let track: Color

    var body: some View {
        GeometryReader { proxy in
            let width = max(0, proxy.size.width * fillFraction)

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(track)
                Capsule()
                    .fill(fill)
                    .frame(width: width)
                    .overlay(alignment: .trailing) {
                        if isGenerating, width > 12 {
                            LinearGradient(
                                colors: [.white.opacity(0), .white.opacity(0.72)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                            .frame(width: min(12, width))
                            .clipShape(Capsule())
                        }
                    }
            }
        }
        .frame(minWidth: 32, idealWidth: 92, maxWidth: .infinity)
        .frame(height: 5)
        .accessibilityHidden(true)
    }

    private var fillFraction: CGFloat {
        guard !reduceMotion else { return isGenerating ? 0.55 : 0.16 }
        let duration = isGenerating ? 0.8 : 3.0
        let unit = phase.truncatingRemainder(dividingBy: duration) / duration
        let triangle = unit < 0.5 ? unit * 2 : (1 - unit) * 2
        let eased = triangle * triangle * (3 - 2 * triangle)
        let lower = isGenerating ? 0.15 : 0.08
        let upper = isGenerating ? 0.75 : 0.25
        return CGFloat(lower + (upper - lower) * eased)
    }
}

private struct LiveParserDots: View {
    let phase: TimeInterval
    let isGenerating: Bool
    let reduceMotion: Bool
    let tint: Color

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<5, id: \.self) { index in
                let strength = dotStrength(index: index)
                Circle()
                    .fill(tint)
                    .frame(width: 4, height: 4)
                    .scaleEffect(dotScale(strength))
                    .opacity(dotOpacity(strength))
            }
        }
        .frame(width: 32)
        .accessibilityHidden(true)
    }

    private func dotStrength(index: Int) -> Double {
        guard !reduceMotion else { return isGenerating ? 0.8 : 0.35 }
        let duration = isGenerating ? 0.4 : 2.0
        let stagger = isGenerating ? 0.08 : 0.15
        let local = (phase - Double(index) * stagger) * (2 * .pi / duration)
        return (sin(local) + 1) / 2
    }

    private func dotScale(_ strength: Double) -> CGFloat {
        let lower = isGenerating ? 0.2 : 0.5
        let upper = isGenerating ? 1.3 : 1.0
        return CGFloat(lower + (upper - lower) * strength)
    }

    private func dotOpacity(_ strength: Double) -> Double {
        let lower = isGenerating ? 0.2 : 0.15
        let upper = isGenerating ? 1.0 : 0.5
        return lower + (upper - lower) * strength
    }
}

private struct LiveParserShimmer: View {
    let phase: TimeInterval
    let tint: Color

    var body: some View {
        GeometryReader { proxy in
            let width = max(24, proxy.size.width * 0.22)
            let travel = proxy.size.width + width
            let unit = phase.truncatingRemainder(dividingBy: 1.15) / 1.15

            Rectangle()
                .fill(tint)
                .frame(width: width)
                .offset(x: -width + travel * unit)
        }
        .accessibilityHidden(true)
    }
}

#Preview("Live parser states") {
    VStack(spacing: 20) {
        LiveParserBar(
            isGenerating: false,
            status: "Listening",
            badge: "NO MODEL",
            tokenRate: 0
        )
        LiveParserBar(
            isGenerating: true,
            status: "Parsing",
            badge: "ACTIVE",
            tokenRate: 42
        )
    }
    .padding()
    .background(Color(white: 0.96))
}
