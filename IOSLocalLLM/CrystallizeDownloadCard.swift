import SwiftUI

// MARK: - Crystallize concept
//
// As chunks of a model file download, a 12×3 grid of cells "crystallizes"
// from gray liquid into black solid. Each newly filled cell snap-flips
// (scale overshoot + 180° rotation) like ice spreading across a surface.

/// Visual phase for one crystallize grid cell.
enum CrystallizeCellState: Equatable {
    /// Not yet downloaded — loose, light gray, slightly shrunk.
    case liquid
    /// Progress just crossed this cell — playing the snap-flip.
    case freezing
    /// Downloaded — locked black, full scale, rotated.
    case frozen
}

/// Model download card with an experimental crystallization progress grid.
struct CrystallizeDownloadCard: View {
    @Binding var isDownloading: Bool
    @Binding var progress: Double
    @Binding var speedMBps: Double
    @Binding var etaSeconds: Int?

    var fileName: String
    var fileSize: String

    static let columnCount = 12
    static let rowCount = 3
    static let cellCount = columnCount * rowCount

    @State private var cellStates: [CrystallizeCellState]
    /// Indices that have already played (or skipped) the freeze animation.
    @State private var animatedIndices: Set<Int> = []
    @State private var ambientOpacity: Double = 0.45

    @Environment(\.koduTheme) private var T

    private var borderIdle: Color { LASDesignTokens.hairline }
    private var borderActive: Color { T.ink.opacity(0.85) }
    private var footerIdle: Color { T.ink3 }
    private var sizeColor: Color { T.ink3 }

    init(
        isDownloading: Binding<Bool>,
        progress: Binding<Double>,
        speedMBps: Binding<Double>,
        etaSeconds: Binding<Int?>,
        fileName: String,
        fileSize: String
    ) {
        self._isDownloading = isDownloading
        self._progress = progress
        self._speedMBps = speedMBps
        self._etaSeconds = etaSeconds
        self.fileName = fileName
        self.fileSize = fileSize

        let initialFrozen = Self.frozenCount(for: progress.wrappedValue)
        _cellStates = State(
            initialValue: (0..<Self.cellCount).map { $0 < initialFrozen ? .frozen : .liquid }
        )
        if progress.wrappedValue > 0 || isDownloading.wrappedValue {
            _animatedIndices = State(initialValue: Set(0..<initialFrozen))
        }
    }

    var body: some View {
        VStack(spacing: 10) {
            CrystallizeHeader(
                fileName: fileName,
                fileSize: fileSize,
                titleColor: T.ink,
                sizeColor: sizeColor
            )
            CrystallizeGrid(
                cellStates: cellStates,
                ambientOpacity: isDownloading ? 1.0 : ambientOpacity,
                solidColor: T.ink,
                liquidColor: Color.primary.opacity(0.08),
                onFreezeFinished: { index in markFrozen(index) }
            )
            CrystallizeFooter(
                progress: progress,
                speedMBps: speedMBps,
                etaSeconds: etaSeconds,
                isDownloading: isDownloading,
                activeColor: T.ink,
                idleColor: footerIdle
            )
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 18)
        .frame(maxWidth: 520)
        .glassSurface(.card, cornerRadius: 16)
        .overlay {
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(isDownloading ? borderActive : borderIdle, lineWidth: 1.5)
        }
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .animation(
            .timingCurve(0.4, 0, 0.2, 1, duration: 0.6),
            value: isDownloading
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(
            "Downloading \(fileName), \(Int(progress.rounded(.down))) percent complete"
        )
        .onAppear {
            syncCells(to: progress, animateNew: false)
            updateAmbientPulse(isDownloading: isDownloading)
        }
        .onChange(of: progress) { _, newProgress in
            syncCells(to: newProgress, animateNew: isDownloading)
        }
        .onChange(of: isDownloading) { _, downloading in
            updateAmbientPulse(isDownloading: downloading)
            if downloading {
                syncCells(to: progress, animateNew: true)
            } else if progress <= 0 {
                // Only clear the grid when progress is fully reset (cancel /
                // new idle). Paused mid-download must keep frozen cells.
                resetToIdle()
            } else {
                // Settle any in-flight freezing cells without wiping progress.
                syncCells(to: progress, animateNew: false)
            }
        }
    }

    // MARK: - Progress → cells

    private static func frozenCount(for progress: Double) -> Int {
        let clamped = min(max(progress, 0), 100)
        return Int((clamped / 100.0) * Double(cellCount))
    }

    private func syncCells(to progress: Double, animateNew: Bool) {
        let count = Self.frozenCount(for: progress)

        if !animateNew {
            for index in cellStates.indices {
                cellStates[index] = index < count ? .frozen : .liquid
            }
            animatedIndices = Set(0..<count)
            return
        }

        for index in count..<Self.cellCount where cellStates[index] != .liquid {
            cellStates[index] = .liquid
            animatedIndices.remove(index)
        }

        let newlyCrossed = (0..<count).filter { !animatedIndices.contains($0) }
        for index in newlyCrossed {
            animatedIndices.insert(index)
            cellStates[index] = .freezing
        }
    }

    private func markFrozen(_ index: Int) {
        guard cellStates.indices.contains(index), cellStates[index] == .freezing else { return }
        cellStates[index] = .frozen
    }

    private func resetToIdle() {
        animatedIndices.removeAll()
        for index in cellStates.indices {
            cellStates[index] = .liquid
        }
    }

    private func updateAmbientPulse(isDownloading: Bool) {
        if isDownloading {
            withAnimation(.easeOut(duration: 0.25)) {
                ambientOpacity = 1.0
            }
        } else {
            ambientOpacity = 0.3
            withAnimation(.easeInOut(duration: 4).repeatForever(autoreverses: true)) {
                ambientOpacity = 0.6
            }
        }
    }

    /// Formats remaining download time for the footer ETA label.
    static func formatETA(_ seconds: Int?) -> String {
        guard let seconds, seconds > 0 else { return "—" }
        if seconds < 60 { return "\(seconds)s" }
        let m = seconds / 60
        let s = seconds % 60
        return "\(m)m \(s)s"
    }
}

// MARK: - Header

private struct CrystallizeHeader: View {
    let fileName: String
    let fileSize: String
    let titleColor: Color
    let sizeColor: Color

    var body: some View {
        HStack {
            Text(fileName)
                .font(.system(.subheadline, design: .monospaced).weight(.semibold))
                .foregroundStyle(titleColor)
                .tracking(-0.3)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 8)
            Text(fileSize)
                .font(.system(.caption, design: .monospaced).weight(.medium))
                .foregroundStyle(sizeColor)
        }
    }
}

// MARK: - Grid

private struct CrystallizeGrid: View {
    let cellStates: [CrystallizeCellState]
    let ambientOpacity: Double
    let solidColor: Color
    let liquidColor: Color
    let onFreezeFinished: (Int) -> Void

    var body: some View {
        VStack(spacing: 3) {
            ForEach(0..<CrystallizeDownloadCard.rowCount, id: \.self) { row in
                HStack(spacing: 3) {
                    ForEach(0..<CrystallizeDownloadCard.columnCount, id: \.self) { column in
                        let index = row * CrystallizeDownloadCard.columnCount + column
                        CrystallizeCellView(
                            index: index,
                            state: cellStates[index],
                            ambientOpacity: ambientOpacity,
                            solidColor: solidColor,
                            liquidColor: liquidColor,
                            onFreezeFinished: onFreezeFinished
                        )
                        .aspectRatio(1, contentMode: .fit)
                        .frame(maxWidth: .infinity)
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }
}

private struct CrystallizeCellView: View {
    let index: Int
    let state: CrystallizeCellState
    let ambientOpacity: Double
    let solidColor: Color
    let liquidColor: Color
    let onFreezeFinished: (Int) -> Void

    @State private var displayScale: CGFloat = 0.7
    @State private var displayRotation: Double = 0
    @State private var isSolid = false

    var body: some View {
        RoundedRectangle(cornerRadius: 2, style: .continuous)
            .fill(isSolid ? solidColor : liquidColor)
            .scaleEffect(displayScale)
            .rotationEffect(.degrees(displayRotation))
            .opacity(state == .liquid ? ambientOpacity : 1)
            .onAppear {
                applySettledAppearance(for: state, animated: false)
            }
            .onChange(of: state) { oldState, newState in
                if newState == .freezing, oldState != .freezing {
                    runFreezeAnimation()
                } else if newState != .freezing {
                    applySettledAppearance(for: newState, animated: true)
                }
            }
    }

    private func applySettledAppearance(for state: CrystallizeCellState, animated: Bool) {
        let updates = {
            switch state {
            case .liquid:
                displayScale = 0.7
                displayRotation = 0
                isSolid = false
            case .freezing:
                break
            case .frozen:
                displayScale = 1.0
                displayRotation = 180
                isSolid = true
            }
        }
        if animated {
            withAnimation(.easeInOut(duration: 0.4), updates)
        } else {
            updates()
        }
    }

    /// Snap-flip: shrink → overshoot pop with 180° rotation → settle solid black (0.4s total).
    private func runFreezeAnimation() {
        displayScale = 0.5
        displayRotation = 0
        isSolid = false

        let curve = Animation.timingCurve(0.4, 0, 0.2, 1, duration: 0.2)

        withAnimation(curve) {
            displayScale = 1.15
            displayRotation = 180
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            withAnimation(curve) {
                displayScale = 1.0
                isSolid = true
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                onFreezeFinished(index)
            }
        }
    }
}

// MARK: - Footer

private struct CrystallizeFooter: View {
    let progress: Double
    let speedMBps: Double
    let etaSeconds: Int?
    let isDownloading: Bool
    let activeColor: Color
    let idleColor: Color

    var body: some View {
        HStack {
            Text("\(Int(progress.rounded(.down)))%")
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(speedLabel)
                .frame(maxWidth: .infinity, alignment: .center)
            Text(CrystallizeDownloadCard.formatETA(isDownloading ? etaSeconds : nil))
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .foregroundStyle(isDownloading ? activeColor : idleColor)
        .animation(.easeInOut(duration: 0.5), value: isDownloading)
    }

    private var speedLabel: String {
        guard isDownloading else { return "— MB/s" }
        return String(format: "%.1f MB/s", speedMBps)
    }
}

// MARK: - Previews

struct CrystallizeDownloadCard_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 20) {
            CrystallizeDownloadCard(
                isDownloading: .constant(false),
                progress: .constant(0),
                speedMBps: .constant(0),
                etaSeconds: .constant(nil),
                fileName: "gemma-2b-it.gguf",
                fileSize: "1.6 GB"
            )

            CrystallizeDownloadCard(
                isDownloading: .constant(true),
                progress: .constant(45),
                speedMBps: .constant(8.3),
                etaSeconds: .constant(134),
                fileName: "gemma-2b-it.gguf",
                fileSize: "1.6 GB"
            )

            CrystallizeDownloadCard(
                isDownloading: .constant(true),
                progress: .constant(100),
                speedMBps: .constant(0),
                etaSeconds: .constant(0),
                fileName: "gemma-2b-it.gguf",
                fileSize: "1.6 GB"
            )
        }
        .padding()
        .background(Color(red: 0.96, green: 0.96, blue: 0.96))
        .previewLayout(.sizeThatFits)
        .previewDisplayName("Idle · 45% · Complete")
    }
}
