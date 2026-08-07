import Foundation
import Combine

/// Small, in-app event stream for the server operator.
///
/// The terminal view intentionally receives structured, safe events instead
/// of scraping stdout. Callers must keep credentials, prompts, model output,
/// and captured media out of messages.
@MainActor
final class RuntimeLogCenter: ObservableObject {
    static let shared = RuntimeLogCenter()

    enum Level: String, CaseIterable, Sendable {
        case debug = "DEBUG"
        case info = "INFO"
        case warning = "WARN"
        case error = "ERROR"

        var symbol: String {
            switch self {
            case .debug: return "·"
            case .info: return "›"
            case .warning: return "!"
            case .error: return "×"
            }
        }
    }

    struct Entry: Identifiable, Equatable, Sendable {
        let id: UUID
        let timestamp: Date
        let level: Level
        let subsystem: String
        let message: String

        init(
            id: UUID = UUID(),
            timestamp: Date = Date(),
            level: Level,
            subsystem: String,
            message: String
        ) {
            self.id = id
            self.timestamp = timestamp
            self.level = level
            self.subsystem = subsystem
            self.message = message
        }
    }

    @Published private(set) var entries: [Entry] = []

    private let maximumEntries = 800

    private init() {}

    func append(
        _ message: String,
        level: Level = .info,
        subsystem: String = "runtime"
    ) {
        let normalized = message
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }

        entries.append(Entry(
            level: level,
            subsystem: subsystem,
            message: normalized
        ))
        if entries.count > maximumEntries {
            entries.removeFirst(entries.count - maximumEntries)
        }
    }

    func clear() {
        entries.removeAll(keepingCapacity: true)
    }

    // Safe bridge for URLSession delegates and other non-main-actor callers.
    nonisolated static func emit(
        _ message: String,
        level: Level = .info,
        subsystem: String = "runtime"
    ) {
        Task { @MainActor in
            RuntimeLogCenter.shared.append(
                message,
                level: level,
                subsystem: subsystem
            )
        }
    }
}
