import Combine
import Foundation

/// Lightweight notification sink retained for shared runtime services.
///
/// The full toast and confetti overlays belong to the removed
/// assistant UI. Local API clients do not need an on-device overlay, but the
/// inference and model services still report progress through this interface.
@MainActor
final class ToastCenter: ObservableObject {
    static let shared = ToastCenter()

    @Published private(set) var current: Toast?
    @Published private(set) var topPadding: CGFloat = 52

    private init() {}

    func error(_ title: String, detail: String? = nil, duration: Double = 5.0) {
        current = Toast(kind: .error, title: title, detail: detail, duration: duration)
    }

    func success(_ title: String, detail: String? = nil, duration: Double = 2.5) {
        current = Toast(kind: .success, title: title, detail: detail, duration: duration)
    }

    func info(_ title: String, detail: String? = nil, duration: Double = 3.0) {
        current = Toast(kind: .info, title: title, detail: detail, duration: duration)
    }

    func dismiss() {
        current = nil
    }

    func setTopPadding(_ padding: CGFloat) {
        topPadding = padding
    }
}

struct Toast: Identifiable, Equatable {
    let id = UUID()
    let kind: Kind
    let title: String
    let detail: String?
    let duration: Double

    enum Kind: Equatable {
        case error
        case success
        case info
    }
}
