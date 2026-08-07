import Foundation
import Security
import Combine

// MARK: - HFTokenStore
//
// Keychain-backed store for the user's Hugging Face access token.
// Used to authenticate downloads of gated repos (google/gemma-3-*,
// meta-llama/*, mistralai/Ministral-*, etc.).
//
// Storage: Generic password under service "com.mesutcydev.ioslocalllm.huggingface".
// Accessibility is `AfterFirstUnlockThisDeviceOnly` — survives reboots
// but never leaves the device or syncs via iCloud Keychain. Encryption
// is iOS Data Protection, tied to the user's passcode.
//
// Threading: @MainActor for the @Published surface so SwiftUI views
// observe `hasToken` directly without re-reading Keychain on every
// body evaluation. The actual Keychain calls happen on the calling
// thread (cheap — kSecMatchLimitOne completes in microseconds).
//
// Why a separate store from AppSettings:
//   • AppSettings is @AppStorage backed (NSUserDefaults), which is
//     unencrypted and visible in iTunes backups in cleartext.
//   • Tokens are credentials, not preferences.
//   • AppSettings.useHFToken stays in defaults (it's a boolean
//     preference, not a secret).

@MainActor
final class HFTokenStore: ObservableObject {

    static let shared = HFTokenStore()

    // MARK: - Observable surface

    /// True iff a non-empty token is stored. SwiftUI binds against this
    /// (e.g. to swap "Set Token" → "Update Token" / show a green dot).
    @Published private(set) var hasToken: Bool = false

    /// Masked preview of the stored token for display purposes —
    /// `hf_••••abcd`. nil when no token stored. Never expose the
    /// raw token in UI; the user typed it and there's no reason to
    /// surface it again.
    @Published private(set) var maskedPreview: String? = nil

    /// Username returned by the last successful `validate(_:)` call.
    /// Cleared on save/clear. Surface as a "Signed in as @user" eyebrow.
    @Published private(set) var lastValidatedUsername: String? = nil

    // MARK: - Keychain identifiers
    //
    // Keep these stable — renaming would orphan tokens stored under
    // the old service name and force users to re-enter.
    private let service = "com.mesutcydev.ioslocalllm.huggingface"
    private let account = "default-token"

    private init() {
        refresh()
    }

    // MARK: - Read

    /// Returns the raw token string, or nil if none stored.
    /// Call this when building Authorization headers — NEVER cache
    /// the result in memory longer than the single request.
    func currentToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String:  true,
            kSecMatchLimit as String:  kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = withUnsafeMutablePointer(to: &item) {
            SecItemCopyMatching(query as CFDictionary, $0)
        }
        guard status == errSecSuccess,
              let data = item as? Data,
              let str = String(data: data, encoding: .utf8),
              !str.isEmpty else {
            return nil
        }
        return str
    }

    // MARK: - Write

    /// Save `token` to Keychain. Trimmed of whitespace first. Replaces
    /// any existing entry. Returns true on success.
    @discardableResult
    func save(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let data = trimmed.data(using: .utf8) else { return false }

        // No upsert in Keychain — delete then add.
        _ = deleteRaw()

        let query: [String: Any] = [
            kSecClass as String:           kSecClassGenericPassword,
            kSecAttrService as String:     service,
            kSecAttrAccount as String:     account,
            kSecAttrAccessible as String:  kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
            kSecValueData as String:       data,
        ]
        let status = SecItemAdd(query as CFDictionary, nil)
        if status == errSecSuccess {
            lastValidatedUsername = nil
            refresh()
            return true
        }
        return false
    }

    @discardableResult
    func clear() -> Bool {
        let ok = deleteRaw()
        lastValidatedUsername = nil
        refresh()
        return ok
    }

    private func deleteRaw() -> Bool {
        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    // MARK: - Refresh + mask

    private func refresh() {
        let token = currentToken()
        hasToken = (token != nil)
        maskedPreview = token.map { Self.mask($0) }
    }

    private static func mask(_ token: String) -> String {
        guard token.count > 8 else { return "••••" }
        let prefix = token.prefix(3)
        let suffix = token.suffix(4)
        return "\(prefix)••••\(suffix)"
    }

    // MARK: - Validation
    //
    // Hits HF's /api/whoami-v2 with the supplied token. Used by the
    // "Test" button in the token sheet so the user gets immediate
    // feedback before saving. Returns the authenticated username on
    // success, nil on any failure (invalid token, network error,
    // unexpected payload). Doesn't write to the store — callers
    // decide whether to save based on the result.

    /// Validate `token` against the HF API. Returns username on
    /// success, nil otherwise. 10-second timeout.
    func validate(_ token: String) async -> String? {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let url = URL(string: "https://huggingface.co/api/whoami-v2") else {
            return nil
        }
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.setValue("Bearer \(trimmed)", forHTTPHeaderField: "Authorization")
        req.timeoutInterval = 10
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            let json: [String: Any]
            do {
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    Diagnostics.shared.warning(
                        "HF token validation returned an unexpected payload",
                        category: "huggingFace"
                    )
                    return nil
                }
                json = object
            } catch {
                Diagnostics.shared.warning(
                    "HF token validation payload decode failed: \(error.localizedDescription)",
                    category: "huggingFace"
                )
                return nil
            }
            if let name = json["name"] as? String, !name.isEmpty {
                lastValidatedUsername = name
                return name
            }
            return nil
        } catch {
            Diagnostics.shared.warning(
                "HF token validation request failed: \(error.localizedDescription)",
                category: "huggingFace"
            )
            return nil
        }
    }

    // MARK: - Convenience for downloaders
    //
    // Adds `Authorization: Bearer <token>` to `req` iff a token is
    // present AND the user has `useHFToken` enabled. No-op otherwise
    // (some users may want to disable the header to test public
    // routes or work around auth weirdness). Idempotent — safe to
    // call multiple times on the same request.

    /// True iff the Authorization header SHOULD be applied right now
    /// (token present AND user toggle enabled).
    var isActive: Bool {
        hasToken && AppSettings.shared.useHFToken
    }

    /// Mutates `req` in place, adding the Authorization header when
    /// `isActive`. No-op otherwise.
    func authorize(_ req: inout URLRequest) {
        guard isActive, let token = currentToken() else { return }
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    // MARK: - Nonisolated access for background contexts
    //
    // BackgroundDownloadCoordinator (URLSessionDelegate) isn't on the
    // main actor and can't await a hop just to read Keychain.
    // Keychain (Security framework) is thread-safe — the SecItem*
    // calls block briefly but can be called from any thread. The
    // useHFToken preference lives in UserDefaults, which is also
    // thread-safe.
    //
    // This static helper mirrors the (currentToken + isActive) logic
    // without requiring an actor hop, so background download tasks
    // can apply the header in their request-builder closure.

    /// Returns the full Authorization header value (`"Bearer <token>"`)
    /// when the user has a token AND `useHFToken` is enabled. Returns
    /// nil otherwise. Safe to call from any thread.
    nonisolated static func authorizationHeaderValue() -> String? {
        // Match AppSettings' default-true behavior: when no value has
        // ever been written to UserDefaults, treat the toggle as ON.
        // @AppStorage doesn't auto-persist defaults, so a fresh install
        // returns nil from `object(forKey:)` here.
        let raw = UserDefaults.standard.object(forKey: "useHFToken")
        let useToken = (raw as? Bool) ?? true
        guard useToken else { return nil }

        let query: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: "com.mesutcydev.ioslocalllm.huggingface",
            kSecAttrAccount as String: "default-token",
            kSecReturnData as String:  true,
            kSecMatchLimit as String:  kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = withUnsafeMutablePointer(to: &item) {
            SecItemCopyMatching(query as CFDictionary, $0)
        }
        guard status == errSecSuccess,
              let data = item as? Data,
              let token = String(data: data, encoding: .utf8),
              !token.isEmpty else {
            return nil
        }
        return "Bearer \(token)"
    }

    /// Apply the Authorization header to `req` if available. Identical
    /// to `authorize(_:)` but callable from any thread.
    nonisolated static func authorize(_ req: inout URLRequest) {
        guard let header = authorizationHeaderValue() else { return }
        req.setValue(header, forHTTPHeaderField: "Authorization")
    }
}

// MARK: - HFAuthError
//
// Structured error surfaced when a download fails specifically because
// of auth (401/403). Catalog UI can `as?` against this to render a
// helpful "Add a token in Settings" prompt instead of a generic
// failure toast.

enum HFAuthError: LocalizedError {
    /// The repo is gated and the request had no token (or an invalid one).
    case tokenRequired(repoID: String)
    /// The token was sent but rejected — wrong scope or revoked.
    case tokenRejected(repoID: String)

    var errorDescription: String? {
        switch self {
        case .tokenRequired(let r):
            return "\(r) requires a Hugging Face token. Add one in Settings → API Settings, then retry."
        case .tokenRejected(let r):
            return "Your Hugging Face token doesn't have access to \(r). Check the token or visit the repo to accept terms."
        }
    }
}
