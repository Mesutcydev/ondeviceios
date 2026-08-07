import Foundation
import Security

// MARK: - KeychainStore
// Minimal wrapper around the Security framework for API-key storage. Keys
// are stored with `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` so they
// survive backup OFF the device's keychain but stay locked when the device
// is asleep.
//
// Used by the Web Tool's API-key providers and the local API's bearer key.
// Never persists chat history, prompts, or any user content.

public enum KeychainStore {

    /// Account name namespace — keep it scoped to the web-tool so other parts
    /// of the app can use their own keychain space if they want one.
    // Keep this product's keychain namespace separate from the reference app.
    // A sideloaded clone must not read or overwrite the reference app's keys.
    public static let service = "com.mesutcydev.ondevicelas.webtool"

    public static func set(
        _ value: String,
        account: String,
        serviceName: String = KeychainStore.service
    ) {
        guard let data = value.data(using: .utf8) else { return }
        let q: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        // Try update first; fall back to add.
        let upd: [String: Any] = [kSecValueData as String: data]
        let status = SecItemUpdate(q as CFDictionary, upd as CFDictionary)
        if status == errSecItemNotFound {
            var add = q
            add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            SecItemAdd(add as CFDictionary, nil)
        }
    }

    public static func get(
        account: String,
        serviceName: String = KeychainStore.service
    ) -> String? {
        let q: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account,
            kSecReturnData as String:  true,
            kSecMatchLimit as String:  kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        let status = withUnsafeMutablePointer(to: &item) { ptr -> OSStatus in
            SecItemCopyMatching(q as CFDictionary, ptr)
        }
        guard status == errSecSuccess,
              let data = item as? Data,
              let str = String(data: data, encoding: .utf8) else { return nil }
        return str
    }

    public static func delete(
        account: String,
        serviceName: String = KeychainStore.service
    ) {
        let q: [String: Any] = [
            kSecClass as String:       kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: account
        ]
        SecItemDelete(q as CFDictionary)
    }

    /// Removes every web-tool API key stored under this service namespace.
    static func deleteAll() {
        for account in knownAccounts {
            delete(account: account)
        }
    }

    private static let knownAccounts = [
        "brave.apiKey",
        "tavily.apiKey",
        "exa.apiKey",
        "localAPI.bearerKey",
    ]

    public static func has(
        account: String,
        serviceName: String = KeychainStore.service
    ) -> Bool {
        get(account: account, serviceName: serviceName) != nil
    }
}
