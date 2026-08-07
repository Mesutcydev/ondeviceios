import SwiftUI

/// Keychain-backed Hugging Face access-token management for gated model
/// downloads. The raw token is never copied into the runtime log.
struct LASHFTokenSettingsView: View {
    @ObservedObject private var tokenStore = HFTokenStore.shared
    @Environment(\.dismiss) private var dismiss

    @State private var tokenText = ""
    @State private var isValidating = false
    @State private var statusMessage: String?
    @State private var statusIsError = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    SecureField("hf_… token", text: $tokenText)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textContentType(.password)

                    HStack {
                        Label(
                            tokenStore.hasToken
                                ? (tokenStore.lastValidatedUsername.map { "Saved · @\($0)" } ?? "Saved token")
                                : "No token saved",
                            systemImage: tokenStore.hasToken
                                ? "checkmark.shield.fill"
                                : "lock.open"
                        )
                        .foregroundStyle(tokenStore.hasToken ? .green : .secondary)
                        Spacer()
                        if let preview = tokenStore.maskedPreview {
                            Text(preview)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Hugging Face access")
                } footer: {
                    Text("A token is only needed for gated or private repositories. It is stored in this device's Keychain and is never shown in the verbose terminal.")
                }

                Section {
                    Button {
                        validate()
                    } label: {
                        if isValidating {
                            ProgressView()
                                .frame(maxWidth: .infinity)
                        } else {
                            Label("Validate token", systemImage: "checkmark.seal")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .disabled(tokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isValidating)

                    Button("Save token", systemImage: "key.fill") {
                        save()
                    }
                    .disabled(tokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    if tokenStore.hasToken {
                        Button("Clear saved token", systemImage: "trash", role: .destructive) {
                            tokenStore.clear()
                            tokenText = ""
                            statusMessage = "Saved token cleared"
                            statusIsError = false
                            RuntimeLogCenter.emit("Hugging Face token cleared", subsystem: "huggingFace")
                        }
                    }
                }

                if let statusMessage {
                    Section {
                        Label(
                            statusMessage,
                            systemImage: statusIsError
                                ? "exclamationmark.triangle.fill"
                                : "checkmark.circle.fill"
                        )
                        .foregroundStyle(statusIsError ? .red : .green)
                    }
                }
            }
            .navigationTitle("Hugging Face token")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
    }

    private func validate() {
        let candidate = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty else { return }
        isValidating = true
        statusMessage = nil

        Task { @MainActor in
            let username = await tokenStore.validate(candidate)
            isValidating = false
            if let username {
                statusMessage = "Validated for @\(username)"
                statusIsError = false
            } else {
                statusMessage = "Token was rejected or Hugging Face was unreachable"
                statusIsError = true
            }
        }
    }

    private func save() {
        let candidate = tokenText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !candidate.isEmpty else { return }
        guard tokenStore.save(candidate) else {
            statusMessage = "Could not save token to Keychain"
            statusIsError = true
            return
        }
        statusMessage = "Token saved"
        statusIsError = false
        RuntimeLogCenter.emit("Hugging Face token saved", subsystem: "huggingFace")
    }
}
