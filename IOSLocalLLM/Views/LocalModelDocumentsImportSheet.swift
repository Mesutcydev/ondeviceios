import SwiftUI

/// Sideload-proof folder import: pick a model folder already inside the app
/// sandbox (copied via Files / Finder with UIFileSharingEnabled).
struct LocalModelDocumentsImportSheet: View {
    /// Display name shown in the empty-state copy (Files → On My iPhone → …).
    var appDocumentsName: String = "On Device: LAS"
    let onPick: (URL) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var folders: [URL] = []

    init(
        appDocumentsName: String = "On Device: LAS",
        onPick: @escaping (URL) -> Void
    ) {
        self.appDocumentsName = appDocumentsName
        self.onPick = onPick
    }

    var body: some View {
        NavigationStack {
            Group {
                if folders.isEmpty {
                    ContentUnavailableView(
                        "No model folders found",
                        systemImage: "folder",
                        description: Text(
                            "In the Files app, copy your model folder into \(appDocumentsName) (On My iPhone), then return here."
                        )
                    )
                } else {
                    List(folders, id: \.path) { url in
                        Button {
                            onPick(url)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(url.lastPathComponent)
                                    .font(.body.weight(.semibold))
                                    .foregroundStyle(.primary)
                                Text(url.deletingLastPathComponent().lastPathComponent)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("App Documents")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        folders = LocalModelDocumentsScanner.candidateModelFolders()
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .onAppear {
                folders = LocalModelDocumentsScanner.candidateModelFolders()
            }
        }
    }
}
