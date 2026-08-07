import Foundation

/// Curated Core AI packs that are known to target iPhone / iOS bundles.
/// Direct Hub links are first-class so operators can verify license and
/// files before downloading.
struct CoreAIZooModel: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let subtitle: String
    let hfRepo: String
    let revision: String
    /// When non-nil, only files under this repo subtree are downloaded
    /// (e.g. `"ios"` or a nested GPU-pipelined pack).
    let pathPrefix: String?
    let approxDownloadBytes: Int64
    let contextWindow: Int
    let supportsThinking: Bool
    let supportsTools: Bool
    let licenseNotice: String

    var hubURL: URL {
        URL(string: "https://huggingface.co/\(hfRepo)")!
    }

    var treeURL: URL {
        if let pathPrefix, !pathPrefix.isEmpty {
            return URL(string: "https://huggingface.co/\(hfRepo)/tree/\(revision)/\(pathPrefix)")!
        }
        return URL(string: "https://huggingface.co/\(hfRepo)/tree/\(revision)")!
    }
}

enum CoreAIZooCatalog {
    /// Hand-curated iPhone-first LLM packs. Prefer official-recipe iOS
    /// trees when available; otherwise use compact community packs that
    /// ship a complete resource directory (metadata + tokenizer + .aimodel).
    static let iphoneLanguageModels: [CoreAIZooModel] = [
        CoreAIZooModel(
            id: "zoo-qwen3-0.6b-official-ios",
            displayName: "Qwen3 0.6B · Official iOS",
            subtitle: "Apple recipe · mlboydaisuke · ios/",
            hfRepo: "mlboydaisuke/qwen3-0.6b-CoreAI-official",
            revision: "main",
            pathPrefix: "ios",
            approxDownloadBytes: 450_000_000,
            contextWindow: 4_096,
            supportsThinking: true,
            supportsTools: false,
            licenseNotice: "Apache-2.0 (Qwen). Verify the Hub card before redistribution."
        ),
        CoreAIZooModel(
            id: "zoo-qwen3-4b-official-ios",
            displayName: "Qwen3 4B · Official iOS",
            subtitle: "Apple recipe · mlboydaisuke · ios/",
            hfRepo: "mlboydaisuke/qwen3-4b-CoreAI-official",
            revision: "main",
            pathPrefix: "ios",
            approxDownloadBytes: 2_400_000_000,
            contextWindow: 4_096,
            supportsThinking: true,
            supportsTools: false,
            licenseNotice: "Apache-2.0 (Qwen). Larger footprint — check free disk and RAM."
        ),
        CoreAIZooModel(
            id: "zoo-qwen35-0.8b",
            displayName: "Qwen3.5 0.8B",
            subtitle: "Community pack · ChatML · iPhone-friendly",
            hfRepo: "mlboydaisuke/qwen3.5-0.8B-CoreAI",
            revision: "main",
            pathPrefix: nil,
            approxDownloadBytes: 700_000_000,
            contextWindow: 8_192,
            supportsThinking: true,
            supportsTools: true,
            licenseNotice: "Apache-2.0 (Qwen). Import the inner resource folder if nested."
        ),
        CoreAIZooModel(
            id: "zoo-qwen35-2b",
            displayName: "Qwen3.5 2B",
            subtitle: "Community pack · stronger coding/chat",
            hfRepo: "mlboydaisuke/qwen3.5-2B-CoreAI",
            revision: "main",
            pathPrefix: nil,
            approxDownloadBytes: 1_600_000_000,
            contextWindow: 8_192,
            supportsThinking: true,
            supportsTools: true,
            licenseNotice: "Apache-2.0 (Qwen). Prefer GPU-pipelined iOS subtree when present."
        ),
        CoreAIZooModel(
            id: "zoo-lfm25-1.2b",
            displayName: "LFM2.5 1.2B Instruct",
            subtitle: "LiquidAI · compact instruction model",
            hfRepo: "mlboydaisuke/LFM2.5-1.2B-CoreAI",
            revision: "main",
            pathPrefix: nil,
            approxDownloadBytes: 900_000_000,
            contextWindow: 8_192,
            supportsThinking: false,
            supportsTools: false,
            licenseNotice: "Review LiquidAI license on the Hub card."
        ),
        CoreAIZooModel(
            id: "zoo-nanbeige41-3b",
            displayName: "Nanbeige4.1 3B",
            subtitle: "Community pack · Chinese/English chat",
            hfRepo: "mlboydaisuke/Nanbeige4.1-3B-CoreAI",
            revision: "main",
            pathPrefix: nil,
            approxDownloadBytes: 2_000_000_000,
            contextWindow: 8_192,
            supportsThinking: false,
            supportsTools: false,
            licenseNotice: "Review Nanbeige license on the Hub card."
        ),
        CoreAIZooModel(
            id: "zoo-qwen25-coder-1.5b",
            displayName: "Qwen2.5 Coder 1.5B",
            subtitle: "Coding-focused · community Core AI pack",
            hfRepo: "kevinqz/Qwen2.5-Coder-1.5B-Instruct-CoreAI",
            revision: "main",
            pathPrefix: nil,
            approxDownloadBytes: 1_100_000_000,
            contextWindow: 8_192,
            supportsThinking: false,
            supportsTools: true,
            licenseNotice: "Apache-2.0 (Qwen). Confirm tokenizer + metadata.json are present."
        )
    ]

    static let zooIndexURL = URL(
        string: "https://raw.githubusercontent.com/john-rocky/coreai-model-zoo/main/models/index.json"
    )!

    static let zooHomeURL = URL(string: "https://github.com/john-rocky/coreai-model-zoo")!
}
