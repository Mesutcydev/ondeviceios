import Foundation

/// Foreground-only Hugging Face pack downloader for Core AI.
///
/// Sideloaded builds often stall forever on `URLSessionConfiguration.background`
/// (no delegate callbacks until a background relaunch). This manager uses a
/// dedicated foreground session with byte progress, timeouts, and retries so
/// multi-hundred-MB `ios/` packs actually finish while the app is open.
@MainActor
final class CoreAIHFDownloadManager: ObservableObject, Identifiable {
    enum State: Equatable {
        case idle
        case enumerating
        case downloading
        case paused
        case installing
        case ready
        case failed(String)

        var isActive: Bool {
            self == .enumerating || self == .downloading || self == .installing
        }
    }

    let id: String
    let repoID: String
    let revision: String
    let pathPrefix: String?
    let displayName: String

    @Published private(set) var state: State = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var downloadedBytes: Int64 = 0
    @Published private(set) var totalBytes: Int64 = 0
    @Published private(set) var currentFile = ""
    @Published private(set) var filesDone = 0
    @Published private(set) var filesTotal = 0
    @Published private(set) var speedMBps: Double = 0
    @Published private(set) var statusDetail = ""

    private var runTask: Task<Void, Never>?
    private var pauseRequested = false
    private var files: [RemoteFile] = []
    private var stagingRoot: URL?
    private var lastSampleBytes: Int64 = 0
    private var lastSampleAt = Date()
    private var committedBytes: Int64 = 0
    private var activeTransfer: CoreAIForegroundTransfer?

    private struct RemoteFile: Sendable {
        let remotePath: String
        let localPath: String
        var size: Int64
    }

    init(
        id: String,
        repoID: String,
        revision: String = "main",
        pathPrefix: String? = nil,
        displayName: String
    ) {
        self.id = id
        self.repoID = repoID
        self.revision = revision
        self.pathPrefix = pathPrefix
        self.displayName = displayName
    }

    convenience init(model: CoreAIZooModel) {
        self.init(
            id: model.id,
            repoID: model.hfRepo,
            revision: model.revision,
            pathPrefix: model.pathPrefix,
            displayName: model.displayName
        )
    }

    func start() {
        guard runTask == nil else { return }
        pauseRequested = false
        if state == .paused, stagingRoot != nil, !files.isEmpty {
            runTask = Task { [weak self] in
                await self?.downloadRemaining()
            }
            return
        }
        runTask = Task { [weak self] in
            await self?.run()
        }
    }

    func pause() {
        guard state == .downloading || state == .enumerating else { return }
        pauseRequested = true
        activeTransfer?.cancel()
        activeTransfer = nil
        runTask?.cancel()
        runTask = nil
        state = .paused
        statusDetail = "Paused"
        RuntimeLogCenter.emit("Paused Core AI download · \(repoID)", subsystem: "coreai")
    }

    func resume() {
        switch state {
        case .paused, .failed, .idle, .ready:
            pauseRequested = false
            start()
        default:
            break
        }
    }

    func cancel() {
        pauseRequested = false
        activeTransfer?.cancel()
        activeTransfer = nil
        runTask?.cancel()
        runTask = nil
        if let stagingRoot {
            try? FileManager.default.removeItem(at: stagingRoot)
        }
        stagingRoot = nil
        files = []
        state = .idle
        progress = 0
        downloadedBytes = 0
        committedBytes = 0
        totalBytes = 0
        filesDone = 0
        filesTotal = 0
        currentFile = ""
        speedMBps = 0
        statusDetail = ""
    }

    private func run() async {
        state = .enumerating
        currentFile = ""
        statusDetail = "Listing Hub files…"
        committedBytes = 0
        downloadedBytes = 0
        progress = 0
        do {
            let listed = try await listRemoteFiles()
            try Task.checkCancellation()
            guard !pauseRequested else {
                state = .paused
                runTask = nil
                return
            }
            files = listed
            filesTotal = listed.count
            totalBytes = listed.reduce(0) { $0 + max(0, $1.size) }
            statusDetail = "\(listed.count) files · \(ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file))"
            let staging = try prepareStaging()
            stagingRoot = staging
            state = .downloading
            RuntimeLogCenter.emit(
                "Core AI download started · \(displayName) · \(listed.count) files",
                subsystem: "coreai"
            )
            await downloadRemaining()
        } catch is CancellationError {
            if pauseRequested { state = .paused }
            runTask = nil
        } catch {
            state = .failed(error.localizedDescription)
            statusDetail = error.localizedDescription
            runTask = nil
            RuntimeLogCenter.emit(
                "Core AI download failed · \(error.localizedDescription)",
                level: .error,
                subsystem: "coreai"
            )
            ToastCenter.shared.error(error.localizedDescription)
        }
    }

    private func downloadRemaining() async {
        guard let stagingRoot else {
            state = .failed("Missing staging directory.")
            runTask = nil
            return
        }
        state = .downloading
        pauseRequested = false
        lastSampleBytes = downloadedBytes
        lastSampleAt = Date()

        committedBytes = 0
        for file in files {
            let destination = stagingRoot
                .appendingPathComponent("resources", isDirectory: true)
                .appendingPathComponent(file.localPath)
            if fileAlreadyComplete(at: destination, expectedSize: file.size) {
                committedBytes += max(file.size, 0)
            }
        }
        downloadedBytes = committedBytes
        recalculateProgress()

        do {
            for (index, file) in files.enumerated() {
                try Task.checkCancellation()
                if pauseRequested {
                    state = .paused
                    runTask = nil
                    return
                }
                let destination = stagingRoot
                    .appendingPathComponent("resources", isDirectory: true)
                    .appendingPathComponent(file.localPath)
                if fileAlreadyComplete(at: destination, expectedSize: file.size) {
                    filesDone = index + 1
                    recalculateProgress()
                    continue
                }
                currentFile = file.remotePath
                statusDetail = "File \(index + 1)/\(filesTotal)"
                try await downloadFile(file, to: destination)
                filesDone = index + 1
                recalculateProgress()
            }

            state = .installing
            currentFile = ""
            statusDetail = "Validating pack…"
            let resources = stagingRoot.appendingPathComponent("resources", isDirectory: true)
            try CoreAIModelStore.shared.importModel(
                from: resources,
                preferredID: id,
                preferredDisplayName: displayName
            )
            try? FileManager.default.removeItem(at: stagingRoot)
            self.stagingRoot = nil
            state = .ready
            progress = 1
            statusDetail = "Installed"
            currentFile = ""
            runTask = nil
            RuntimeLogCenter.emit("Core AI pack installed · \(displayName)", subsystem: "coreai")
            ToastCenter.shared.success("Installed \(displayName)")
            CodingAssistantService.shared.startLoad()
        } catch is CancellationError {
            if pauseRequested { state = .paused }
            runTask = nil
        } catch {
            if Self.isUserCancellation(error) {
                if pauseRequested { state = .paused }
                runTask = nil
                return
            }
            state = .failed(error.localizedDescription)
            statusDetail = error.localizedDescription
            runTask = nil
            RuntimeLogCenter.emit(
                "Core AI download failed · \(error.localizedDescription)",
                level: .error,
                subsystem: "coreai"
            )
            ToastCenter.shared.error(error.localizedDescription)
        }
    }

    private func prepareStaging() throws -> URL {
        let root = CoreAIModelStore.shared.modelDirectory
            .appendingPathComponent(".download-\(id)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let resources = root.appendingPathComponent("resources", isDirectory: true)
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        return root
    }

    private func listRemoteFiles() async throws -> [RemoteFile] {
        var collected: [RemoteFile] = []
        var cursor: String?
        repeat {
            var components = URLComponents(
                string: "https://huggingface.co/api/models/\(repoID)/tree/\(revision)"
            )!
            var items: [URLQueryItem] = [
                URLQueryItem(name: "recursive", value: "1")
            ]
            if let cursor {
                items.append(URLQueryItem(name: "cursor", value: cursor))
            }
            components.queryItems = items
            guard let url = components.url else { throw URLError(.badURL) }
            var request = URLRequest(url: url)
            request.timeoutInterval = 30
            request.setValue("application/json", forHTTPHeaderField: "Accept")
            request.setValue(
                "CoreAI-LAS/3.2.6 (iOS; Core AI pack download)",
                forHTTPHeaderField: "User-Agent"
            )
            HFTokenStore.authorize(&request)
            let (data, response) = try await withRetries {
                try await URLSession.shared.data(for: request)
            }
            guard let http = response as? HTTPURLResponse else { throw URLError(.badServerResponse) }
            if http.statusCode == 401 || http.statusCode == 403 {
                throw CoreAIDownloadError.tokenRequired
            }
            guard 200..<300 ~= http.statusCode else {
                throw CoreAIDownloadError.httpStatus(http.statusCode)
            }
            let page = try JSONDecoder().decode([HFTreeEntry].self, from: data)
            for entry in page where entry.type == "file" {
                if let pathPrefix, !pathPrefix.isEmpty {
                    let normalized = pathPrefix.hasSuffix("/") ? pathPrefix : pathPrefix + "/"
                    guard entry.path == pathPrefix
                        || entry.path.hasPrefix(normalized) else { continue }
                }
                let lower = entry.path.lowercased()
                if lower.hasSuffix(".md") || lower.hasSuffix(".gitattributes") { continue }
                let size = entry.lfs?.size ?? entry.size ?? 0
                collected.append(
                    RemoteFile(
                        remotePath: entry.path,
                        localPath: stripPrefix(entry.path),
                        size: size
                    )
                )
            }
            cursor = http.value(forHTTPHeaderField: "X-Next-Cursor")
                ?? http.value(forHTTPHeaderField: "x-next-cursor")
        } while cursor != nil && !(cursor?.isEmpty ?? true)

        guard !collected.isEmpty else { throw CoreAIDownloadError.emptyTree }
        return collected
    }

    private func stripPrefix(_ path: String) -> String {
        guard let pathPrefix, !pathPrefix.isEmpty else { return path }
        if path == pathPrefix {
            return pathPrefix.split(separator: "/").last.map(String.init) ?? path
        }
        let normalized = pathPrefix.hasSuffix("/") ? pathPrefix : pathPrefix + "/"
        if path.hasPrefix(normalized) {
            return String(path.dropFirst(normalized.count))
        }
        return path
    }

    private func downloadFile(_ file: RemoteFile, to destination: URL) async throws {
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let encodedPath = file.remotePath
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard let url = URL(
            string: "https://huggingface.co/\(repoID)/resolve/\(revision)/\(encodedPath)"
        ) else {
            throw URLError(.badURL)
        }

        let maxAttempts = 4
        var attempt = 0
        while true {
            do {
                let transfer = CoreAIForegroundTransfer()
                activeTransfer = transfer
                defer { if activeTransfer === transfer { activeTransfer = nil } }

                _ = try await transfer.download(from: url, to: destination) { [weak self] received, _ in
                    Task { @MainActor [weak self] in
                        guard let self else { return }
                        self.downloadedBytes = self.committedBytes + max(0, received)
                        self.recalculateProgress()
                        self.sampleSpeed()
                    }
                }

                let onDisk = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize)
                    .map(Int64.init) ?? max(file.size, 0)
                committedBytes += max(onDisk, 0)
                downloadedBytes = committedBytes
                sampleSpeed()
                return
            } catch {
                attempt += 1
                if Task.isCancelled || Self.isUserCancellation(error) {
                    throw CancellationError()
                }
                let ns = error as NSError
                if ns.domain == "HFDownload" || ns.domain == "CoreAIDownload",
                   ns.code == 401 || ns.code == 403 {
                    throw CoreAIDownloadError.tokenRequired
                }
                guard attempt < maxAttempts, Self.isRetryable(error) else { throw error }
                statusDetail = "Retry \(attempt)/\(maxAttempts - 1)…"
                RuntimeLogCenter.emit(
                    "\(repoID): \(file.remotePath) attempt \(attempt) failed; retrying — \(error.localizedDescription)",
                    level: .warning,
                    subsystem: "coreai"
                )
                downloadedBytes = committedBytes
                recalculateProgress()
                try? await Task.sleep(nanoseconds: Self.retryDelayNs(attempt: attempt, error: error))
            }
        }
    }

    private func fileAlreadyComplete(at url: URL, expectedSize: Int64) -> Bool {
        guard expectedSize > 0 else { return false }
        guard FileManager.default.fileExists(atPath: url.path) else { return false }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? -1
        return size == expectedSize
    }

    private func recalculateProgress() {
        if totalBytes > 0 {
            progress = min(0.999, Double(downloadedBytes) / Double(totalBytes))
            if filesDone >= filesTotal, filesTotal > 0 {
                progress = min(1, progress)
            }
        } else if filesTotal > 0 {
            progress = Double(filesDone) / Double(filesTotal)
        }
    }

    private func sampleSpeed() {
        let now = Date()
        let elapsed = now.timeIntervalSince(lastSampleAt)
        guard elapsed >= 0.35 else { return }
        let delta = Double(downloadedBytes - lastSampleBytes)
        speedMBps = max(0, (delta / elapsed) / 1_000_000)
        lastSampleBytes = downloadedBytes
        lastSampleAt = now
    }

    private func withRetries<T>(
        maxAttempts: Int = 3,
        _ operation: () async throws -> T
    ) async throws -> T {
        var attempt = 0
        while true {
            do {
                return try await operation()
            } catch {
                attempt += 1
                if Task.isCancelled || Self.isUserCancellation(error) {
                    throw CancellationError()
                }
                guard attempt < maxAttempts, Self.isRetryable(error) else { throw error }
                try? await Task.sleep(nanoseconds: Self.retryDelayNs(attempt: attempt, error: error))
            }
        }
    }

    private static func isRetryable(_ error: Error) -> Bool {
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut,
                 NSURLErrorNetworkConnectionLost,
                 NSURLErrorNotConnectedToInternet,
                 NSURLErrorCannotConnectToHost,
                 NSURLErrorDNSLookupFailed,
                 NSURLErrorResourceUnavailable,
                 NSURLErrorCannotFindHost,
                 NSURLErrorCancelled:
                // Cancelled by our stall watchdog / pause is handled separately;
                // network cancels from the session are still worth one retry.
                return ns.code != NSURLErrorCancelled
            default:
                return false
            }
        }
        if ns.domain == "HFDownload" || ns.domain == "CoreAIDownload" {
            return ns.code == 429 || (500...599).contains(ns.code) || ns.code == -6
        }
        return false
    }

    private static func isUserCancellation(_ error: Error) -> Bool {
        let ns = error as NSError
        return ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled
    }

    private static func retryDelayNs(attempt: Int, error: Error) -> UInt64 {
        let capSeconds: Double = 60
        if let retryAfter = (error as NSError).userInfo["Retry-After"] as? String,
           let seconds = Double(retryAfter), seconds > 0 {
            return UInt64(min(seconds, capSeconds) * 1_000_000_000)
        }
        let base = 1.5 * pow(2, Double(attempt - 1))
        let jittered = base * Double.random(in: 0.75...1.25)
        return UInt64(min(jittered, capSeconds) * 1_000_000_000)
    }
}

// MARK: - Foreground transfer

/// One-shot foreground download with live byte progress.
/// Avoids background URLSession, which frequently never resumes continuations
/// in ad-hoc / TrollStore sideloads.
final class CoreAIForegroundTransfer: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private var continuation: CheckedContinuation<URL, Error>?
    private var progressHandler: ((Int64, Int64) -> Void)?
    private var destination: URL?
    private var expectedSize: Int64 = 0
    private var task: URLSessionDownloadTask?
    private var session: URLSession?
    private var lastProgressAt = Date()
    private var watchdog: Task<Void, Never>?

    func download(
        from url: URL,
        to destination: URL,
        progress: @escaping (Int64, Int64) -> Void
    ) async throws -> URL {
        self.destination = destination
        self.progressHandler = progress

        let wifiOnly = await MainActor.run { AppSettings.shared.wifiOnlyDownloads }
        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 6 * 60 * 60
        cfg.waitsForConnectivity = true
        cfg.httpMaximumConnectionsPerHost = 4
        cfg.allowsCellularAccess = !wifiOnly
        cfg.allowsExpensiveNetworkAccess = !wifiOnly

        let session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
        self.session = session

        var request = URLRequest(url: url)
        request.timeoutInterval = 60
        request.setValue(
            "CoreAI-LAS/3.2.6 (iOS; Core AI pack download)",
            forHTTPHeaderField: "User-Agent"
        )
        if let host = url.host?.lowercased(),
           host == "huggingface.co" || host.hasSuffix(".huggingface.co") {
            HFTokenStore.authorize(&request)
        }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<URL, Error>) in
                self.continuation = cont
                let task = session.downloadTask(with: request)
                self.task = task
                self.lastProgressAt = Date()
                self.watchdog = Task { [weak self] in
                    await self?.watchForStall()
                }
                task.resume()
            }
        } onCancel: { [weak self] in
            self?.cancel()
        }
    }

    func cancel() {
        watchdog?.cancel()
        watchdog = nil
        task?.cancel()
        task = nil
        session?.invalidateAndCancel()
        session = nil
        if let cont = continuation {
            continuation = nil
            cont.resume(throwing: CancellationError())
        }
        progressHandler = nil
        destination = nil
    }

    private func watchForStall() async {
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: 20 * 1_000_000_000)
            guard continuation != nil else { return }
            if Date().timeIntervalSince(lastProgressAt) >= 90 {
                RuntimeLogCenter.emit(
                    "Core AI transfer stalled 90s; cancelling for retry",
                    level: .warning,
                    subsystem: "coreai"
                )
                let cont = continuation
                continuation = nil
                task?.cancel()
                session?.invalidateAndCancel()
                cont?.resume(throwing: NSError(
                    domain: "CoreAIDownload",
                    code: -6,
                    userInfo: [NSLocalizedDescriptionKey: "Download stalled — retrying."]
                ))
                return
            }
        }
    }

    private func finish(_ result: Result<URL, Error>) {
        watchdog?.cancel()
        watchdog = nil
        guard let cont = continuation else { return }
        continuation = nil
        progressHandler = nil
        session?.finishTasksAndInvalidate()
        session = nil
        task = nil
        cont.resume(with: result)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        lastProgressAt = Date()
        let expected = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : expectedSize
        progressHandler?(totalBytesWritten, expected)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let destination else {
            finish(.failure(URLError(.cannotCreateFile)))
            return
        }
        let response = downloadTask.response as? HTTPURLResponse
        let code = response?.statusCode ?? -1
        if !(200..<300 ~= code) {
            var userInfo: [String: Any] = [
                NSLocalizedDescriptionKey: "HTTP \(code) while downloading \(destination.lastPathComponent)"
            ]
            if let retryAfter = response?.value(forHTTPHeaderField: "Retry-After") {
                userInfo["Retry-After"] = retryAfter
            }
            finish(.failure(NSError(domain: "CoreAIDownload", code: code, userInfo: userInfo)))
            return
        }

        do {
            let fm = FileManager.default
            try fm.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if fm.fileExists(atPath: destination.path) {
                try fm.removeItem(at: destination)
            }
            // URLSession may delete `location` when this delegate returns —
            // move synchronously before returning.
            do {
                try fm.moveItem(at: location, to: destination)
            } catch {
                let tmp = destination.appendingPathExtension("tmp")
                try? fm.removeItem(at: tmp)
                try fm.copyItem(at: location, to: tmp)
                try fm.moveItem(at: tmp, to: destination)
            }
            FileManager.excludeFromBackup(destination)
            finish(.success(destination))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error else { return }
        finish(.failure(error))
    }
}

private struct HFTreeEntry: Decodable {
    let type: String
    let path: String
    let size: Int64?
    let lfs: LFS?

    struct LFS: Decodable {
        let size: Int64?
    }
}

enum CoreAIDownloadError: LocalizedError {
    case tokenRequired
    case httpStatus(Int)
    case emptyTree

    var errorDescription: String? {
        switch self {
        case .tokenRequired:
            return "Hugging Face returned 401/403. Add an access token for gated repos."
        case .httpStatus(let code):
            return "Hugging Face HTTP \(code)"
        case .emptyTree:
            return "No downloadable files found under the selected Hub path."
        }
    }
}

/// Active Core AI downloads observed by the Models UI.
@MainActor
final class CoreAIDownloadCenter: ObservableObject {
    static let shared = CoreAIDownloadCenter()

    @Published private(set) var downloads: [CoreAIHFDownloadManager] = []

    func existing(id: String) -> CoreAIHFDownloadManager? {
        downloads.first(where: { $0.id == id })
    }

    func manager(for model: CoreAIZooModel) -> CoreAIHFDownloadManager {
        if let existing = existing(id: model.id) { return existing }
        let created = CoreAIHFDownloadManager(model: model)
        downloads.insert(created, at: 0)
        objectWillChange.send()
        return created
    }

    func manager(
        repoID: String,
        pathPrefix: String? = nil,
        displayName: String? = nil
    ) -> CoreAIHFDownloadManager {
        let id = "hf:\(repoID)#\(pathPrefix ?? "")"
        if let existing = existing(id: id) { return existing }
        let created = CoreAIHFDownloadManager(
            id: id,
            repoID: repoID,
            pathPrefix: pathPrefix,
            displayName: displayName ?? repoID
        )
        downloads.insert(created, at: 0)
        objectWillChange.send()
        return created
    }

    func start(model: CoreAIZooModel) {
        manager(for: model).start()
    }

    func removeFinished() {
        downloads.removeAll {
            if case .ready = $0.state { return true }
            if case .idle = $0.state { return true }
            return false
        }
        objectWillChange.send()
    }
}
