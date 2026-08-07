import Foundation
import Network
import UIKit

// HTTPS server that implements the LocalCoderBridge iOS-side protocol.
//
// Endpoints:
//   POST /v1/pair  — exchange QR nonce for bearer token (no auth required)
//   GET  /v1/info  — device + model info (bearer auth)
//   POST /v1/infer — SSE-streamed inference (bearer auth)
//
// TLS is terminated by NWListener using the self-signed cert from BridgeIdentity.
// All connections are short-lived (Connection: close).
actor BridgeServer {

    private var listener: NWListener?

    /// Brute-force guard on `/v1/pair` — counts failed nonce attempts.
    private var failedPairAttempts: [Date] = []
    private var pairLockedUntil: Date?

    private static let maxFailedPairAttempts = 5
    private static let pairAttemptWindow: TimeInterval = 60
    private static let pairLockoutDuration: TimeInterval = 60

    static let port: NWEndpoint.Port = 8443

    // MARK: - Lifecycle

    func start(identity: SecIdentity) throws {
        guard listener == nil else { return }

        let tlsOpts = NWProtocolTLS.Options()
        let secID = sec_identity_create(identity)!
        sec_protocol_options_set_local_identity(tlsOpts.securityProtocolOptions, secID)
        sec_protocol_options_set_peer_authentication_required(tlsOpts.securityProtocolOptions, false)

        let tcpOpts = NWProtocolTCP.Options()
        tcpOpts.noDelay = true

        let params = NWParameters(tls: tlsOpts, tcp: tcpOpts)
        params.allowLocalEndpointReuse = true
        params.includePeerToPeer = true

        let l = try NWListener(using: params, on: Self.port)
        l.stateUpdateHandler = { [weak self] state in
            Task { await self?.onListenerState(state) }
        }
        l.newConnectionHandler = { [weak self] conn in
            Task { await self?.handle(conn) }
        }
        l.start(queue: .global(qos: .userInitiated))
        listener = l
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection

    private func handle(_ conn: NWConnection) async {
        conn.start(queue: .global(qos: .userInitiated))
        do {
            let req = try await readRequest(conn)
            await route(req, conn: conn)
        } catch {
            conn.cancel()
        }
    }

    // MARK: - HTTP Request Reader

    /// Hard ceiling on a single request. `/v1/infer` bodies are small
    /// JSON (prompt + a file's worth of text); nothing legitimate needs
    /// megabytes. Without this cap a peer who completed the TLS
    /// handshake (the server does not require a client cert) could send
    /// an endless stream with no header terminator or a huge
    /// Content-Length and drive the app to an OOM kill, since
    /// HTTPRequest(data:) returns nil until the whole body has arrived.
    private static let maxRequestBytes = 4 * 1024 * 1024  // 4 MiB

    /// Hard ceiling on how long a single request may take to ARRIVE in full.
    /// Without it, a peer who completed the TLS handshake can hold the
    /// connection open indefinitely — dribbling one byte at a time, or
    /// promising a Content-Length body that never comes — and the awaiting
    /// `conn.receive` continuation would never resume (a permanently
    /// suspended task per connection: a slowloris DoS). The size cap above
    /// bounds memory; this bounds time.
    private static let requestReadTimeout: TimeInterval = 15

    private func readRequest(_ conn: NWConnection) async throws -> HTTPRequest {
        var buffer = Data()
        let deadline = Date().addingTimeInterval(Self.requestReadTimeout)
        while true {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw BridgeServerError.requestTimedOut }
            let chunk = try await recv(conn, timeout: remaining)
            guard !chunk.isEmpty else { throw BridgeServerError.connectionClosed }
            buffer.append(chunk)
            guard buffer.count <= Self.maxRequestBytes else {
                throw BridgeServerError.requestTooLarge
            }
            if let req = HTTPRequest(data: buffer) { return req }
        }
    }

    private func recv(_ conn: NWConnection, timeout: TimeInterval) async throws -> Data {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Data, Error>) in
            // Resume exactly once — whichever fires first, the receive callback
            // or the timeout below.
            let box = ThrowingResumeOnce(cont)
            conn.receive(minimumIncompleteLength: 1, maximumLength: 65536) { data, _, _, error in
                if let error { box.resume(throwing: error) }
                else         { box.resume(returning: data ?? Data()) }
            }
            Task {
                try? await Task.sleep(nanoseconds: UInt64(max(0, timeout) * 1_000_000_000))
                if box.resume(throwing: BridgeServerError.requestTimedOut) {
                    // Cancel so the still-pending receive callback fires and
                    // its (now no-op) resume doesn't leak the connection.
                    conn.cancel()
                }
            }
        }
    }

    // MARK: - Router

    private func route(_ req: HTTPRequest, conn: NWConnection) async {
        switch (req.method, req.path) {
        case ("POST", "/v1/pair"):   await handlePair(req, conn: conn)
        case ("GET",  "/v1/info"):   await handleInfo(req, conn: conn)
        case ("POST", "/v1/infer"):  await handleInfer(req, conn: conn)
        default:                     await respond(conn, status: 404, body: "Not Found")
        }
    }

    // MARK: - /v1/pair

    private func handlePair(_ req: HTTPRequest, conn: NWConnection) async {
        if let locked = pairLockedUntil, locked > Date() {
            await respond(conn, status: 429, body: "Too many pairing attempts")
            return
        }
        guard let body = req.body,
              let pr = try? JSONDecoder().decode(PairRequestDTO.self, from: body) else {
            await respond(conn, status: 400, body: "Bad Request")
            return
        }
        let ok = await MainActor.run { BridgeManager.shared.verifyAndConsumeNonce(pr.nonce) }
        guard ok else {
            recordFailedPairAttempt()
            await respond(conn, status: 401, body: "Invalid nonce")
            return
        }
        failedPairAttempts.removeAll()
        pairLockedUntil = nil
        let token    = UUID().uuidString + UUID().uuidString
        let deviceId = await MainActor.run {
            UIDevice.current.identifierForVendor?.uuidString ?? UUID().uuidString
        }
        try? BridgePairingStore.shared.save(token: token, clientName: pr.clientName)
        await MainActor.run { BridgeManager.shared.onPaired(clientName: pr.clientName) }

        guard let data = try? JSONEncoder().encode(PairResponseDTO(bearerToken: token, deviceId: deviceId)) else {
            await respond(conn, status: 500, body: "Encode error"); return
        }
        await respond(conn, status: 200, contentType: "application/json", bodyData: data)
    }

    // MARK: - /v1/info

    private func handleInfo(_ req: HTTPRequest, conn: NWConnection) async {
        guard bearerValid(req) else { await respond(conn, status: 401, body: "Unauthorized"); return }
        let info = await MainActor.run { BridgeManager.shared.deviceInfoResponse() }
        guard let data = try? JSONEncoder().encode(info) else {
            await respond(conn, status: 500, body: "Encode error"); return
        }
        await respond(conn, status: 200, contentType: "application/json", bodyData: data)
    }

    // MARK: - /v1/infer (SSE streaming)

    private func handleInfer(_ req: HTTPRequest, conn: NWConnection) async {
        guard bearerValid(req) else { await respond(conn, status: 401, body: "Unauthorized"); return }
        guard let lease = RemoteInferenceGate.shared.acquire() else {
            await respond(conn, status: 503, body: "Model busy")
            return
        }
        defer { RemoteInferenceGate.shared.release(lease) }

        guard let body = req.body,
              let ir = try? JSONDecoder().decode(InferenceRequestDTO.self, from: body) else {
            await respond(conn, status: 400, body: "Bad Request"); return
        }
        let ready = await MainActor.run { CodingAssistantService.shared.state == .ready }
        guard ready else { await respond(conn, status: 503, body: "Model not loaded"); return }

        let sseHeaders = "HTTP/1.1 200 OK\r\n" +
            "Content-Type: text/event-stream\r\n" +
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        do { try await sendRaw(conn, Data(sseHeaders.utf8)) }
        catch { conn.cancel(); return }

        let messages = ir.toChatMessages()
        for await token in inferStream(messages: messages) {
            let esc = token
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\r", with: "\\r")
            let line = "data: {\"type\":\"token\",\"text\":\"\(esc)\"}\n\n"
            do { try await sendRaw(conn, Data(line.utf8)) }
            catch { break }
        }
        try? await sendRaw(conn, Data("event: done\n\n".utf8))
        conn.cancel()
    }

    // Wraps CodingAssistantService.generate() in an AsyncStream.
    private func inferStream(messages: [ChatMessage]) -> AsyncStream<String> {
        AsyncStream { continuation in
            Task { @MainActor in
                CodingAssistantService.shared.generate(
                    messages: messages,
                    onToken:    { continuation.yield($0) },
                    onComplete: { _ in continuation.finish() }
                )
            }
            // When the SSE peer disconnects mid-stream, handleInfer's send
            // throws and breaks the consuming loop — but generate() keeps
            // producing tokens into a now-unconsumed stream and pins the
            // single-flight slot until it finishes on its own. Terminating
            // the stream (consumer stops iterating, or finish() on natural
            // completion) now cancels the underlying generation so the model
            // stops immediately and isGenerating is released promptly.
            continuation.onTermination = { termination in
                // Natural completion is not cancellation. An unconditional
                // stop here can race the next request and cancel its model
                // generation when the previous stream is being torn down.
                guard case .cancelled = termination else { return }
                Task { @MainActor in
                    CodingAssistantService.shared.stopGeneration()
                }
            }
        }
    }

    // MARK: - HTTP Write Helpers

    private func respond(_ conn: NWConnection, status: Int, body: String) async {
        await respond(conn, status: status, contentType: "text/plain", bodyData: Data(body.utf8))
    }

    private func respond(_ conn: NWConnection, status: Int, contentType: String = "text/plain", bodyData: Data) async {
        let statusLine = httpStatusLine(status)
        let header = "HTTP/1.1 \(statusLine)\r\n" +
            "Content-Type: \(contentType)\r\n" +
            "Content-Length: \(bodyData.count)\r\n" +
            "Connection: close\r\n\r\n"
        var payload = Data(header.utf8)
        payload.append(bodyData)
        try? await sendRaw(conn, payload)
        conn.cancel()
    }

    private func sendRaw(_ conn: NWConnection, _ data: Data) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            conn.send(content: data, completion: .contentProcessed { error in
                if let error { cont.resume(throwing: error) }
                else         { cont.resume(returning: ()) }
            })
        }
    }

    // MARK: - Auth

    private func recordFailedPairAttempt() {
        let cutoff = Date().addingTimeInterval(-Self.pairAttemptWindow)
        failedPairAttempts.removeAll { $0 < cutoff }
        failedPairAttempts.append(Date())
        if failedPairAttempts.count >= Self.maxFailedPairAttempts {
            pairLockedUntil = Date().addingTimeInterval(Self.pairLockoutDuration)
            failedPairAttempts.removeAll()
        }
    }

    private func bearerValid(_ req: HTTPRequest) -> Bool {
        guard let auth = req.headers["authorization"],
              auth.hasPrefix("Bearer ") else { return false }
        return BridgePairingStore.shared.isValid(token: String(auth.dropFirst(7)))
    }

    private func httpStatusLine(_ code: Int) -> String {
        switch code {
        case 200: return "200 OK"
        case 400: return "400 Bad Request"
        case 401: return "401 Unauthorized"
        case 404: return "404 Not Found"
        case 429: return "429 Too Many Requests"
        case 503: return "503 Service Unavailable"
        default:  return "500 Internal Server Error"
        }
    }

    // MARK: - Listener State

    private func onListenerState(_ state: NWListener.State) {
        if case .failed(let err) = state {
            print("[BridgeServer] Listener failed: \(err)")
            listener?.cancel()
            listener = nil
        }
    }

    private enum BridgeServerError: Error {
        case connectionClosed
        case requestTooLarge
        case requestTimedOut
    }
}

// MARK: - Throwing continuation resume-once guard

/// Resumes a throwing `CheckedContinuation<Data, Error>` exactly once, no
/// matter how many callers race (the receive callback vs. the read timeout).
/// `resume` returns `true` only for the call that actually performed it.
private final class ThrowingResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var cont: CheckedContinuation<Data, Error>?

    init(_ cont: CheckedContinuation<Data, Error>) { self.cont = cont }

    @discardableResult
    func resume(returning value: Data) -> Bool {
        lock.lock(); let c = cont; cont = nil; lock.unlock()
        c?.resume(returning: value); return c != nil
    }

    @discardableResult
    func resume(throwing error: Error) -> Bool {
        lock.lock(); let c = cont; cont = nil; lock.unlock()
        c?.resume(throwing: error); return c != nil
    }
}

// MARK: - Minimal HTTP/1.1 Request Parser

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data?

    init?(data: Data) {
        guard let str = String(data: data, encoding: .utf8),
              let sep = str.range(of: "\r\n\r\n") else { return nil }

        let headerSection = String(str[str.startIndex..<sep.lowerBound])
        var lines = headerSection.components(separatedBy: "\r\n")
        guard !lines.isEmpty else { return nil }

        let requestLine = lines.removeFirst()
        let parts = requestLine.components(separatedBy: " ")
        guard parts.count >= 2 else { return nil }
        method = parts[0]
        path   = String(parts[1].split(separator: "?", maxSplits: 1).first ?? Substring(parts[1]))

        var hdrs: [String: String] = [:]
        for line in lines {
            if let colonRange = line.range(of: ": ") {
                let key = String(line[line.startIndex..<colonRange.lowerBound]).lowercased()
                let val = String(line[colonRange.upperBound...])
                hdrs[key] = val
            }
        }
        headers = hdrs

        let contentLength = Int(hdrs["content-length"] ?? "0") ?? 0
        let headerByteCount = headerSection.utf8.count + 4   // 4 = "\r\n\r\n"
        if contentLength > 0 {
            guard data.count >= headerByteCount + contentLength else { return nil }
            body = data[headerByteCount..<(headerByteCount + contentLength)]
        } else {
            body = nil
        }
    }
}

// MARK: - Wire DTOs

private struct PairRequestDTO: Codable {
    let clientName: String
    let nonce: String
}

private struct PairResponseDTO: Codable {
    let bearerToken: String
    let deviceId: String
}

struct DeviceInfoResponseDTO: Codable {
    let deviceId: String
    let deviceName: String
    let modelId: String
    let modelDisplayName: String
    /// Compatibility names consumed by the current Mac bridge.
    let modelName: String
    let modelLoaded: Bool
    let busy: Bool
    let version: String
    let contextWindow: Int
}

struct InferenceRequestDTO: Codable {
    let requestID: UUID
    let filePath: String?
    let language: String?
    let fileContent: String?
    let selection: String?
    let selectionStartLine: Int?
    let selectionEndLine: Int?
    let instruction: String
    let promptPreset: String
    let conversationId: String?
    let systemPrompt: String?

    // Decode redactionReport loosely — it was already applied on the Mac; we just ignore it.
    private enum CodingKeys: String, CodingKey {
        case requestID, filePath, language, fileContent, selection
        case selectionStartLine, selectionEndLine, instruction, promptPreset
        case conversationId, systemPrompt
    }

    func toChatMessages() -> [ChatMessage] {
        var msgs: [ChatMessage] = []
        msgs.append(ChatMessage(
            role: .system,
            content: systemPrompt ?? CodingAssistantService.systemPrompt
        ))
        var user = ""
        if let fp = filePath, let content = fileContent {
            user += "File: `\(fp)`\n\n```\(language ?? "")\n\(content)\n```\n\n"
        }
        if let sel = selection {
            let range = [
                selectionStartLine.map { "L\($0)" },
                selectionEndLine.map   { "-L\($0)" }
            ].compactMap { $0 }.joined()
            user += "Selection\(range.isEmpty ? "" : " \(range)"):\n```\n\(sel)\n```\n\n"
        }
        user += instruction
        msgs.append(ChatMessage(role: .user, content: user))
        return msgs
    }
}
