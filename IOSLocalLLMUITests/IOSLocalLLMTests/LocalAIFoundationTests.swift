import XCTest
@testable import IOSLocalLLM

final class LocalAIFoundationTests: XCTestCase {
    func testUTF8DecoderHoldsIncompleteMultibyteTail() {
        let decoder = UTF8StreamDecoder()

        XCTAssertEqual(decoder.append([0xC3]), "")
        XCTAssertEqual(decoder.append([0xA9]), "é")
        XCTAssertEqual(decoder.finish(), "")
    }

    func testStreamingCoordinatorSendsFirstTokenAndCoalescedPartial() async throws {
        let events = LockedEvents()
        let coordinator = StreamingResponseCoordinator(uiBufferIntervalMs: 10) { event in
            events.append(event)
        }

        await coordinator.start()
        await coordinator.appendToken("Hel")
        await coordinator.appendToken("lo")
        try await Task.sleep(nanoseconds: 30_000_000)
        await coordinator.finish(tokensPerSecond: 12, inputTokens: 2, outputTokens: 1)

        let snapshot = events.snapshot()
        XCTAssertEqual(snapshot.first, .started)
        XCTAssertTrue(snapshot.contains(.token("Hel")))
        XCTAssertTrue(snapshot.contains(.partialText("Hello")))
        XCTAssertTrue(snapshot.contains(.usage(tokensPerSecond: 12, inputTokens: 2, outputTokens: 1)))
        XCTAssertEqual(snapshot.last, .completed)
    }

    func testPIIRedactorRemovesObviousSensitiveStrings() {
        let input = """
        Email me at person@example.com with Bearer abcdefghijklmnopqrstuvwxyz123.
        Local callback: http://127.0.0.1:8080/private.
        """

        let redacted = PIIRedactor.redact(input)

        XCTAssertFalse(redacted.contains("person@example.com"))
        XCTAssertFalse(redacted.contains("abcdefghijklmnopqrstuvwxyz123"))
        XCTAssertFalse(redacted.contains("127.0.0.1"))
        XCTAssertTrue(redacted.contains("[REDACTED_EMAIL]"))
        XCTAssertTrue(redacted.contains("[REDACTED_BEARER_TOKEN]"))
        XCTAssertTrue(redacted.contains("[REDACTED_PRIVATE_URL]"))
    }

    func testGGUFPairRequiresMagicBytes() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data("NOPE".utf8).write(to: dir.appendingPathComponent("model.gguf"))
        try Data("NOPE".utf8).write(to: dir.appendingPathComponent("mmproj-model.gguf"))
        XCTAssertFalse(LocalModelFileValidator.hasCompleteGGUFVLMPair(in: dir))
        XCTAssertFalse(ModelCacheProbe.isUsableModelDirectory(dir))

        try Data([0x47, 0x47, 0x55, 0x46, 0x03, 0x00]).write(to: dir.appendingPathComponent("model.gguf"))
        try Data([0x47, 0x47, 0x55, 0x46, 0x03, 0x00]).write(to: dir.appendingPathComponent("mmproj-model.gguf"))
        XCTAssertTrue(LocalModelFileValidator.hasCompleteGGUFVLMPair(in: dir))
        XCTAssertTrue(ModelCacheProbe.isUsableModelDirectory(dir))
    }

    func testShardedSafetensorsRequireEveryShard() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        let index = #"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#
        try Data(index.utf8).write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try Data([1, 2, 3]).write(to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))

        XCTAssertFalse(ModelCacheProbe.isUsableModelDirectory(dir))

        try Data([4, 5, 6]).write(to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        XCTAssertTrue(ModelCacheProbe.isUsableModelDirectory(dir))
    }

    func testPendingIndexQueueProcessesOnlyWhenPolicyAllows() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let storeURL = dir.appendingPathComponent("queue.json")
        let docURL = dir.appendingPathComponent("doc.txt")
        try Data("hello".utf8).write(to: docURL)

        let processed = LockedStrings()
        let blocked = PendingIndexQueue(
            storeURL: storeURL,
            policyProvider: {
                PendingIndexPolicySnapshot(
                    canProcessBulkIngestion: false,
                    blockedReason: "testing"
                )
            },
            processor: { job in
                processed.append(job.documentID)
            }
        )
        try await blocked.enqueue(documentURL: docURL, documentID: "doc-1")
        let blockedCount = try await blocked.processNextBatch(limit: 1)
        XCTAssertEqual(blockedCount, 0)
        XCTAssertEqual(processed.snapshot(), [])

        let allowed = PendingIndexQueue(
            storeURL: storeURL,
            policyProvider: { .allowed },
            processor: { job in
                processed.append(job.documentID)
            }
        )
        let allowedCount = try await allowed.processNextBatch(limit: 1)
        XCTAssertEqual(allowedCount, 1)
        XCTAssertEqual(processed.snapshot(), ["doc-1"])
        let pending = await allowed.pendingJobs()
        XCTAssertTrue(pending.isEmpty)
    }

    private func makeTempDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LocalAIFoundationTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

private final class LockedEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [TokenEvent] = []

    func append(_ event: TokenEvent) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    func snapshot() -> [TokenEvent] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

private final class LockedStrings: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []

    func append(_ value: String) {
        lock.lock()
        values.append(value)
        lock.unlock()
    }

    func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}
