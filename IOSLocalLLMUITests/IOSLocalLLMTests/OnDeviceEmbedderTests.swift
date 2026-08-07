import XCTest
@testable import IOSLocalLLM

// MARK: - OnDeviceEmbedderTests
//
// Math + chunking guarantees the RAG pipeline relies on. Embedding-model
// presence is environment-dependent (NaturalLanguage assets), so those checks
// are skipped when no model is available rather than failing the suite.

final class OnDeviceEmbedderTests: XCTestCase {

    // MARK: Cosine (pure math — always runs)

    func test_cosineOfIdenticalUnitVectorsIsOne() {
        let v: [Float] = [1, 0, 0]
        XCTAssertEqual(OnDeviceEmbedder.cosine(v, v), 1, accuracy: 1e-6)
    }

    func test_cosineOfOrthogonalVectorsIsZero() {
        XCTAssertEqual(OnDeviceEmbedder.cosine([1, 0, 0], [0, 1, 0]), 0, accuracy: 1e-6)
    }

    func test_cosineOfOppositeVectorsIsMinusOne() {
        XCTAssertEqual(OnDeviceEmbedder.cosine([1, 0, 0], [-1, 0, 0]), -1, accuracy: 1e-6)
    }

    func test_cosineDimensionMismatchIsZero() {
        XCTAssertEqual(OnDeviceEmbedder.cosine([1, 0], [1, 0, 0]), 0)
    }

    // MARK: Embedding (skipped when no model)

    func test_embeddingIsNormalizedAndDeterministic() throws {
        let embedder = OnDeviceEmbedder()
        try XCTSkipUnless(embedder.isAvailable, "No on-device embedding model for this environment")
        guard let v1 = embedder.embed("the quick brown fox"),
              let v2 = embedder.embed("the quick brown fox") else {
            return XCTFail("embed returned nil despite availability")
        }
        XCTAssertEqual(v1, v2, "embedding must be deterministic")
        let norm = sqrt(v1.reduce(0) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1, accuracy: 1e-3, "embedding must be L2-normalized")
    }

    func test_relatedTextScoresHigherThanUnrelated() throws {
        let embedder = OnDeviceEmbedder()
        try XCTSkipUnless(embedder.isAvailable, "No on-device embedding model for this environment")
        guard let q = embedder.embed("how do I cook pasta"),
              let related = embedder.embed("boil the pasta in salted water"),
              let unrelated = embedder.embed("the stock market fell today") else {
            return XCTFail("embed returned nil")
        }
        XCTAssertGreaterThan(OnDeviceEmbedder.cosine(q, related),
                             OnDeviceEmbedder.cosine(q, unrelated))
    }
}

// MARK: - KnowledgeBaseChunkingTests

final class KnowledgeBaseChunkingTests: XCTestCase {

    func test_shortTextIsSingleChunk() {
        let chunks = KnowledgeBaseService.chunk("hello world", chunkChars: 1600, overlapChars: 200, maxChunks: 400)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first, "hello world")
    }

    func test_emptyTextProducesNoChunks() {
        XCTAssertTrue(KnowledgeBaseService.chunk("   ", chunkChars: 100, overlapChars: 10, maxChunks: 10).isEmpty)
    }

    func test_longTextSplitsIntoMultipleChunks() {
        let sentence = "This is a sentence. "
        let text = String(repeating: sentence, count: 200)   // ~4000 chars
        let chunks = KnowledgeBaseService.chunk(text, chunkChars: 400, overlapChars: 50, maxChunks: 400)
        XCTAssertGreaterThan(chunks.count, 1)
        // Each chunk should be roughly within the window (allow snap slack).
        for c in chunks { XCTAssertLessThanOrEqual(c.count, 600) }
    }

    func test_maxChunksCapRespected() {
        let text = String(repeating: "word ", count: 10_000)
        let chunks = KnowledgeBaseService.chunk(text, chunkChars: 100, overlapChars: 10, maxChunks: 5)
        XCTAssertLessThanOrEqual(chunks.count, 5)
    }
}
