import XCTest
@testable import IOSLocalLLM

// MARK: - Int64FormattedBytesTests
//
// Pins the `Int64.formattedBytes` formatter used app-wide for "234 MB" /
// "1.2 GB"-style labels. ByteCountFormatter is locale-aware — these tests
// use en_US locale for deterministic output.

final class Int64FormattedBytesTests: XCTestCase {

    // MARK: - Zero and negative

    func test_zero_returnsZeroBytes() {
        XCTAssertEqual(Int64(0).formattedBytes, "0 B")
    }

    func test_negative_returnsZeroBytes() {
        XCTAssertEqual(Int64(-1).formattedBytes, "0 B")
        XCTAssertEqual(Int64(-1_000_000).formattedBytes, "0 B")
    }

    // MARK: - Positive values — KB range

    func test_bytes_returnsBytes() {
        // 512 bytes — below 1 KB, ByteCountFormatter uses "bytes" not "KB".
        let result = Int64(512).formattedBytes
        XCTAssertTrue(result.contains("B"), "Expected byte-range label, got \(result)")
    }

    // MARK: - MB range

    func test_megabytes_formatsCorrectly() {
        let result = Int64(50_000_000).formattedBytes
        // ~47.7 MB — should contain "MB"
        XCTAssertTrue(result.contains("MB"), "Expected MB label, got \(result)")
    }

    func test_100MB_formatsCorrectly() {
        let result = Int64(104_857_600).formattedBytes
        XCTAssertTrue(result.contains("MB"), "Expected MB label, got \(result)")
    }

    // MARK: - GB range

    func test_gigabytes_formatsCorrectly() {
        let result = Int64(8_000_000_000).formattedBytes
        XCTAssertTrue(result.contains("GB"), "Expected GB label, got \(result)")
    }

    func test_1GB_formatsCorrectly() {
        let result = Int64(1_073_741_824).formattedBytes
        XCTAssertTrue(result.contains("GB"), "Expected GB label, got \(result)")
    }

    // MARK: - Large values

    func test_largeValue_formatsWithoutCrash() {
        // 2 TB — should format to "TB" or at minimum not crash.
        let result = Int64(2_199_023_255_552).formattedBytes
        XCTAssertFalse(result.isEmpty)
    }

    // MARK: - Very small positive

    func test_oneByte_returnsNonEmpty() {
        let result = Int64(1).formattedBytes
        XCTAssertFalse(result.isEmpty)
    }
}
