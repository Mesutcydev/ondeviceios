import XCTest
@testable import IOSLocalLLM

// MARK: - DiagnosticsTests
//
// Exercises the debug-system core: level ordering, the minimum-level filter,
// the ring buffer, breadcrumb level, category filtering, and that the export
// bundle carries the environment header.

final class DiagnosticsTests: XCTestCase {

    override func setUp() {
        super.setUp()
        Diagnostics.shared.clear()
        Diagnostics.shared.minimumLevel = .debug   // capture everything for the test
    }

    override func tearDown() {
        Diagnostics.shared.clear()
        Diagnostics.shared.minimumLevel = .info
        super.tearDown()
    }

    func test_levelOrdering() {
        XCTAssertLessThan(DiagLevel.debug, DiagLevel.info)
        XCTAssertLessThan(DiagLevel.warning, DiagLevel.error)
        XCTAssertLessThan(DiagLevel.error, DiagLevel.fault)
    }

    func test_logsAreRecordedAndRetrievable() {
        Diagnostics.shared.info("hello world", category: "test")
        let entries = Diagnostics.shared.recentEntries(category: "test")
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.message, "hello world")
        XCTAssertEqual(entries.first?.level, .info)
    }

    func test_minimumLevelFiltersBelowThreshold() {
        Diagnostics.shared.minimumLevel = .warning
        Diagnostics.shared.info("dropped", category: "test")
        Diagnostics.shared.error("kept", category: "test")
        let entries = Diagnostics.shared.recentEntries(minLevel: .debug, category: "test")
        XCTAssertEqual(entries.map(\.message), ["kept"])
    }

    func test_recentEntriesMinLevelFilter() {
        Diagnostics.shared.debug("d", category: "test")
        Diagnostics.shared.error("e", category: "test")
        let errorsOnly = Diagnostics.shared.recentEntries(minLevel: .error, category: "test")
        XCTAssertEqual(errorsOnly.map(\.message), ["e"])
    }

    func test_categoryFilter() {
        Diagnostics.shared.info("a", category: "alpha")
        Diagnostics.shared.info("b", category: "beta")
        XCTAssertEqual(Diagnostics.shared.recentEntries(category: "alpha").map(\.message), ["a"])
        XCTAssertEqual(Diagnostics.shared.recentEntries(category: "beta").map(\.message), ["b"])
    }

    func test_breadcrumbIsNoticeLevel() {
        Diagnostics.shared.breadcrumb("crumb", category: "test")
        let entry = Diagnostics.shared.recentEntries(category: "test").first
        XCTAssertEqual(entry?.level, .notice)
    }

    func test_exportContainsHeaderAndEntry() {
        Diagnostics.shared.error("boom", category: "test")
        let text = Diagnostics.shared.exportText()
        XCTAssertTrue(text.contains("iOS Local LLM diagnostics"))
        XCTAssertTrue(text.contains("boom"))
        XCTAssertTrue(text.contains("RECENT LOG (1 entries)"))
    }

    func test_entryLineFormatIncludesLevelAndCategory() {
        Diagnostics.shared.warning("watch out", category: "zone")
        let line = Diagnostics.shared.recentEntries(category: "zone").first?.line ?? ""
        XCTAssertTrue(line.contains("[WARN]"))
        XCTAssertTrue(line.contains("(zone)"))
        XCTAssertTrue(line.contains("watch out"))
    }
}
