import XCTest
@testable import IOSLocalLLM

final class AppBridgeSecurityTests: XCTestCase {

    func test_isSafeShareFilename_rejectsTraversal() {
        XCTAssertFalse(AppBridge.isSafeShareFilename("../secrets.txt"))
        XCTAssertFalse(AppBridge.isSafeShareFilename(".."))
        XCTAssertFalse(AppBridge.isSafeShareFilename("foo/bar.jpg"))
        XCTAssertFalse(AppBridge.isSafeShareFilename("..\\windows"))
        XCTAssertFalse(AppBridge.isSafeShareFilename(".hidden"))
    }

    func test_isSafeShareFilename_acceptsSingleComponent() {
        XCTAssertTrue(AppBridge.isSafeShareFilename("photo.jpg"))
        XCTAssertTrue(AppBridge.isSafeShareFilename("share-2026-07-09.txt"))
    }
}
