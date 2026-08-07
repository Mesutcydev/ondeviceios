import XCTest
@testable import IOSLocalLLM

final class BridgePairingStoreTests: XCTestCase {

    override func tearDown() {
        BridgePairingStore.shared.deleteAll()
        super.tearDown()
    }

    func test_isValid_returnsFalseForUnknownToken() {
        XCTAssertFalse(BridgePairingStore.shared.isValid(token: "not-a-real-token"))
    }

    func test_saveAndValidate_roundTrip() throws {
        let token = UUID().uuidString + UUID().uuidString
        try BridgePairingStore.shared.save(token: token, clientName: "Test Mac")
        XCTAssertTrue(BridgePairingStore.shared.isValid(token: token))
        XCTAssertEqual(BridgePairingStore.shared.clients().count, 1)
    }

    func test_deleteAll_removesAllClients() throws {
        try BridgePairingStore.shared.save(token: "token-a", clientName: "Mac A")
        try BridgePairingStore.shared.save(token: "token-b", clientName: "Mac B")
        XCTAssertEqual(BridgePairingStore.shared.clients().count, 2)

        BridgePairingStore.shared.deleteAll()

        XCTAssertTrue(BridgePairingStore.shared.clients().isEmpty)
        XCTAssertFalse(BridgePairingStore.shared.isValid(token: "token-a"))
    }
}
