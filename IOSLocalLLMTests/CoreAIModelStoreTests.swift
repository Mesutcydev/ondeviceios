import XCTest
@testable import CoreAIOnDeviceLAS

final class CoreAIModelStoreTests: XCTestCase {
    private var tempRoot: URL!

    override func setUpWithError() throws {
        tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("CoreAIModelStoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempRoot {
            try? FileManager.default.removeItem(at: tempRoot)
        }
        // Reset store state between tests by clearing Application Support install.
        try? CoreAIModelStore.shared.removeModel()
    }

    func testAimodelIsRequiredForDownloadedAsset() {
        XCTAssertEqual(CoreAIModelStore.defaultModelID, "coreai-qwen3-0.6b")
    }

    func testDownloadRejectsNonAimodelExtension() {
        XCTAssertEqual(
            CoreAIModelStoreError.invalidExtension.errorDescription,
            "Core AI models must use the .aimodel format."
        )
    }

    func testImportResolvesNestedResourcesFolder() throws {
        let wrapper = tempRoot.appendingPathComponent("hub-checkout", isDirectory: true)
        let resources = wrapper.appendingPathComponent("resources", isDirectory: true)
        try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
        try makeMinimalPack(at: resources, name: "Nested.aimodel")

        try CoreAIModelStore.shared.importModel(
            from: wrapper,
            preferredID: "test-nested",
            preferredDisplayName: "Nested Pack"
        )

        guard case .ready(_, let manifest) = CoreAIModelStore.shared.state else {
            return XCTFail("Expected ready state, got \(CoreAIModelStore.shared.state)")
        }
        XCTAssertEqual(manifest.id, "test-nested")
        XCTAssertEqual(manifest.displayName, "Nested Pack")
        XCTAssertNotNil(CoreAIModelStore.shared.modelResourcesURL)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: CoreAIModelStore.shared.modelResourcesURL!
                    .appendingPathComponent("metadata.json").path
            )
        )
    }

    func testImportResolvesIOSSubtree() throws {
        let wrapper = tempRoot.appendingPathComponent("official", isDirectory: true)
        let ios = wrapper.appendingPathComponent("ios", isDirectory: true)
        try FileManager.default.createDirectory(at: ios, withIntermediateDirectories: true)
        try makeMinimalPack(at: ios, name: "Official.aimodel")

        try CoreAIModelStore.shared.importModel(
            from: wrapper,
            preferredID: "test-ios",
            preferredDisplayName: "iOS Pack"
        )

        guard case .ready = CoreAIModelStore.shared.state else {
            return XCTFail("Expected ready state, got \(CoreAIModelStore.shared.state)")
        }
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: CoreAIModelStore.shared.modelResourcesURL!
                    .appendingPathComponent("Official.aimodel").path
            )
        )
    }

    func testPackageContentTypesDoNotIncludeFolderOnlyMix() {
        let packageTypes = LocalModelDocumentPickerSession.contentTypes(for: .package)
        let folderTypes = LocalModelDocumentPickerSession.contentTypes(for: .folder)
        XCTAssertTrue(packageTypes.contains(.package))
        XCTAssertFalse(packageTypes.contains(.folder))
        XCTAssertTrue(folderTypes.contains(.folder))
        XCTAssertFalse(folderTypes.contains(.package))
    }

    // MARK: - Helpers

    private func makeMinimalPack(at root: URL, name: String) throws {
        let aimodel = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: aimodel, withIntermediateDirectories: true)
        try Data([0x01, 0x02, 0x03]).write(to: aimodel.appendingPathComponent("main.mlirb"))
        let metadata = """
        {"schema_version":"0.2","kind":"language","name":"test"}
        """
        try metadata.write(
            to: root.appendingPathComponent("metadata.json"),
            atomically: true,
            encoding: .utf8
        )
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("tokenizer", isDirectory: true),
            withIntermediateDirectories: true
        )
        try "{}".write(
            to: root.appendingPathComponent("tokenizer/tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
    }
}
