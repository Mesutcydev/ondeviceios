import XCTest
@testable import IOSLocalLLM

// Focused regression coverage for the three services with the largest
// persistence, download, and inference blast radius.

final class ModelDownloadCenterServiceTests: XCTestCase {
    @MainActor
    func testDownloadableVisionModelAppearsInAssistantAndVisionCategories() {
        let model = DownloadableModel(
            id: "test-vlm",
            displayName: "Test VLM",
            subtitle: "Qwen/Test-VL",
            sizeLabel: "1 GB",
            category: .assistant,
            repoID: "Qwen/Test-VL",
            capabilities: [.vision]
        )

        XCTAssertEqual(model.sourceRepoID, "Qwen/Test-VL")
        XCTAssertEqual(model.vendor, .qwen)
        XCTAssertTrue(model.supportsCategory(.assistant))
        XCTAssertTrue(model.supportsCategory(.vlm))
        XCTAssertFalse(model.supportsCategory(.voice))
    }

    @MainActor
    func testModelWithoutDownloaderHasSafeIdleState() {
        let model = DownloadableModel(
            id: "metadata-only",
            displayName: "Metadata Only",
            subtitle: "example/model",
            sizeLabel: "Unknown",
            category: .assistant
        )

        XCTAssertEqual(model.state, .idle)
        XCTAssertEqual(model.progress, 0)
        XCTAssertFalse(model.isReady)
    }
}

final class CodingAssistantServicePolicyTests: XCTestCase {
    func testStaleMLXGenerationCannotPublishOverCurrentGeneration() {
        let current = UUID()
        let stale = UUID()

        XCTAssertTrue(AssistantGenerationOwnership.isCurrent(
            activeID: current,
            completingID: current
        ))
        XCTAssertFalse(AssistantGenerationOwnership.isCurrent(
            activeID: current,
            completingID: stale
        ))
        XCTAssertFalse(AssistantGenerationOwnership.isCurrent(
            activeID: nil,
            completingID: stale
        ))
    }

    func testBackgroundCleanupNeverWaitsForMetalOrUnloadsItsRuntime() {
        let actions = LifecycleBackgroundCleanupPlan.referenceCompatible.actions

        XCTAssertEqual(actions, [.cancelQueuedMLX])
        XCTAssertFalse(actions.contains(.awaitNativeGeneration))
        XCTAssertFalse(actions.contains(.unloadResidentRuntimes))
    }

    func testTransitionOwnedUnloadNeverDrainsItsOwnTask() {
        XCTAssertTrue(AssistantUnloadDrainPolicy.transitionOwned.loadTask)
        XCTAssertFalse(AssistantUnloadDrainPolicy.transitionOwned.transitionTask)
    }

    func testLoadOwnedUnloadCannotDrainEitherPossibleOwner() {
        XCTAssertFalse(AssistantUnloadDrainPolicy.loadOwned.loadTask)
        XCTAssertFalse(AssistantUnloadDrainPolicy.loadOwned.transitionTask)
    }

    func testExternalUnloadDrainsAllServiceOwnedWork() {
        XCTAssertTrue(AssistantUnloadDrainPolicy.external.loadTask)
        XCTAssertTrue(AssistantUnloadDrainPolicy.external.transitionTask)
    }

    func testGGUFBackendComputeFailureHasActionableDescription() {
        XCTAssertEqual(
            LlamaCppError.decodeFailed(-3).errorDescription,
            "llama.cpp backend compute failed while decoding (code -3)"
        )
    }

    func testStorageBackedGGUFPolicyDisablesGPULayersAndBoundsHeadroom() {
        let policy = GGUFLoadPolicy.resolve(
            fileBytes: 8 * 1_024 * 1_024 * 1_024,
            pagingEnabled: true
        )

        XCTAssertTrue(policy.storageBacked)
        XCTAssertEqual(policy.gpuLayers, 0)
        XCTAssertEqual(
            policy.minimumAvailableBytes,
            GGUFLoadPolicy.storageBackedHeadroom
        )
    }

    func testNormalLargeGGUFPolicyKeepsAcceleratedLayerBudget() {
        let policy = GGUFLoadPolicy.resolve(
            fileBytes: 4 * 1_024 * 1_024 * 1_024,
            pagingEnabled: false
        )

        XCTAssertFalse(policy.storageBacked)
        XCTAssertEqual(policy.gpuLayers, 12)
        XCTAssertGreaterThan(policy.minimumAvailableBytes, 4_000_000_000)
    }

    func testEntitledGGUFReloadAllowsOnlyNarrowReclamationGap() {
        let policy = GGUFLoadPolicy.resolve(
            fileBytes: 7_505_194_272,
            pagingEnabled: false
        )

        XCTAssertTrue(policy.canAdmit(
            availableBytes: policy.minimumAvailableBytes - 64_000_000,
            hasIncreasedMemoryEntitlement: true
        ))
        XCTAssertFalse(policy.canAdmit(
            availableBytes: policy.minimumAvailableBytes - 64_000_000,
            hasIncreasedMemoryEntitlement: false
        ))
        XCTAssertFalse(policy.canAdmit(
            availableBytes: policy.minimumAvailableBytes - 256_000_000,
            hasIncreasedMemoryEntitlement: true
        ))
    }

    func testStandardMLXPolicyUsesNativeAllocatorDefaults() {
        let policy = MLXLowMemoryPolicy.resolve(
            enabled: false,
            physicalMemoryBytes: 8_000_000_000,
            processCeilingBytes: 6_000_000_000
        )

        XCTAssertNil(policy.memoryLimitBytes)
        XCTAssertNil(policy.cacheLimitBytes)
    }
}

final class CloudSyncServiceTests: XCTestCase {
    func testPartialPushErrorReportsFailedAndTotalCounts() {
        let error = CloudSyncError.partialPush(failed: 2, total: 5)
        XCTAssertEqual(
            error.errorDescription,
            "iCloud sync incomplete — 2 of 5 conversations failed to upload."
        )
    }

    func testDeletionTombstoneRoundTrip() throws {
        let id = UUID()
        CloudSyncTombstones.clear(id.uuidString)
        defer { CloudSyncTombstones.clear(id.uuidString) }

        CloudSyncTombstones.mark(id)
        let deletionDate = try XCTUnwrap(
            CloudSyncTombstones.deletionDate(id.uuidString)
        )

        XCTAssertLessThan(abs(deletionDate.timeIntervalSinceNow), 2)
        CloudSyncTombstones.clear(id.uuidString)
        XCTAssertNil(CloudSyncTombstones.deletionDate(id.uuidString))
    }
}
