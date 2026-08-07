// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Synchronization
import Testing

@testable import CoreAILanguageModels

@Suite("PipelineGate")
struct PipelineGateTests {
    // MARK: - Barrier helper
    //
    // Synchronize on observable state (waiter/inFlight counters) instead of
    // wall-clock sleeps. `Task.yield()` gives the runtime a chance to schedule
    // the target task without blocking a thread. Bounded by a total-yield budget
    // so a regression (waiter never enqueues) fails the test instead of hanging.
    static func waitUntil(
        _ condition: @Sendable () -> Bool,
        maxYields: Int = 10_000,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        for _ in 0..<maxYields {
            if condition() { return }
            await Task.yield()
        }
        try #require(condition(), "Barrier condition not met within \(maxYields) yields")
    }

    @Test("acquire beyond capacity suspends until release")
    func acquireBlocksAtCapacity() async throws {
        let gate = PipelineGate(capacity: 2)
        await gate.acquire()
        await gate.acquire()
        #expect(gate._inFlightForTesting == 2)

        let waiterDone = Atomic(false)
        let waiter = Task {
            await gate.acquire()
            waiterDone.store(true, ordering: .relaxed)
        }

        // Barrier: wait for the task to reach the slow-path enqueue.
        try await Self.waitUntil { gate._waitersForTesting == 1 }

        let doneBeforeRelease = waiterDone.load(ordering: .relaxed)
        #expect(!doneBeforeRelease)
        #expect(gate._inFlightForTesting == 2)

        gate.release()
        _ = await waiter.value

        let doneAfterRelease = waiterDone.load(ordering: .relaxed)
        #expect(doneAfterRelease)
        // Slot transferred to waiter, not decremented.
        #expect(gate._inFlightForTesting == 2)
        #expect(gate._waitersForTesting == 0)
    }

    @Test("release transfers slot to waiter (inFlight unchanged)")
    func releaseTransfersSlotToWaiter() async throws {
        let gate = PipelineGate(capacity: 1)
        await gate.acquire()
        #expect(gate._inFlightForTesting == 1)

        let waiter = Task { await gate.acquire() }

        // Barrier: waiter has enqueued on the slow path.
        try await Self.waitUntil { gate._waitersForTesting == 1 }

        gate.release()
        _ = await waiter.value

        // Slot stayed at 1 through the handoff — no decrement + re-increment.
        #expect(gate._inFlightForTesting == 1)
        #expect(gate._waitersForTesting == 0)
    }

    @Test("multiple waiters wake in FIFO order")
    func fifoWaiters() async throws {
        let gate = PipelineGate(capacity: 1)
        await gate.acquire()

        // Enqueue waiters serially: spawn each task only after the previous one
        // has reached the acquire() enqueue point. Guarantees FIFO enqueue
        // order without wall-clock staggering.
        let (stream, continuation) = AsyncStream.makeStream(of: Int.self)
        var tasks: [Task<Void, Never>] = []
        for i in 0..<4 {
            tasks.append(
                Task {
                    await gate.acquire()
                    continuation.yield(i)
                })
            // Barrier: wait for task i to be the (i+1)-th waiter.
            try await Self.waitUntil { gate._waitersForTesting == i + 1 }
        }
        #expect(gate._waitersForTesting == 4)

        // Release one slot at a time; each release wakes the next waiter.
        // The AsyncStream acts as the completion barrier (no sleep).
        var order: [Int] = []
        for _ in 0..<4 {
            gate.release()
            for await value in stream.prefix(1) {
                order.append(value)
            }
        }
        continuation.finish()
        for t in tasks { _ = await t.value }

        #expect(order == [0, 1, 2, 3])
    }

    @Test("many concurrent acquires with release converge to empty state")
    func stressConcurrentAcquireRelease() async {
        let gate = PipelineGate(capacity: 3)
        let totalTasks = 50

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<totalTasks {
                group.addTask {
                    await gate.acquire()
                    gate.release()
                }
            }
        }

        #expect(gate._inFlightForTesting == 0)
        #expect(gate._waitersForTesting == 0)
    }
}
