import Foundation
import MLX
import UIKit

// MARK: - MLX simulator safety
//
// `Memory.clearCache()` (and any first touch of the MLX GPU) constructs an
// `mlx::core::metal::Device`, which calls `abort()` on the iOS Simulator —
// the Simulator has no Metal GPU that MLX can bind to. That turned ordinary
// navigation (switching to the Models / Mac tab unloads models and flushes
// the buffer pool) into a hard crash whenever the app ran in the Simulator,
// making UI work un-runnable there.
//
// Routing every app-side cache clear through this shim keeps the app
// navigable in the Simulator. On a real device the `#if` compiles straight
// to `Memory.clearCache()` — identical behavior, zero overhead — so device
// memory reclamation is completely unchanged.
@inline(__always)
func mlxClearCache() {
    #if !targetEnvironment(simulator)
    Memory.clearCache()
    #endif
}

// MARK: - MLXGenerationGate
//
// App-wide serial gate for MLX inference and model load.
//
// Why this exists: MLX submits work to Metal's global command queue. If two
// `container.perform { ... generate(...) }` calls run concurrently — e.g.
// CodingAssistantService streaming a reply while AnalysisService analyzes a
// frame — the Metal command buffer can come back with status `Error`, and
// MLX then throws `std::runtime_error` from `mlx::core::gpu::check_error`.
// That's a C++ exception; Swift's `do { try } catch` does NOT catch it, and
// the runtime calls `terminate()` → SIGABRT. The crash signature is:
//
//   Thread N: signal SIGABRT
//   ... mlx::core::gpu::check_error(MTL::CommandBuffer*) at metal_impl.h:25
//
// Every MLX inference and load across the app must funnel through
// `MLXGenerationGate.shared.run { ... }` so they execute strictly serially.
actor MLXGenerationGate {

    static let shared = MLXGenerationGate()

    /// Thrown when a queued body is skipped — either the caller's outer
    /// `Task` was cancelled while waiting in the chain, or `cancelAll()`
    /// drained the queue.
    struct Cancelled: Error {}

    /// Tail of the serial chain. Each new call awaits the previous task
    /// before running, which serializes work even though Swift actors are
    /// reentrant at await points.
    private var tail: Task<Void, Never> = Task {}

    /// Monotonic generation token. Incremented by `cancelAll()` so any body
    /// queued at the old generation skips itself instead of executing after
    /// the user-visible cancel.
    private var generation: UInt64 = 0

    /// Foreground latch. iOS denies GPU submissions from a backgrounded app
    /// with `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`,
    /// which MLX surfaces as an uncatchable C++ `std::runtime_error` from a
    /// Metal completion handler. We flip this to `false` on `willResignActive`
    /// (which fires BEFORE the OS revokes GPU access) and refuse all new
    /// submissions while it's `false`. Initialized from the current app
    /// state at construction so the gate behaves correctly if the very first
    /// caller arrives before the notification observers have fired.
    private var isForegrounded: Bool = true

    private init() {
        Task { [weak self] in
            await self?.bootstrapForegroundState()
            await self?.installSystemObservers()
        }
    }

    // MARK: - Public API

    /// Run `body` after every previously-submitted gate operation finishes.
    /// Wrap every MLX `container.perform { ... }` and model load call site
    /// in this.
    ///
    /// Throws `Cancelled` if the caller's surrounding Task is cancelled
    /// while waiting in the chain, or if `cancelAll()` was invoked between
    /// enqueue and execution. Re-throws whatever `body` throws otherwise.
    func run<T: Sendable>(
        _ body: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        // Pre-flight background guard. The notification observer flips this
        // false on willResignActive, which is the earliest cross-app signal
        // that the OS is about to revoke GPU access. Refusing here is the
        // ONLY reliable place — by the time the body actually executes, it
        // may have already submitted a Metal command buffer.
        if !isForegrounded { throw Cancelled() }
        let previous = tail
        let enqueuedAt = generation
        let task = Task<T, Error> { [weak self] in
            await previous.value
            // Cancellation guard: bail before touching the GPU if either
            //   (a) the caller's outer Task was cancelled while we waited,
            //   (b) cancelAll() bumped the generation token, or
            //   (c) the app backgrounded while we were queued.
            // Without this we'd still run stale work the user wanted dropped,
            // which on a memory-warning path is the exact thing that crashes.
            if Task.isCancelled { throw Cancelled() }
            if let current = await self?.generation, current != enqueuedAt {
                throw Cancelled()
            }
            if let fg = await self?.isForegrounded, !fg { throw Cancelled() }
            return try await body()
        }
        tail = Task { _ = try? await task.value }
        // `task` is unstructured, so cancelling the awaiting caller neither
        // interrupts this await nor reaches the body's cooperative
        // `Task.isCancelled` checks — propagate it explicitly.
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    /// Drain the chain: any body already queued but not yet executing will
    /// throw `Cancelled` instead of running. The currently-executing body
    /// is left alone — Metal can't be interrupted safely mid-command-buffer.
    /// Cache reclamation is appended to the serial chain so it cannot race
    /// that current Metal work. Calling `mlxClearCache()` directly here was a
    /// real crash path: cancellation could arrive while a VLM command buffer
    /// was still executing, and the buffer pool was cleared underneath it.
    /// Call on critical memory pressure or when the app backgrounds.
    func cancelAll() {
        generation &+= 1
        enqueueCacheClear()
    }

    /// Wait for all MLX work already admitted to the gate, then clear the
    /// allocator cache before allowing later work to begin. The clear itself
    /// becomes the new `tail`, which closes the actor-reentrancy hole where a
    /// new `run` could otherwise start after `await previous.value` but before
    /// cache reclamation completed.
    func clearCacheWhenIdle() async {
        let clear = enqueueCacheClear()
        await clear.value
    }

    /// Run non-GPU runtime teardown after all admitted MLX work has drained,
    /// then clear the allocator cache on a utility executor. Large
    /// `ModelContainer` instances can synchronously release Metal-backed
    /// storage in `deinit`; doing that from an `@MainActor` owner can freeze
    /// the app long enough for the foreground watchdog to terminate it.
    ///
    /// Installing this operation as `tail` preserves the same ordering as a
    /// normal cache clear: no later load or generation can begin until both
    /// the owner release and allocator reclamation have finished.
    func cleanupRuntimeWhenIdle(
        _ cleanup: @Sendable @escaping () -> Void
    ) async {
        let previous = tail
        let cleanupTask = Task.detached(priority: .utility) {
            await previous.value
            cleanup()
            mlxClearCache()
        }
        tail = Task { await cleanupTask.value }
        await cleanupTask.value
    }

    // MARK: - Internal

    private func bootstrapForegroundState() async {
        // Capture the application state ON THE MAIN ACTOR (the only thread
        // UIApplication.shared is safe to read from) before any caller
        // reaches `run()`. Without this, an inference fired in the first
        // few milliseconds of process launch could race the observer
        // registration and assume foreground when we're actually launching
        // from a background-launch path (background URL session, push, etc).
        // BOOTSTRAP-only: treat .inactive as foreground — on a cold launch
        // the app sits in .inactive until didBecomeActive, and reading that
        // as "backgrounded" made early auto-loads throw Cancelled spuriously.
        // Later transitions are still latched by the willResignActive observer.
        let active = await MainActor.run {
            UIApplication.shared.applicationState != .background
        }
        isForegrounded = active
    }

    private func installSystemObservers() {
        let center = NotificationCenter.default
        // Memory warning → append GPU-pool reclamation behind admitted work.
        // Even allocator cleanup must not overlap an in-flight Metal command.
        Task { [weak self] in
            for await _ in center.notifications(
                named: UIApplication.didReceiveMemoryWarningNotification
            ).map({ _ in () }) {
                await self?.handleMemoryWarning()
            }
        }
        // willResignActive is the EARLIEST signal that the OS is about to
        // stop scheduling our GPU work. It fires before didEnterBackground
        // (which we also observe as a backup), and it gives us a window —
        // typically a few hundred ms — where the foreground GPU lease is
        // still valid and any in-flight command buffer can drain.
        // Flipping `isForegrounded = false` here makes `run()` refuse new
        // submissions immediately so we don't accept work that the OS will
        // kill mid-buffer.
        Task { [weak self] in
            for await _ in center.notifications(
                named: UIApplication.willResignActiveNotification
            ).map({ _ in () }) {
                await self?.markBackgrounded()
            }
        }
        // Backgrounding → drop queued work AND ensure the latch is down.
        // iOS suspends GPU access for background apps; an inference fired
        // right as we suspend can leave a command buffer in `Error` status,
        // which is another SIGABRT path when MLX inspects the result.
        Task { [weak self] in
            for await _ in center.notifications(
                named: UIApplication.didEnterBackgroundNotification
            ).map({ _ in () }) {
                await self?.markBackgrounded()
                await self?.cancelAll()
            }
        }
        // didBecomeActive raises the latch so new inferences can run again
        // once iOS hands GPU access back. willEnterForeground is too early
        // — the GPU lease isn't restored until we're fully active.
        Task { [weak self] in
            for await _ in center.notifications(
                named: UIApplication.didBecomeActiveNotification
            ).map({ _ in () }) {
                await self?.markForegrounded()
            }
        }
    }

    private func handleMemoryWarning() {
        enqueueCacheClear()
    }

    private func markBackgrounded() {
        isForegrounded = false
    }

    private func markForegrounded() {
        isForegrounded = true
    }

    /// Append a cache clear to the same chain used by loads/inference.
    /// Actor isolation makes reading and replacing `tail` atomic.
    @discardableResult
    private func enqueueCacheClear() -> Task<Void, Never> {
        let previous = tail
        let clear = Task {
            await previous.value
            mlxClearCache()
        }
        tail = clear
        return clear
    }

    /// Read-only foreground check for short-circuiting callers (e.g.
    /// LensInferenceLoop.describe) that want to refuse work BEFORE
    /// constructing the gate body. Kept separate from `run()` so callers
    /// don't have to await a thrown `Cancelled` to discover the state.
    var foregrounded: Bool { isForegrounded }
}
