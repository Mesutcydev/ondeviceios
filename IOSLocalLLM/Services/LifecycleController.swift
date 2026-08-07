import Foundation
import UIKit
import SwiftUI

enum LifecycleBackgroundCleanupAction: Equatable, Sendable {
    case cancelQueuedMLX
    case awaitNativeGeneration
    case unloadResidentRuntimes
}

struct LifecycleBackgroundCleanupPlan: Equatable, Sendable {
    let actions: [LifecycleBackgroundCleanupAction]

    /// Cancellation is signalled immediately, while native Metal retirement
    /// and model ownership are left out of the short iOS background assertion.
    static let referenceCompatible = Self(actions: [.cancelQueuedMLX])
}

// MARK: - LifecycleController
//
// Single authority for reacting to scene-phase transitions.
//
// Design invariants:
//  1. The SwiftUI `.onChange(of: scenePhase)` callback MUST return immediately.
//     It only signals this actor's `handle(phase:)`, never does real work.
//  2. Repeated `.inactive` events are idempotent — an epoch counter invalidates
//     stale transition work.
//  3. `.inactive` is treated as transient (Control Center, alert dialogs, app
//     switcher preview). Models are NOT unloaded; only camera-feed submission
//     is paused.
//  4. `.background` does bounded cleanup: cancel inference cooperatively, stop
//     camera, persist a small recovery breadcrumb, and invalidate queued MLX
//     work. It never waits for an active Metal command buffer or tears down a
//     runtime that command buffer may still own.
//  5. `.active` cancels any obsolete background cleanup and resumes camera only
//     after permissions + runtime state are valid. Models are NOT auto-reloaded.

@MainActor
final class LifecycleController: Sendable {

    static let shared = LifecycleController()

    // MARK: - Epoch system

    /// Incremented on every lifecycle intent. Stale transition tasks check
    /// this after every suspension point and bail if it changed.
    private var lifecycleEpoch: UInt64 = 0

    /// Currently running background-transition task, if any. Cancelled when
    /// a new transition arrives or when `.active` is reached.
    private var backgroundTransitionTask: Task<Void, Never>?

    /// Background task identifier from UIApplication. Nil when no background
    /// task is active.
    private var bgTaskID: UIBackgroundTaskIdentifier = .invalid

    /// True when the expiration handler fired — cleanup must bail immediately.
    private var bgTaskExpired = false

    // MARK: - Lifecycle entry points

    /// Called from SwiftUI's `.onChange(of: scenePhase)`. Returns immediately;
    /// all work is dispatched asynchronously.
    func handle(phase: ScenePhase) {
        // Bump the epoch so any in-flight transition work sees it's stale.
        lifecycleEpoch &+= 1
        let epoch = lifecycleEpoch

        switch phase {
        case .active:
            Task { await onActive(epoch: epoch) }
        case .inactive:
            Task { await onInactive(epoch: epoch) }
        case .background:
            Task { await onBackground(epoch: epoch) }
        @unknown default:
            break
        }
    }

    // MARK: - .inactive

    /// Transient — do NOT unload models. Pause camera, stop accepting new
    /// Lens requests. Debounce destructive cleanup because Control Center,
    /// alerts, and system overlays all trigger `.inactive`.
    private func onInactive(epoch: UInt64) async {
        guard epoch == lifecycleEpoch else { return }
        Diagnostics.shared.breadcrumb("scene inactive · epoch=\(epoch)", category: "lifecycle")

        // Camera pauses itself via its own didEnterBackground notification
        // observer. The capture session stays alive during .inactive (Control
        // Center / alert dialogs don't kill it).

        guard epoch == lifecycleEpoch else { return }

        // Stop accepting new Lens requests — any in-flight describe() will
        // complete naturally.
        LensInferenceLoop.shared.suspendNewRequests()

        RecoveryManager.shared.markRunning(
            slot: LensInferenceLoop.shared.activeRepoID != nil ? "lens" : "assistant",
            modelID: CodingAssistantService.shared.activeModelRepoID,
            operation: "inactive"
        )
    }

    // MARK: - .background

    /// Bounded cleanup. Must not block the suspension. Uses an iOS background
    /// task with an expiration handler so the OS can kill us cleanly if
    /// cleanup takes too long.
    private func onBackground(epoch: UInt64) async {
        guard epoch == lifecycleEpoch else { return }
        Diagnostics.shared.breadcrumb("scene background · epoch=\(epoch)", category: "lifecycle")

        // Clear the crash marker before the first suspension point. iOS may
        // suspend or terminate a backgrounded process while an awaited
        // service shutdown is still pending; leaving the foreground marker
        // armed until after that await misclassifies an ordinary background
        // termination as a jetsam/watchdog crash on the next launch.
        CrashReporter.shared.markCleanShutdown()

        await LocalAPIManager.shared.stop()

        // Cancel any previous background transition.
        backgroundTransitionTask?.cancel()

        // Mark running state BEFORE starting cleanup.
        // If we get killed mid-cleanup, next launch sees unclean exit.
        RecoveryManager.shared.markRunning(
            slot: LensInferenceLoop.shared.activeRepoID != nil ? "lens" : "assistant",
            modelID: CodingAssistantService.shared.activeModelRepoID,
            operation: "background_transition"
        )

        // Block new inference IMMEDIATELY — must not start generation in background.
        LensInferenceLoop.shared.suspendNewRequests()
        // Camera handles its own lifecycle via didEnterBackground notification.

        // Remember an explicitly-started cold model load before cancellation
        // changes its state, so returning to the foreground can resume it.
        CodingAssistantService.shared.noteLifecycleInterruption()

        // Cancel active generation cooperatively. This sets the cancel flag
        // so the next token-iteration sees it and unwinds.
        CodingAssistantService.shared.stopGeneration()
        FastVLMService.shared.stopGeneration()
        ImageGenerationService.shared.cancel()
        MLXVisionService.shared.cancelCurrentInference()

        guard epoch == lifecycleEpoch else { return }

        // Save the minimal recovery record synchronously (small, atomic write).
        RecoveryManager.shared.markCleanShutdown(phase: "background")
        RecoveryManager.shared.flush()

        // Persist the conversation store. scheduleSave() is async; flush()
        // writes immediately but is small JSON.
        ConversationStore.shared.flush()

        // Take a short background assertion only for the bounded bookkeeping
        // below. Native generation and runtime teardown are deliberately not
        // awaited under this assertion.
        bgTaskExpired = false
        bgTaskID = UIApplication.shared.beginBackgroundTask(
            withName: "LifecycleCleanup"
        ) { [weak self] in
            Task { @MainActor [weak self] in
                self?.bgTaskExpired = true
                self?.backgroundTransitionTask?.cancel()
                Diagnostics.shared.breadcrumb("background task expired", category: "lifecycle")
            }
        }

        backgroundTransitionTask = Task {
            defer { endBackgroundTaskIfNeeded() }

            await performBackgroundCleanup(epoch: epoch)
        }
    }

    private func performBackgroundCleanup(epoch: UInt64) async {
        Diagnostics.shared.breadcrumb("background cleanup begin · epoch=\(epoch)", category: "lifecycle")

        // Cancel inference when focus is lost, but never keep an iOS
        // background task alive while waiting
        // for an already-submitted Metal command buffer. A long MLX prefill is
        // not synchronously cancellable. Awaiting it here made the background
        // task expire, and iOS watchdog-terminated the otherwise healthy app
        // with the final breadcrumb "draining generation task".
        //
        // `cancelAll()` invalidates queued work and returns without waiting for
        // the active command buffer. The resident model remains owned and can
        // finish retiring safely. MemoryPressureCoordinator is the sole low-
        // headroom authority and may still unload on a genuine pressure event.
        for action in LifecycleBackgroundCleanupPlan.referenceCompatible.actions {
            switch action {
            case .cancelQueuedMLX:
                await MLXGenerationGate.shared.cancelAll()
            case .awaitNativeGeneration:
                await CodingAssistantService.shared.cancelAndDrainInference()
            case .unloadResidentRuntimes:
                await CodingAssistantService.shared.unloadAndWaitForCleanup()
                LensInferenceLoop.shared.unload()
                FastVLMService.shared.unload()
                await LlamaCppVLMService.shared.unloadAndWaitForCleanup()
            }
        }

        guard epoch == lifecycleEpoch, !bgTaskExpired else { return }

        Diagnostics.shared.breadcrumb(
            "background cleanup complete · inference cancelled · runtime retained · epoch=\(epoch)",
            category: "lifecycle"
        )
    }

    // MARK: - .active

    private func onActive(epoch: UInt64) async {
        guard epoch == lifecycleEpoch else { return }
        Diagnostics.shared.breadcrumb("scene active · epoch=\(epoch)", category: "lifecycle")
        if AppSettings.shared.localAPIEnabled {
            await LocalAPIManager.shared.start()
        }
        // A paired Mac must be able to reconnect after an app relaunch without
        // requiring the user to open the Mac settings sheet first. The pairing
        // credentials already authorize the client; starting the LAN listener
        // here only restores the previously established local connection.
        if !BridgePairingStore.shared.clients().isEmpty,
           BridgeManager.shared.serverState == .stopped {
            await BridgeManager.shared.start()
        }

        // Cancel any background cleanup that's still running — we're back.
        backgroundTransitionTask?.cancel()
        endBackgroundTaskIfNeeded()

        // Re-arm the unclean-exit marker. A kill while active leaves it set.
        RecoveryManager.shared.markRunning(
            slot: "foreground",
            modelID: CodingAssistantService.shared.activeModelRepoID,
            operation: "active"
        )

        // Resume Lens request acceptance.
        LensInferenceLoop.shared.resumeNewRequests()

        // Camera handles its own lifecycle via willEnterForeground notification.
        // Do not normally auto-load a large model. The sole exception is an
        // explicit cold load that this lifecycle controller interrupted when
        // the app backgrounded; resume that same selection without reporting
        // a false model failure.
        await CodingAssistantService.shared.resumeInterruptedLoadIfNeeded()
        Diagnostics.shared.breadcrumb("scene active complete · epoch=\(epoch)", category: "lifecycle")
    }

    // MARK: - Background task management

    private func endBackgroundTaskIfNeeded() {
        guard bgTaskID != .invalid else { return }
        UIApplication.shared.endBackgroundTask(bgTaskID)
        bgTaskID = .invalid
        bgTaskExpired = false
    }
}
