import Foundation
import UIKit

// MARK: - DeviceSafetyMonitor
// Single source of truth for "is the device in a healthy state for heavy
// on-device ML work?" Watches three signals:
//
//   • ProcessInfo.thermalState      — hot device → throttle / refuse
//   • ProcessInfo.isLowPowerModeEnabled — explicit user signal to back off
//   • UIApplication memory warnings — late but unambiguous OS distress signal
//
// Anything that's about to spend GPU cycles for more than a second or two
// (model load, inference, model download) should consult `shouldThrottle`
// or `shouldStopHeavyWork` before starting and again periodically while running.
//
// We intentionally avoid heuristics like "CPU temp °C" — Apple doesn't expose
// it publicly, and ProcessInfo.thermalState already integrates that signal.

@MainActor
final class DeviceSafetyMonitor: ObservableObject {

    static let shared = DeviceSafetyMonitor()

    // MARK: - Published state

    @Published private(set) var thermalState: ProcessInfo.ThermalState
    @Published private(set) var lowPowerMode: Bool
    /// Increments every time iOS sends a memory warning. Useful for UI badges
    /// and for backoff heuristics ("we got 3 warnings in 30s — stop loading").
    @Published private(set) var memoryWarningCount: Int = 0
    /// Timestamp of the most recent memory warning, or nil if none yet.
    @Published private(set) var lastMemoryWarning: Date?
    /// True while the device is plugged in (charging or full). Speculative
    /// work (model prefetch) gates on this so it never burns battery.
    @Published private(set) var isCharging: Bool = false
    /// 0.0–1.0, or -1 when unknown (simulator / monitoring unavailable).
    /// Callers MUST treat negative values as "don't know — don't block".
    @Published private(set) var batteryLevel: Float = -1

    // MARK: - Derived

    /// True when the device needs to back off non-essential work.
    ///
    /// Aligned with Apple's official ProcessInfo.thermalState guidance:
    /// .nominal and .fair are normal operating states — `.fair` literally
    /// means "the system is slightly warmer than nominal, no action
    /// required". An A19 Pro sustaining tok/s on a 4B model lives in
    /// `.fair` permanently; treating that as "throttle" was why the app
    /// felt unusable after 30 seconds on a 17 Pro Max despite the
    /// device being objectively fine.
    ///
    /// We now follow Apple's actual guidance: throttle at `.serious`,
    /// stop at `.critical`. Peer on-device LLM/VLM apps (MLC, Ollama,
    /// Private LLM, Apple's own VLMEval sample) all use this same
    /// schedule.
    var shouldThrottle: Bool {
        guard AppSettings.shared.thermalWarningsEnabled else { return false }
        switch effectiveThermalState {
        case .serious, .critical: return true
        default:                   return false
        }
    }

    /// Why heavy work should stop right now, or nil when it's safe to proceed.
    /// Lets callers show an *accurate* message instead of blaming everything
    /// on heat.
    enum StopReason: Equatable {
        case thermal          // device genuinely at .critical thermal state
        case memoryPressure   // sustained low-memory warnings from iOS

        /// User-facing toast title/detail for this reason.
        var title: String {
            switch self {
            case .thermal:        return "Paused — device too hot"
            case .memoryPressure: return "Paused — low on memory"
            }
        }
        var detail: String {
            switch self {
            case .thermal:
                return "Let it cool for a minute before sending another message."
            case .memoryPressure:
                return "Free up memory — close other apps or unload a model — then retry."
            }
        }
    }

    /// The reason heavy work should stop, or nil when safe.
    ///
    /// Thermal only fires at `.critical` (imminent thermal shutdown). Memory
    /// pressure requires SUSTAINED warnings, not a single one: iOS fires
    /// memory warnings routinely during normal inference even when the
    /// process has ample headroom (see `CodingAssistantService`'s warning
    /// handler), so gating on one warning produced spurious "paused" banners
    /// on a device that was neither hot nor actually low on memory. We require
    /// several warnings inside a short window to treat it as real distress.
    var stopReason: StopReason? {
        // .critical is a SAFETY stop (imminent thermal shutdown), not a
        // UX nicety — it fires regardless of the thermalWarningsEnabled
        // setting. That setting only gates the softer .serious throttling,
        // labels, and toasts.
        if effectiveThermalState == .critical {
            return .thermal
        }
        if hasSustainedMemoryPressure {
            return .memoryPressure
        }
        return nil
    }

    /// True when the device is unsafe for new heavy work — generation
    /// should refuse to start and any in-flight loop should bail out.
    var shouldStopHeavyWork: Bool { stopReason != nil }

    /// Number of memory warnings inside the pressure window and the bar that
    /// counts as genuine distress. A real low-memory emergency produces a
    /// rapid burst; one or two warnings during a model load are normal.
    private static let memoryPressureWindow: TimeInterval = 10
    private static let memoryPressureThreshold = 3

    private var hasSustainedMemoryPressure: Bool {
        let cutoff = Date().addingTimeInterval(-Self.memoryPressureWindow)
        return memoryWarningTimestamps.filter { $0 >= cutoff }.count >= Self.memoryPressureThreshold
    }

    /// Recent memory-warning timestamps, pruned to the pressure window.
    private var memoryWarningTimestamps: [Date] = []

    /// Short human-readable label of the current state. Suitable for status
    /// pills in the UI. Returns nil when everything is nominal OR the user
    /// has opted out of thermal warnings.
    ///
    /// Critically: `.fair` returns nil. That state is the normal place a
    /// busy AI device lives — surfacing a "warming up" pill there is
    /// alarmist and made users think something was wrong when nothing
    /// was. Low-power mode still surfaces a passive label since the user
    /// explicitly opted in and likely wants to see why generation feels
    /// slower.
    var statusLabel: String? {
        guard AppSettings.shared.thermalWarningsEnabled else {
            return lowPowerMode ? "low-power mode" : nil
        }
        switch effectiveThermalState {
        case .critical: return "device too hot — paused"
        case .serious:  return "device warm — throttled"
        default:
            return lowPowerMode ? "low-power mode" : nil
        }
    }

    // MARK: - Hysteresis

    /// Brief spikes — a single second of .serious during the heavy first
    /// inference of a session — shouldn't pop the pill. We require the state
    /// to hold for 4 seconds before flipping the UI, and require .nominal/
    /// .fair to hold for 6 seconds before clearing a hot indicator.
    /// `effectiveThermalState` is what UI + gates consult.
    @Published private(set) var effectiveThermalState: ProcessInfo.ThermalState = .nominal
    private var pendingTransition: Task<Void, Never>?

    /// Glyph color hint. "red" / "orange" / nil — `.fair` no longer
    /// returns a yellow because we don't surface anything at that state.
    var statusColor: String? {
        // Switch on the SAME hysteresis-filtered state the label and
        // gates use — switching on the raw state made the color flap
        // and disagree with `statusLabel` during brief spikes.
        switch effectiveThermalState {
        case .critical: return "red"
        case .serious:  return "orange"
        default:        return lowPowerMode ? "orange" : nil
        }
    }

    // MARK: - Recommendations

    /// Suggested max-token cap for the current thermal state. Generation
    /// paths pass this through `min(userSetting, recommendedMaxTokens)`.
    ///
    /// Aligned with Apple's thermal guidance and what's actually safe on
    /// modern hardware: `.nominal` and `.fair` are full speed (no cap
    /// beyond the user's setting). `.serious` only mildly reduces. The
    /// previous schedule (1280 cap at `.fair`!) made replies feel
    /// truncated on a perfectly healthy A19 Pro that was just *slightly
    /// warm* — exactly the regime Apple says is fine.
    ///
    /// Low-power mode doesn't cap at all anymore: iOS already throttles
    /// CPU/GPU clocks at the system level when LPM is on. Adding a token
    /// cap on top of that is double-counting and produces shorter replies
    /// for no extra battery saving. We just surface a passive "low-power
    /// mode" badge so the user knows clocks are reduced.
    var recommendedMaxTokens: Int {
        // Device-tier ceiling, folded in so a low-RAM device can't exceed its
        // safe KV-cache budget even with thermal warnings off. Uses a generous
        // 3× of DeviceTierAdvisor's per-tier value so the SHIPPING tiers don't
        // regress: .max (1536×3=4608)→4096, .pro (1280×3=3840)→3840 ≈ no change;
        // only hypothetical low-RAM tiers (lite 512×3=1536) get clamped down.
        let tierCeiling = DeviceTierAdvisor.recommendedMaxTokens * 3
        // The .critical clamp is a safety measure and applies even when
        // the user opted out of thermal warnings; only the softer
        // .serious clamp is gated by the setting.
        if effectiveThermalState == .critical { return min(512, tierCeiling) }
        guard AppSettings.shared.thermalWarningsEnabled else { return min(4096, tierCeiling) }
        switch effectiveThermalState {
        case .serious:  return min(1536, tierCeiling)
        default:        return min(4096, tierCeiling)   // .nominal, .fair, low-power
        }
    }

    // MARK: - Init / lifecycle

    // INVARIANT: `.shared` must be FIRST touched on the main actor — this
    // @MainActor init reads UIDevice/UIApplication and registers observers, all
    // main-actor work. In practice the first access is always on main
    // (ContentView / app startup), and because the singleton is @MainActor the
    // compiler already requires `await` to reach it from any off-main context,
    // so an accidental off-main first-touch can't compile. Keep it that way.
    private init() {
        self.thermalState  = ProcessInfo.processInfo.thermalState
        self.effectiveThermalState = ProcessInfo.processInfo.thermalState
        self.lowPowerMode  = ProcessInfo.processInfo.isLowPowerModeEnabled
        UIDevice.current.isBatteryMonitoringEnabled = true
        let batteryState = UIDevice.current.batteryState
        self.isCharging   = batteryState == .charging || batteryState == .full
        self.batteryLevel = UIDevice.current.batteryLevel
        subscribe()
    }

    /// Manual reset hook — useful when the user is confident the device is
    /// fine but iOS hasn't dropped the .serious flag yet. Flips the effective
    /// state to .nominal until iOS reports a change.
    func clearThermalWarning() {
        pendingTransition?.cancel()
        effectiveThermalState = .nominal
        // iOS only posts thermalStateDidChange on TRANSITIONS — if the
        // device stays hot, no further notification arrives and the forced
        // .nominal would stick forever. Re-sync with ProcessInfo after a
        // while through the normal path. Stored in `pendingTransition` so
        // a real notification (which reschedules via
        // scheduleEffectiveTransition) cancels this re-sync first.
        pendingTransition = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 45_000_000_000)
            guard !Task.isCancelled, let self else { return }
            let now = ProcessInfo.processInfo.thermalState
            self.thermalState = now
            self.scheduleEffectiveTransition(to: now)
        }
    }

    // MARK: - Keep-awake (idle timer)

    /// Reasons that currently want the screen kept awake (e.g. "benchmark",
    /// "lens-live-loop"). The idle timer is disabled while ANY reason is
    /// active and restored the moment the set empties — reference-counted
    /// by reason string so two owners can't stomp each other.
    private var keepAwakeReasons: Set<String> = []

    /// Central UIApplication.isIdleTimerDisabled owner. Callers MUST pair
    /// every `true` with a `false` for the same reason (use `defer`).
    /// The background observer below force-clears the flag so a leaked
    /// `true` can never keep the screen unlocked past the app's lifetime
    /// in foreground.
    func setKeepAwake(_ on: Bool, reason: String) {
        if on {
            keepAwakeReasons.insert(reason)
        } else {
            keepAwakeReasons.remove(reason)
        }
        UIApplication.shared.isIdleTimerDisabled = !keepAwakeReasons.isEmpty
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        pendingTransition?.cancel()
    }

    private func subscribe() {
        let nc = NotificationCenter.default
        nc.addObserver(
            self, selector: #selector(thermalStateChanged),
            name: ProcessInfo.thermalStateDidChangeNotification, object: nil
        )
        nc.addObserver(
            self, selector: #selector(powerModeChanged),
            name: .NSProcessInfoPowerStateDidChange, object: nil
        )
        nc.addObserver(
            self, selector: #selector(memoryWarning),
            name: UIApplication.didReceiveMemoryWarningNotification, object: nil
        )
        nc.addObserver(
            self, selector: #selector(batteryChanged),
            name: UIDevice.batteryStateDidChangeNotification, object: nil
        )
        nc.addObserver(
            self, selector: #selector(batteryChanged),
            name: UIDevice.batteryLevelDidChangeNotification, object: nil
        )
        // Keep-awake hygiene: never leave the idle timer disabled across a
        // background transition, and re-apply outstanding reasons on return
        // so an in-flight benchmark / live loop keeps working.
        nc.addObserver(
            self, selector: #selector(appDidEnterBackground),
            name: UIApplication.didEnterBackgroundNotification, object: nil
        )
        nc.addObserver(
            self, selector: #selector(appWillEnterForeground),
            name: UIApplication.willEnterForegroundNotification, object: nil
        )
    }

    @objc private func thermalStateChanged() {
        let new = ProcessInfo.processInfo.thermalState
        Task { @MainActor [weak self] in
            self?.thermalState = new
            self?.scheduleEffectiveTransition(to: new)
        }
    }

    /// Apply hysteresis before letting the UI / gates see a thermal change.
    /// Filters out brief spikes so the pill doesn't flicker.
    ///
    /// Going hotter is slower than going cooler: we don't want to alarm
    /// the user the instant `.serious` is touched (Apple sometimes reports
    /// it for a few seconds during prompt-prefill, then drops back to
    /// `.fair`). Going cooler is also relaxed — the previous 6s cool was
    /// short enough that the pill would re-appear immediately on the next
    /// inference. 8s/15s feels stable in long chat sessions.
    private func scheduleEffectiveTransition(to new: ProcessInfo.ThermalState) {
        pendingTransition?.cancel()

        // .critical means imminent thermal shutdown — apply it IMMEDIATELY,
        // no hysteresis. An 8s wait here would let heavy GPU work keep
        // running through the most dangerous window. The 15s cool-down
        // path below is unchanged.
        if new == .critical {
            let before = effectiveThermalState
            effectiveThermalState = .critical
            if before != .critical {
                surfaceToast(for: .critical)
            }
            return
        }

        let goingHotter = new.rawValue > effectiveThermalState.rawValue
        let delaySec: Double = goingHotter ? 8.0 : 15.0

        pendingTransition = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delaySec * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            // Re-read so we transition to whatever the CURRENT state is,
            // not the one we observed when scheduling.
            let now = ProcessInfo.processInfo.thermalState
            let before = self.effectiveThermalState
            self.effectiveThermalState = now
            if before != now {
                self.surfaceToast(for: now)
            }
        }
    }

    private func surfaceToast(for new: ProcessInfo.ThermalState) {
        guard AppSettings.shared.thermalWarningsEnabled else { return }
        // Only the truly dangerous state surfaces a toast now. `.serious` is
        // common on heavy on-device inference and the previous "Device is
        // warm" pill fired multiple times per session, making the app feel
        // panicky compared to peer SmolVLM demos that say nothing in the
        // same conditions.
        if case .critical = new {
            ToastCenter.shared.error(
                "Device too hot",
                detail: "Pausing AI work to protect your device. Set it down to cool for a minute."
            )
        }
    }

    @objc private func powerModeChanged() {
        let isOn = ProcessInfo.processInfo.isLowPowerModeEnabled
        Task { @MainActor [weak self] in self?.lowPowerMode = isOn }
    }

    @objc private func batteryChanged() {
        let state = UIDevice.current.batteryState
        let level = UIDevice.current.batteryLevel
        Task { @MainActor [weak self] in
            self?.isCharging = (state == .charging || state == .full)
            self?.batteryLevel = level
        }
    }

    @objc private func appDidEnterBackground() {
        // Force-clear the OS flag (keep the reasons set — the owners are
        // still logically active and re-arm on foreground). This makes it
        // impossible for a leaked keep-awake to outlive a backgrounding.
        Task { @MainActor in
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    @objc private func appWillEnterForeground() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            UIApplication.shared.isIdleTimerDisabled = !self.keepAwakeReasons.isEmpty
        }
    }

    @objc private func memoryWarning() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            let now = Date()
            self.memoryWarningCount += 1
            self.lastMemoryWarning = now
            // Record into the sliding window and prune anything older than the
            // pressure window so `hasSustainedMemoryPressure` only sees a
            // genuine recent burst.
            let cutoff = now.addingTimeInterval(-Self.memoryPressureWindow)
            self.memoryWarningTimestamps.removeAll { $0 < cutoff }
            self.memoryWarningTimestamps.append(now)
        }
    }
}

// MARK: - MemoryPressureCoordinator
//
// Owns the KERNEL memory-pressure signal and coordinates emergency model
// teardown. This complements the UIKit `didReceiveMemoryWarning` handling
// above.
//
// Why a second source: `DispatchSource.makeMemoryPressureSource` fires
// milliseconds before Jetsam and, crucially, distinguishes `.warning` from
// `.critical`. UIKit's `didReceiveMemoryWarning` is ROUTINE — iOS posts it
// liberally during normal multi-GB model loads even when the process has
// ample headroom — so dumping the model there causes load→unload→reload
// thrash (and, ironically, more pressure). The kernel source lets us:
//
//   • `.warning`  → reclaim only the cheap/discardable stuff, KEEP the model.
//   • `.critical` → genuinely imminent Jetsam: dump every heavy model now,
//                   then start a short cooldown so the reload doesn't race
//                   iOS while it's still reclaiming.
//
// Plus a background guard: when we're backgrounded with little headroom, shed
// weights so a background spike (push-notification decode, another app
// launching) can't Jetsam us while we're not even on screen.

@MainActor
final class MemoryPressureCoordinator {

    static let shared = MemoryPressureCoordinator()

    private var pressureSource: DispatchSourceMemoryPressure?
    private var backgroundObserver: NSObjectProtocol?

    private init() {}

    /// Headroom below which a background transition triggers a proactive
    /// unload. Mirrors the OS reserve we keep elsewhere.
    private static let backgroundUnloadHeadroom: Int64 = 1_500_000_000

    /// Process headroom below which a kernel CRITICAL event is treated as OUR
    /// emergency (dump weights) rather than another app's. A `.critical`
    /// dispatch pressure event is SYSTEM-WIDE — iOS posts it to every process
    /// when overall device memory is tight — so above this we keep the model
    /// resident and only shed discardable caches.
    private static let criticalDumpHeadroom: Int64 = 700_000_000

    /// Start watching. Idempotent — safe to call from `ContentView.onAppear`,
    /// which can fire more than once.
    func start() {
        guard pressureSource == nil else { return }

        let src = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: DispatchQueue.global(qos: .utility))
        // `src` is captured strongly inside its own handler (a retain cycle),
        // which is the documented pattern for a process-lifetime source that
        // is never cancelled. Reading `src.data` is Sendable and safe off the
        // main actor; the actual reclaim work hops back to @MainActor.
        src.setEventHandler {
            let event = src.data
            Task { @MainActor in
                if event.contains(.critical) {
                    MemoryPressureCoordinator.shared.handleCritical()
                } else if event.contains(.warning) {
                    MemoryPressureCoordinator.shared.handleWarning()
                }
            }
        }
        src.resume()
        pressureSource = src

        backgroundObserver = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil, queue: .main
        ) { _ in
            Task { @MainActor in MemoryPressureCoordinator.shared.handleBackground() }
        }
    }

    // MARK: - Handlers

    /// Routine pressure: free the cheapest-to-reclaim memory (the MLX/Metal
    /// buffer pool + any discardable caches) and KEEP the model resident.
    /// Unloading here would thrash a model that's merely mid-load.
    private func handleWarning() {
        Task { await MLXGenerationGate.shared.clearCacheWhenIdle() }
        NotificationCenter.default.post(name: .iosLocalLLMDropDiscardableCaches, object: nil)
    }

    /// Imminent Jetsam: dump every heavy model and open the reload cooldown so
    /// we don't immediately reload into the same pressure and get killed.
    ///
    /// But the kernel `.critical` event is SYSTEM-WIDE, not per-process: iOS
    /// fires it at every app when overall device memory is tight, even when
    /// THIS process sits comfortably under its own limit. Dumping a healthy
    /// resident model in response (and opening the 20s reload cooldown) is what
    /// made the assistant unusable — another app's pressure tore down the
    /// loaded model and the retry landed inside the cooldown. So first reclaim
    /// the cheap buffer pool, then re-check OUR headroom; only dump if we're
    /// genuinely the one in trouble.
    private func handleCritical() {
        handleWarning()   // flush MLX/Metal pool + drop discardable caches
        if MemoryAdvisor.availableMemoryForModel > Self.criticalDumpHeadroom {
            return        // pressure is another app's — keep our model resident
        }
        unloadAllHeavyModels()
        // Drain queued MLX gate bodies too — without this, work queued
        // behind the gate still executes after the dump and re-allocates
        // into the very pressure we just relieved.
        Task { await MLXGenerationGate.shared.cancelAll() }
        MemoryAdvisor.notePressureDump()
        NotificationCenter.default.post(name: .iosLocalLLMDropDiscardableCaches, object: nil)
        ToastCenter.shared.error(
            "Freed memory to stay running",
            detail: "iOS was about to run out of memory, so models were unloaded. They'll reload automatically when you use them.")
    }

    /// Backgrounded with little headroom → shed weights proactively. When we
    /// have plenty of room we leave the model resident so foregrounding is
    /// instant.
    private func handleBackground() {
        // Compare the RAW kernel headroom, not the entitlement-fused
        // `availableMemoryForModel`: the entitlement fraction is a
        // FOREGROUND figure, and background jetsam limits sit far lower —
        // the fused number would keep weights resident exactly when a
        // backgrounded process is most likely to be killed.
        guard MemoryAdvisor.processAvailableMemory < Self.backgroundUnloadHeadroom else { return }
        unloadAllHeavyModels()
    }

    /// Tear down every multi-hundred-MB model service and flush the GPU pool.
    /// Each `unload()` is safe to call when nothing is loaded (no-op) and
    /// cancels any in-flight inference.
    private func unloadAllHeavyModels() {
        CodingAssistantService.shared.unload()
        MLXVisionService.shared.unload()
        LlamaCppVLMService.shared.unload()
        FastVLMService.shared.unload()
        ImageGenerationService.shared.unload()
        Task { await MLXGenerationGate.shared.clearCacheWhenIdle() }
    }
}

extension Notification.Name {
    /// Posted on memory pressure so caches (decoded images, thumbnails, etc.)
    /// can drop their discardable contents while the model stays resident.
    static let iosLocalLLMDropDiscardableCaches = Notification.Name("iosLocalLLMDropDiscardableCaches")
}
