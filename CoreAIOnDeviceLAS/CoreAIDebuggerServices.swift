import Foundation
import Metal
import UIKit

// Core AI editions of the debugger's supporting services. The on-device
// debugger view (OnDeviceDebuggerView.swift) is shared with OnDeviceLAS, but
// this target deliberately does not compile the reference app's MLX/llama
// admission stack, so the memory and safety services live here with the same
// public surface and the same measurement semantics.

#if os(iOS) && !targetEnvironment(simulator) && !targetEnvironment(macCatalyst)
// Security.framework exports these on iOS, but the Swift overlay does not
// expose SecTask declarations. Keep the ABI bridge local and minimal so the
// running code signature—not the source .entitlements file—is authoritative.
@_silgen_name("SecTaskCreateFromSelf")
private func CoreAICreateSecurityTask(_ allocator: CFAllocator?) -> CFTypeRef?

@_silgen_name("SecTaskCopyValueForEntitlement")
private func CoreAICopyEntitlementValue(
    _ task: CFTypeRef,
    _ entitlement: CFString,
    _ error: UnsafeMutablePointer<Unmanaged<CFError>?>?
) -> CFTypeRef?
#endif

// MARK: - MemoryAdvisor

enum MemoryAdvisor {

    /// Total physical RAM in bytes.
    static var deviceTotalRAM: Int64 {
        Int64(ProcessInfo.processInfo.physicalMemory)
    }

    /// RAM available for app use after a 30% headroom for OS + other apps.
    static var availableRAM: Int64 {
        Int64(Double(deviceTotalRAM) * 0.70)
    }

    /// Live per-process memory headroom in bytes — how much more this app can
    /// allocate before iOS kills it for memory. Reports 0 when unavailable.
    static var processAvailableMemory: Int64 {
        Int64(os_proc_available_memory())
    }

    /// Device RAM not resident in THIS process (deviceTotalRAM − our own
    /// `resident_size`), in bytes. Reports 0 on failure.
    static var nonResidentRAMEstimate: Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            let used = Int64(info.resident_size)
            return max(0, deviceTotalRAM - used)
        }
        return 0
    }

    /// This process's current `phys_footprint` in bytes — the SAME accounting
    /// the kernel uses to decide `os_proc_available_memory()` and to Jetsam
    /// the app. Reports 0 on failure.
    static var physFootprint: Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
    }

    /// Fraction of physical RAM the app can realistically use as a hard
    /// ceiling on this device class (same tier schedule as OnDeviceLAS).
    private static var entitlementCeilingFraction: Double {
        let gb = Double(deviceTotalRAM) / 1_073_741_824
        switch gb {
        case ..<3.5:  return 0.55   // ≤4 GB — keep very tight
        case ..<5.0:  return 0.55
        case ..<7.0:  return 0.62   // 6 GB
        case ..<10.0: return 0.72   // 8 GB → ~6.2 GB usable
        default:      return 0.76   // 12 GB+ → ~9.8 GB usable
        }
    }

    /// Hard reserve always kept for the OS + other apps, regardless of tier.
    private static let physicalOSReserve: Int64 = 1_500_000_000

    /// Conservative process cap for iPhones below the 12 GB Max tier.
    static let maximumIPhoneProcessCeiling: Int64 = 6_200_000_000

    /// 12 GB Pro Max devices have a materially larger entitled process budget.
    static let maximumHighMemoryIPhoneProcessCeiling: Int64 = 9_200_000_000
    static let highMemoryIPhoneRAMThreshold: Int64 = 10_000_000_000

    /// Whether the entitlement is present in the signature iOS is actually
    /// running. Declaring the key in the source entitlements file is not
    /// sufficient: an unsigned IPA, or a sideloading profile that does not
    /// grant it, runs with the normal process memory limit.
    static var hasIncreasedMemoryLimitEntitlement: Bool {
        #if os(iOS) && !targetEnvironment(simulator) && !targetEnvironment(macCatalyst)
        guard let task = CoreAICreateSecurityTask(nil),
              let value = CoreAICopyEntitlementValue(
                task,
                "com.apple.developer.kernel.increased-memory-limit" as CFString,
                nil
              ) else {
            return false
        }
        return (value as? Bool) == true
        #else
        // Simulator and Catalyst are not governed by iOS Jetsam provisioning.
        return true
        #endif
    }

    private static var isIPhoneProcess: Bool {
        #if os(iOS) && !targetEnvironment(macCatalyst)
        return UIDevice.current.userInterfaceIdiom == .phone
        #else
        return false
        #endif
    }

    /// Applies physical and platform bounds to a candidate process ceiling.
    static nonisolated func clampedProcessCeiling(
        candidate: Int64,
        totalRAM: Int64,
        isPhone: Bool
    ) -> Int64 {
        let physicalCap = max(0, totalRAM - physicalOSReserve)
        let phoneCap = totalRAM >= highMemoryIPhoneRAMThreshold
            ? maximumHighMemoryIPhoneProcessCeiling
            : maximumIPhoneProcessCeiling
        let platformCap = isPhone ? min(physicalCap, phoneCap) : physicalCap
        return min(max(0, candidate), platformCap)
    }

    /// Selects the ceiling signal that is legal for the running signature.
    static nonisolated func resolvedProcessCeilingCandidate(
        kernelCeiling: Int64,
        entitlementCeiling: Int64,
        hasIncreasedMemoryEntitlement: Bool,
        lowPowerMode: Bool
    ) -> Int64 {
        guard hasIncreasedMemoryEntitlement, !lowPowerMode else {
            return kernelCeiling
        }
        return max(kernelCeiling, entitlementCeiling)
    }

    /// Best estimate of this process's hard memory ceiling (the per-process
    /// limit iOS enforces), in bytes — independent of what is loaded now.
    /// Fuses the kernel signal (`os_proc_available_memory()` + current
    /// `phys_footprint`, which recover the limit stably) with the
    /// entitlement-fraction estimate, then clamps to the platform budget.
    static var processMemoryCeiling: Int64 {
        let avail = processAvailableMemory
        let footprint = physFootprint
        let kernelCeiling: Int64 = (avail > 0 && footprint > 0)
            ? avail + footprint
            : (avail > 0 ? avail + max(0, deviceTotalRAM - nonResidentRAMEstimate) : 0)
        let entitlementCeiling = Int64(Double(deviceTotalRAM) * entitlementCeilingFraction)
        let best = resolvedProcessCeilingCandidate(
            kernelCeiling: kernelCeiling,
            entitlementCeiling: entitlementCeiling,
            hasIncreasedMemoryEntitlement: hasIncreasedMemoryLimitEntitlement,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
        let fallback = best > 0 ? best : availableRAM
        return clampedProcessCeiling(
            candidate: fallback,
            totalRAM: deviceTotalRAM,
            isPhone: isIPhoneProcess
        )
    }

    /// Live memory the app can realistically still allocate RIGHT NOW, in
    /// bytes — the figure the debugger's admission card reports.
    static var availableMemoryForModel: Int64 {
        let footprint = physFootprint
        return max(0, processMemoryCeiling - footprint)
    }
}

// MARK: - SystemStatusService

@MainActor
final class SystemStatusService: ObservableObject {

    static let shared = SystemStatusService()

    @Published private(set) var snapshot: Snapshot = .empty

    struct Snapshot {
        // Memory
        var totalRAM: Int64 = 0
        var usedByApp: Int64 = 0
        var availableForML: Int64 = 0
        var freeRightNow: Int64 = 0
        var hasIncreasedMemoryLimitEntitlement = false
        var lowPowerModeEnabled = false

        // Disk
        var diskFree: Int64 = 0
        var modelStorageUsed: Int64 = 0

        // Compute
        var device: String = ""
        var os: String = ""
        var metalDeviceName: String = ""
        var metalSupportsRayTracing: Bool = false
        var supportsNeuralEngine: Bool = false

        // Model runtime (Core AI edition: the language model is the only
        // resident runtime in this product).
        var qwen3Ready: Bool = false
        var fastVLMReady: Bool = false
        var voiceEngineReady: Bool = false

        // Last generation perf
        var lastQwen3TPS: Double? = nil
        var lastFastVLMTPS: Double? = nil

        static let empty = Snapshot()
    }

    private var refreshTimer: Timer?
    private var subscriberCount = 0
    private var cachedDiskFree: Int64 = 0
    private var lastDiskRefreshAt = Date.distantPast
    private var diskRefreshTask: Task<Void, Never>?
    private let diskRefreshInterval: TimeInterval = 30

    private init() {
        refresh()
    }

    func startObserving() {
        subscriberCount += 1
        guard refreshTimer == nil else { return }
        refresh()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in self?.refresh() }
        }
    }

    func stopObserving() {
        subscriberCount = max(0, subscriberCount - 1)
        if subscriberCount == 0 {
            refreshTimer?.invalidate()
            refreshTimer = nil
        }
    }

    func refresh() {
        var snap = Snapshot()

        snap.totalRAM       = Int64(ProcessInfo.processInfo.physicalMemory)
        snap.availableForML = MemoryAdvisor.availableMemoryForModel
        snap.freeRightNow   = MemoryAdvisor.processAvailableMemory
        snap.usedByApp      = MemoryAdvisor.physFootprint
        snap.hasIncreasedMemoryLimitEntitlement = MemoryAdvisor.hasIncreasedMemoryLimitEntitlement
        snap.lowPowerModeEnabled = ProcessInfo.processInfo.isLowPowerModeEnabled

        snap.diskFree           = cachedDiskFree
        snap.modelStorageUsed   = Self.directorySize(at: CoreAIModelStore.shared.modelDirectory)

        snap.device = UIDevice.current.modelIdentifier
        snap.os     = "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"

        if let metal = MTLCreateSystemDefaultDevice() {
            snap.metalDeviceName = metal.name
            snap.metalSupportsRayTracing = metal.supportsRaytracing
        } else {
            snap.metalDeviceName = "unavailable"
        }

        snap.supportsNeuralEngine = true
        snap.qwen3Ready    = CoreAIInferenceService.shared.isReady
        snap.fastVLMReady  = false
        snap.voiceEngineReady = false
        snap.lastQwen3TPS  = CodingAssistantService.shared.tokenRate > 0
            ? CodingAssistantService.shared.tokenRate
            : nil

        snapshot = snap
        refreshDiskCapacityIfNeeded()
    }

    /// Volume resource values can block while iOS refreshes filesystem state.
    /// Keep that work off the main actor and avoid repeating it on every
    /// three-second health refresh.
    private func refreshDiskCapacityIfNeeded() {
        guard diskRefreshTask == nil,
              Date().timeIntervalSince(lastDiskRefreshAt) >= diskRefreshInterval else { return }

        lastDiskRefreshAt = Date()
        diskRefreshTask = Task { [weak self] in
            let diskFree = await Task.detached(priority: .utility) {
                Self.freeDiskBytes()
            }.value
            guard !Task.isCancelled, let self else { return }

            cachedDiskFree = diskFree
            if snapshot.diskFree != diskFree {
                var updated = snapshot
                updated.diskFree = diskFree
                snapshot = updated
            }
            diskRefreshTask = nil
        }
    }

    nonisolated static func freeDiskBytes() -> Int64 {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let capacity = (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage
        return capacity ?? 0
    }

    private static func directorySize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            total += Int64(size)
        }
        return total
    }
}

// MARK: - UIDevice + model identifier

extension UIDevice {
    var modelIdentifier: String {
        var sys = utsname()
        uname(&sys)
        let identifier = withUnsafePointer(to: &sys.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(validatingUTF8: $0) ?? "" }
        }
        return identifier
    }
}

// MARK: - DeviceSafetyMonitor

@MainActor
final class DeviceSafetyMonitor: ObservableObject {

    static let shared = DeviceSafetyMonitor()

    @Published private(set) var thermalState: ProcessInfo.ThermalState
    @Published private(set) var lowPowerMode: Bool
    @Published private(set) var memoryWarningCount: Int = 0
    @Published private(set) var lastMemoryWarning: Date?
    @Published private(set) var isCharging: Bool = false
    @Published private(set) var batteryLevel: Float = -1

    /// Throttle at `.serious`, stop at `.critical` — Apple's thermal guidance.
    var shouldThrottle: Bool {
        guard AppSettings.shared.thermalWarningsEnabled else { return false }
        switch effectiveThermalState {
        case .serious, .critical: return true
        default:                  return false
        }
    }

    enum StopReason: Equatable {
        case thermal
        case memoryPressure

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

    var stopReason: StopReason? {
        if effectiveThermalState == .critical {
            return .thermal
        }
        if hasSustainedMemoryPressure {
            return .memoryPressure
        }
        return nil
    }

    var shouldStopHeavyWork: Bool { stopReason != nil }

    private static let memoryPressureWindow: TimeInterval = 10
    private static let memoryPressureThreshold = 3

    private var hasSustainedMemoryPressure: Bool {
        let cutoff = Date().addingTimeInterval(-Self.memoryPressureWindow)
        return memoryWarningTimestamps.filter { $0 >= cutoff }.count >= Self.memoryPressureThreshold
    }

    private var memoryWarningTimestamps: [Date] = []

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

    /// Hysteresis-filtered thermal state. Brief spikes shouldn't pop the UI;
    /// `.critical` applies immediately (imminent thermal shutdown).
    @Published private(set) var effectiveThermalState: ProcessInfo.ThermalState = .nominal
    private var pendingTransition: Task<Void, Never>?

    var statusColor: String? {
        switch effectiveThermalState {
        case .critical: return "red"
        case .serious:  return "orange"
        default:        return lowPowerMode ? "orange" : nil
        }
    }

    /// Suggested max-token cap for the current thermal state. Core AI models
    /// are small, so the schedule only clamps at genuine distress.
    var recommendedMaxTokens: Int {
        if effectiveThermalState == .critical { return 512 }
        guard AppSettings.shared.thermalWarningsEnabled else { return 4096 }
        switch effectiveThermalState {
        case .serious: return 1536
        default:       return 4096
        }
    }

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

    func clearThermalWarning() {
        pendingTransition?.cancel()
        effectiveThermalState = .nominal
        pendingTransition = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 45_000_000_000)
            guard !Task.isCancelled, let self else { return }
            let now = ProcessInfo.processInfo.thermalState
            self.thermalState = now
            self.scheduleEffectiveTransition(to: now)
        }
    }

    // MARK: - Keep-awake (idle timer)

    private var keepAwakeReasons: Set<String> = []

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

    private func scheduleEffectiveTransition(to new: ProcessInfo.ThermalState) {
        pendingTransition?.cancel()

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
            Diagnostics.shared.warning(
                "iOS memory warning #\(self.memoryWarningCount) · footprint=\(MemoryAdvisor.physFootprint.formattedBytes) · available=\(MemoryAdvisor.availableMemoryForModel.formattedBytes)",
                category: "memory"
            )
            let cutoff = now.addingTimeInterval(-Self.memoryPressureWindow)
            self.memoryWarningTimestamps.removeAll { $0 < cutoff }
            self.memoryWarningTimestamps.append(now)
        }
    }
}
