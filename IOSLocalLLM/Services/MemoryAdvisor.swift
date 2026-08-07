import Foundation
import os
import Security
#if canImport(UIKit)
import UIKit
#endif

#if os(iOS) && !targetEnvironment(simulator) && !targetEnvironment(macCatalyst)
// Security.framework exports these on iOS, but the Swift overlay does not
// expose SecTask declarations. Keep the ABI bridge local and minimal so the
// running code signature—not the source .entitlements file—is authoritative.
@_silgen_name("SecTaskCreateFromSelf")
private func LASCreateSecurityTask(_ allocator: CFAllocator?) -> CFTypeRef?

@_silgen_name("SecTaskCopyValueForEntitlement")
private func LASCopyEntitlementValue(
    _ task: CFTypeRef,
    _ entitlement: CFString,
    _ error: UnsafeMutablePointer<Unmanaged<CFError>?>?
) -> CFTypeRef?
#endif

// MARK: - MemoryAdvisor
// Helps decide whether loading a given model is safe on the current device.
//
// We use ProcessInfo.physicalMemory as the headline number, but reserve
// roughly 30% for the OS + foreground apps. Models that exceed the remaining
// budget are flagged as "won't fit" — the UI can warn before load() is called.
//
// The numbers below are approximate working-set sizes during inference, not
// raw weight sizes. A 4-bit Qwen3-4B has ~2.3 GB of weights but typically
// peaks around 3.5–4 GB with the KV cache.

enum MemoryAdvisor {

    // MARK: - Load-time headroom
    //
    // `estimatedFootprint(for:)` returns a model's *peak* resident working set
    // — steady-state weights + KV cache + activations + the transient during
    // load (on-disk read, 4-bit unpack, Metal upload running concurrently).
    // The gate (`safetyBlocker`) therefore compares that peak directly against
    // the live per-process ceiling and adds only a small fixed reserve for
    // app/OS glue. It must NOT multiply the footprint again — doing so was the
    // "everything reports double the RAM it needs" bug: a downloaded model was
    // sized at on-disk × `workingSetOverhead` inside `estimatedFootprint` and
    // then multiplied a *second* time by the spike factor here, yielding
    // on-disk × ~2.5 and refusing 4B/8B models that comfortably fit.
    //
    // `loadSpikeMultiplier` is the single weights→peak factor used by callers
    // that start from *raw* weight bytes (e.g. `LensInferenceLoop`, which sizes
    // the VLM from `estimatedWeightBytes`). It is applied exactly once on that
    // path. Inside this type, `estimatedFootprint` already bakes the peak in.
    static let loadSpikeMultiplier = 1.6

    // Single weights→peak factor for models sized from their on-disk weights
    // (downloaded / imported). On-disk size ≈ quantized weights; the resident
    // peak adds KV cache + activations. 1.3× covers a typical context window
    // for 4-bit/8-bit MLX models, which mmap their weights (the on-disk read
    // does not add resident pages beyond the weights themselves).
    static let workingSetOverhead = 1.3

    // Fixed headroom reserved on top of a model's estimated peak, covering the
    // app's own baseline, the MLX/Metal runtime, and OS glue. A fixed reserve
    // (rather than a percentage of model size) avoids over-penalizing large
    // models — the transient is already inside the peak estimate.
    static let loadHeadroomReserve: Int64 = 500_000_000

    // Reserve used by paging-capable runtimes in edge / developer mode.
    // MLX always keeps `loadHeadroomReserve`: its weights materialize eagerly.
    static let edgeHeadroomReserve: Int64 = 0

    /// Rounding and kernel-accounting jitter can make a model with a previously
    /// observed successful peak appear short by a few dozen MB. A disk-size
    /// estimate is NOT sufficient: Build 116 proved that `weights × 1.3` can
    /// understate an imported Gemma load transient and iOS will jetsam before
    /// MLX reports allocator progress. The grace therefore requires a
    /// calibrated successful peak in addition to the entitlement.
    static let entitledNearFitGrace: Int64 = 128 * 1_024 * 1_024

    // Conservative footprint assumed when a model can't be sized (custom /
    // imported repos with no preset match and no readable on-disk folder).
    // 3.5 GB covers an unknown mid/large model; with `loadHeadroomReserve` an
    // unsized model needs ~4 GB free to load — strict, since "we don't know
    // its size" should fail safe, but no longer wildly over-stated.
    static let unknownFootprintFloor: Int64 = 3_500_000_000

    // MARK: - Post-pressure load cooldown
    //
    // After the kernel reports CRITICAL memory pressure, MemoryPressureCoordinator
    // dumps model weights. For a short window afterward we refuse new HEAVY
    // loads: reloading immediately races iOS while it is still reclaiming, which
    // leaves the dumped model AND its reload both resident for a moment — the
    // exact spike that gets the process Jetsam-killed. Small recovery models
    // that comfortably fit current headroom are still allowed so the user is
    // never fully stuck.

    static let pressureCooldown: TimeInterval = 20
    /// A model at/under this peak may still load during the cooldown if it fits
    /// the live headroom — lets a small camera VLM recover the lens immediately.
    static let smallRecoveryModelCeiling: Int64 = 1_073_741_824   // 1 GB

    /// When set and in the future, heavy loads are on cooldown. MainActor-only;
    /// written by `notePressureDump()` and read by `safetyBlocker()`.
    @MainActor private(set) static var pressureLoadBlockedUntil: Date?

    /// Open the post-pressure cooldown. Called after an emergency weight dump.
    @MainActor static func notePressureDump() {
        pressureLoadBlockedUntil = Date().addingTimeInterval(pressureCooldown)
    }

    /// Seconds remaining on the post-critical-pressure cooldown, or nil when
    /// no cooldown is active. Single home for the date math so `safetyBlocker`
    /// and the VLM load gates (LensInferenceLoop.switchTo) enforce the same
    /// window — a heavy VLM reload must not race the kernel's reclaim either.
    @MainActor static var pressureCooldownRemaining: TimeInterval? {
        guard let until = pressureLoadBlockedUntil, until.timeIntervalSinceNow > 0 else {
            return nil
        }
        return until.timeIntervalSinceNow
    }

    // MARK: - Device

    /// Total physical RAM in bytes.
    static var deviceTotalRAM: Int64 {
        Int64(ProcessInfo.processInfo.physicalMemory)
    }

    /// RAM available for app use after a 30% headroom for OS + other apps.
    static var availableRAM: Int64 {
        Int64(Double(deviceTotalRAM) * 0.70)
    }

    /// Live per-process memory headroom in bytes — how much more this app can
    /// allocate before iOS kills it for memory. On iOS this ceiling sits well
    /// below physical RAM (a 6 GB device often gives a process only ~3 GB even
    /// with the increased-memory entitlement), so it's the real OOM limit.
    /// Reports 0 when unavailable (e.g. some simulator conditions).
    static var processAvailableMemory: Int64 {
        Int64(os_proc_available_memory())
    }

    /// Device RAM not resident in THIS process (deviceTotalRAM − our own
    /// `resident_size`), in bytes. NOT "free physical memory" — other apps'
    /// and the OS's pages still count as available here. Useful only to
    /// recover an approximate resident size (`deviceTotalRAM − this`), e.g.
    /// the `processMemoryCeiling` fallback and the diagnostics snapshots.
    /// Reports 0 on failure.
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
    /// the kernel uses to decide `os_proc_available_memory()` and to Jetsam the
    /// app. Reports 0 on failure.
    ///
    /// This is deliberately NOT `mach_task_basic_info.resident_size`. The two
    /// diverge for GPU / IOKit allocations: an MLX/Metal model's weight buffers
    /// count toward `phys_footprint` (and therefore against the memory limit)
    /// but are largely absent from `resident_size`. Mixing the two — as the old
    /// `processMemoryCeiling` did (`os_proc_available_memory() + resident_size`)
    /// — made the ceiling read ~1 GB+ LOW whenever a VLM was resident, because
    /// `os_proc_available_memory()` had already subtracted the GPU footprint
    /// while `resident_size` never added it back. That's the "open the Lens
    /// VLM, switch to Assistant, Qwen3-4B is suddenly too large for this
    /// device" bug: the ceiling was polluted by the just-used VLM's GPU memory.
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

    // MARK: - Entitlement-aware ceiling
    //
    // IOSLocalLLM ships the `increased-memory-limit` entitlement. On some devices
    // `os_proc_available_memory()` under-reports, so we fuse that signal with
    // an empirical per-tier fraction of physical RAM. The MLX allocator uses
    // TokenAI's validated 0.74 physical-RAM limit; the process cap below keeps
    // the UI/admission math in the same envelope without allowing the cache to
    // grow unbounded. iPad and Mac keep the scalable estimate.

    /// Fraction of physical RAM the app can realistically use as a hard ceiling
    /// on this device class. Tuned empirically against the entitled per-process
    /// limit (it sits well below 100% of device RAM even with the entitlement).
    /// These fractions provide the synthetic candidate. `clampedProcessCeiling`
    /// applies the separately validated iPhone ceiling afterward.
    private static var entitlementCeilingFraction: Double {
        switch DeviceTierAdvisor.current {
        case .lite, .entry: return 0.55   // ≤4 GB — keep very tight
        case .mid:          return 0.62   // 6 GB
        case .pro:          return 0.72   // 8 GB  → ~6.2 GB usable
        case .max:          return 0.76   // 12 GB+ (12 GiB) → ~9.8 GB usable
        }
    }

    /// Hard reserve always kept for the OS + other apps, regardless of tier.
    private static let physicalOSReserve: Int64 = 1_500_000_000

    /// Conservative process cap for iPhones below the 12 GB Max tier.
    static let maximumIPhoneProcessCeiling: Int64 = 6_200_000_000

    /// 12 GB Pro Max devices have a materially larger entitled process budget.
    /// Keep the admission envelope aligned with TokenAI's 0.74 MLX allocator
    /// limit (about 9.1 GB on the reference device) while retaining a small
    /// rounding margin for UI and runtime bookkeeping.
    static let maximumHighMemoryIPhoneProcessCeiling: Int64 = 9_200_000_000
    static let highMemoryIPhoneRAMThreshold: Int64 = 10_000_000_000

    /// Whether the entitlement is present in the signature iOS is actually
    /// running. Declaring the key in the source entitlements file is not
    /// sufficient: an unsigned IPA, or a sideloading profile that does not
    /// grant it, runs with the normal process memory limit.
    static var hasIncreasedMemoryLimitEntitlement: Bool {
        #if os(iOS) && !targetEnvironment(simulator) && !targetEnvironment(macCatalyst)
        guard let task = LASCreateSecurityTask(nil),
              let value = LASCopyEntitlementValue(
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
    /// Kept injectable so the real-device Jetsam regression is unit-testable
    /// without depending on the test host's RAM or interface idiom.
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
    /// The synthetic entitlement estimate must never override the kernel when
    /// the installed profile did not grant increased memory, or the UI admits
    /// a load that iOS will terminate as soon as weights are materialized.
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
    /// limit iOS enforces), in bytes — independent of what is loaded right now.
    ///
    /// Fuses two signals and trusts the larger:
    ///   • Kernel: `os_proc_available_memory()` + current `phys_footprint`.
    ///     Both use the identical footprint accounting, so the sum recovers the
    ///     limit and is STABLE regardless of what is resident. (The previous
    ///     version added `resident_size`, which omits GPU/Metal buffers, so the
    ///     ceiling sagged by a resident VLM's GPU footprint — the "open Lens,
    ///     switch to Assistant, 4B won't load" regression. See `physFootprint`.)
    ///   • Entitlement: `physicalRAM × tierFraction`. Recovers the headroom the
    ///     kernel signal hides on entitled devices.
    /// Clamped to `physicalRAM − 1.5 GB`; iPhone is additionally capped at its
    /// entitlement-aware tier budget.
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

    /// Live memory the app can realistically still allocate RIGHT NOW, in bytes
    /// — the figure the load gate compares a model's footprint against. Like
    /// `processMemoryCeiling` it fuses the kernel headroom with the
    /// entitlement-fraction estimate. The final headroom is always derived
    /// from `processMemoryCeiling`, so neither optimistic signal can exceed the
    /// validated platform budget.
    static var availableMemoryForModel: Int64 {
        let footprint = physFootprint
        return max(0, processMemoryCeiling - footprint)
    }

    // MARK: - Device fit (for suggestion badges)

    enum Fit {
        case fits        // comfortably within the ceiling, with reserve to spare
        case tight       // fits, but with little headroom — may throttle / fail under pressure
        case over        // exceeds the ceiling — won't load on this device

        var label: String {
            switch self {
            case .fits:  return "fits"
            case .tight: return "tight"
            case .over:  return "too big"
            }
        }
    }

    /// Classifies whether a model of the given peak footprint fits this device.
    ///
    /// Measured against live `processAvailableMemory`. On the Models tab the app
    /// has unloaded every model (ContentView does this on entering the tab), so
    /// that figure is the true headroom a fresh load can use — the honest answer
    /// to "will this run without an OOM crash." `.fits` keeps the full reserve;
    /// `.tight` would load only with the reserve dropped (edge mode); `.over`
    /// cannot load at all.
    static func fit(forFootprint footprint: Int64) -> Fit {
        guard footprint > 0 else { return .fits }
        let avail = availableMemoryForModel > 0 ? availableMemoryForModel : availableRAM
        if footprint + loadHeadroomReserve <= avail { return .fits }
        if footprint <= avail { return .tight }
        return .over
    }

    /// Convenience: fit verdict for a model id, using its estimated peak.
    static func fit(forModelID modelID: String) -> Fit {
        let footprint = estimatedFootprint(for: modelID)
        return fit(forFootprint: footprint > 0 ? footprint : unknownFootprintFloor)
    }

    // MARK: - Model footprints (working-set estimates)

    /// Best-effort working-set estimate per model, in bytes.
    /// Falls back to AssistantModelCatalog preset metadata and finally to an
    /// on-disk-size × 1.6 estimate so downloaded / imported models are no
    /// longer reported as "0 — fits anywhere".
    static func estimatedFootprint(for modelID: String) -> Int64 {
        // 1. Built-ins with hand-tuned numbers
        switch modelID {
        case "qwen3-1.7b":       return 1_500_000_000   // ~1.5 GB peak
        case "qwen3-4b":         return 3_800_000_000   // ~3.8 GB peak
        case "qwen3-8b":         return 6_500_000_000   // ~6.5 GB peak
        case FastVLMService.modelID: return 1_400_000_000   // ~1.4 GB peak
        case "kittentts-nano":   return 250_000_000     // ~250 MB
        case "kittentts-mini":   return 700_000_000     // ~700 MB
        case "kokoro":           return 400_000_000     // ~400 MB
        default: break
        }
        // 2. Preset bytes from the assistant catalog
        if let preset = AssistantModelCatalog.model(forID: modelID) {
            return preset.approxRAMBytes
        }
        // 3. Custom/downloaded/imported — parse the prefix and look up the
        //    on-disk repo size. Peak working set ≈ weights × workingSetOverhead
        //    (KV cache + activations). This is the *final* peak estimate; the
        //    gate does not multiply it again.
        if let repoID = nonPresetRepoID(from: modelID),
           let onDisk = onDiskWeightsSize(forRepoID: repoID), onDisk > 0 {
            return estimatedPeakFootprint(
                weightBytes: onDisk,
                profile: ModelCapabilityProfile.resolve(repoID: repoID)
            )
        }
        return 0
    }

    /// Includes the fully populated, bounded KV cache in the peak estimate.
    /// The weight-only multiplier remains the floor for models whose cache is
    /// small; large-context profiles use the explicit cache estimate instead
    /// of relying on a hidden generation-time allocation.
    static func estimatedPeakFootprint(
        weightBytes: Int64,
        profile: ModelCapabilityProfile
    ) -> Int64 {
        guard weightBytes > 0 else { return 0 }
        let weightWorkingSet = Int64(Double(weightBytes) * workingSetOverhead)
        let cacheAwarePeak = weightBytes + max(0, profile.estimatedKVCacheBytes)
        return max(weightWorkingSet, cacheAwarePeak)
    }

    /// Pulls the bare repoID out of `downloaded:…`, `imported:…`, `custom:…`.
    private static func nonPresetRepoID(from modelID: String) -> String? {
        let unwrapped = LocalModelRegistry.unwrapAssistantSelectionID(modelID)
        return unwrapped == modelID ? nil : unwrapped
    }

    // On-disk size cache. `allocatedSizeOfDirectory` recursively sums every
    // file in a multi-GB model folder — far too expensive to run per row on
    // every scroll frame, which is exactly what `estimatedFootprint` did once
    // it started driving the Models-tab fit badges. The result is stable while
    // browsing (a folder doesn't change size unless a download completes), so
    // we memoize it. `-1` is the "checked, nothing on disk" sentinel so a
    // not-yet-downloaded repo isn't re-walked every frame either.
    private static let _diskSizeLock = NSLock()
    nonisolated(unsafe) private static var _diskSizeCache: [String: Int64] = [:]

    /// Drops the on-disk size cache. Call when the installed set changes (a
    /// download finishes, a model is deleted) so freshly-sized repos are
    /// re-measured on next access. Cheap; safe to call liberally.
    static func invalidateFootprintCache() {
        _diskSizeLock.lock()
        _diskSizeCache.removeAll(keepingCapacity: true)
        _diskSizeLock.unlock()
    }

    /// On-disk allocated size of the repo's local copy, in bytes. Memoized —
    /// see `_diskSizeCache`. Returns nil if the directory doesn't exist.
    private static func onDiskWeightsSize(forRepoID repoID: String) -> Int64? {
        _diskSizeLock.lock()
        if let cached = _diskSizeCache[repoID] {
            _diskSizeLock.unlock()
            return cached < 0 ? nil : cached
        }
        _diskSizeLock.unlock()

        let docs = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask)[0]
        let flattened = repoID.replacingOccurrences(of: "/", with: "_")
        let tail = repoID.split(separator: "/").last.map(String.init) ?? repoID
        let dashed = repoID.replacingOccurrences(of: "/", with: "--")
        // Imports use `HFModels/<tail>` while catalog downloads generally use
        // `HFModels/<author>_<repo>`. The old single-path probe knew only the
        // latter, so a large imported model was treated as an unknown 3.5 GB
        // model and admitted into MLX even when its actual weights exceeded
        // the process ceiling. Probe the same storage layouts as the loader.
        let candidates = [
            docs.appendingPathComponent("HFModels", isDirectory: true)
                .appendingPathComponent(flattened, isDirectory: true),
            docs.appendingPathComponent("HFModels", isDirectory: true)
                .appendingPathComponent(tail, isDirectory: true),
            docs.appendingPathComponent("LLMModels", isDirectory: true)
                .appendingPathComponent(tail, isDirectory: true),
            docs.appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("models", isDirectory: true)
                .appendingPathComponent(repoID, isDirectory: true),
            docs.appendingPathComponent("huggingface", isDirectory: true)
                .appendingPathComponent("hub", isDirectory: true)
                .appendingPathComponent("models--\(dashed)", isDirectory: true),
        ]
        let result = candidates.lazy.compactMap { candidate -> Int64? in
            guard FileManager.default.fileExists(atPath: candidate.path) else {
                return nil
            }
            return try? FileManager.default.allocatedSizeOfDirectory(at: candidate)
        }.first(where: { $0 > 0 })

        _diskSizeLock.lock()
        _diskSizeCache[repoID] = result ?? -1
        _diskSizeLock.unlock()
        return result
    }

    // MARK: - Verdicts

    enum Verdict {
        case fitsComfortably         // < 60% of available
        case marginal(String)        // 60–90% of available; warn but allow
        case wontFit(String)         // > 90% — block

        var color: String {
            switch self {
            case .fitsComfortably: return "green"
            case .marginal:        return "orange"
            case .wontFit:         return "red"
            }
        }

        var isBlocking: Bool {
            if case .wontFit = self { return true }
            return false
        }
    }

    /// Returns a verdict for loading `modelID` in addition to whatever is
    /// already considered loaded (sum the estimates).
    static func verdict(for modelID: String, alreadyLoaded: [String] = []) -> Verdict {
        let footprint = estimatedFootprint(for: modelID)
        let loadedFootprints = alreadyLoaded.map { estimatedFootprint(for: $0) }
        return verdict(forFootprint: footprint, alreadyLoadedFootprints: loadedFootprints)
    }

    /// Verdict for callers that already have a hand-tuned peak footprint (image
    /// generation / curated VLM catalog). Keeps picker badges aligned with the
    /// same live-memory requirement enforced by the load gates.
    static func verdict(
        forFootprint footprint: Int64,
        alreadyLoadedFootprints: [Int64] = []
    ) -> Verdict {
        guard footprint > 0 else { return .fitsComfortably }

        let combined = footprint + alreadyLoadedFootprints.reduce(0, +)
        let procAvail = availableMemoryForModel
        let budget = procAvail > 0 ? procAvail : availableRAM
        let reserve = AppSettings.shared.showEdgeModels ? edgeHeadroomReserve : loadHeadroomReserve
        let needed = combined + reserve

        if needed > budget {
            return .wontFit(String(
                format: "Needs ~%.1f GB but only ~%.1f GB is available to the app right now. Close other apps and retry.",
                Double(needed) / 1_000_000_000,
                Double(budget) / 1_000_000_000
            ))
        }
        if Double(needed) / Double(max(1, budget)) > 0.85 {
            return .marginal("This model fits, but uses most of the app's ~\(budget.formattedBytes) current memory headroom. Close other apps if loading is interrupted.")
        }
        return .fitsComfortably
    }

    /// Verdict considering models currently flagged as loaded.
    /// Must be called from MainActor since it reads service singletons.
    @MainActor
    static func verdictWithCurrentlyLoaded(for modelID: String) -> Verdict {
        let footprint = estimatedFootprint(for: modelID)
        return verdictWithCurrentlyLoaded(forFootprint: footprint, excludingModelID: modelID)
    }

    /// Same as `verdictWithCurrentlyLoaded(for:)`, but for callers with a
    /// curated footprint that is not keyed in `estimatedFootprint`.
    @MainActor
    static func verdictWithCurrentlyLoaded(
        forFootprint footprint: Int64,
        excludingModelID modelID: String? = nil
    ) -> Verdict {
        var loaded: [String] = []
        if CodingAssistantService.shared.state == .ready {
            // Whatever the assistant currently holds, not a hardcoded ID.
            loaded.append(CodingAssistantService.shared.activeModel.id)
        }
        if FastVLMService.shared.componentStatus.canGenerate { loaded.append(FastVLMService.modelID) }
        let others = loaded.filter { loadedID in
            guard let modelID else { return true }
            return loadedID != modelID
        }
        return verdict(
            forFootprint: footprint,
            alreadyLoadedFootprints: others.map { estimatedFootprint(for: $0) }
        )
    }

    // MARK: - Combined device-safety verdict

    /// Combined verdict: RAM + live free memory + thermal state + low-power.
    /// Use this before kicking off a model load. Returns nil when it's safe.
    ///
    /// Thermal handling here follows Apple's actual guidance: only block
    /// at `.critical`. The previous threshold (block >1.5 GB models at
    /// `.serious`) refused Qwen3-4B (~2.3 GB) the moment the device got
    /// even moderately warm, which on an A19 Pro is the working state
    /// during sustained inference. iOS's own thermal scheduler already
    /// throttles CPU/GPU clocks at `.serious`; we don't need to layer a
    /// hard refuse on top.
    @MainActor
    static func safetyBlocker(
        for modelID: String,
        allowTightFit: Bool = false,
        runtime: ModelRuntime? = nil,
        allowUnsafeMemoryLoad: Bool = false,
        measuredFootprintBytes: Int64? = nil
    ) -> String? {
        // Callers that can start MLX work must await
        // `MLXGenerationGate.clearCacheWhenIdle()` before entering this
        // synchronous measurement. A state-based "nothing is generating"
        // check is racy: a native load or a just-cancelled Metal command can
        // still be live after the published UI state changes.

        // 1. Thermal: only refuse at .critical. .serious is workable on
        //    modern silicon and iOS already does its own backoff there.
        if DeviceSafetyMonitor.shared.thermalState == .critical {
            return "Device is too hot to safely load this model. Let it cool for a minute, then retry."
        }

        // An unentitled sideload cannot opt out of the kernel's hard process
        // limit. Enforce this before the experimental-memory bypass so an MLX
        // load cannot knowingly enter a guaranteed Jetsam path.
        let measuredFootprint = max(
            0,
            measuredFootprintBytes ?? estimatedFootprint(for: modelID)
        )

        if runtime == .mlx, !hasIncreasedMemoryLimitEntitlement {
            let footprint = measuredFootprint
            let assumed = footprint > 0 ? footprint : unknownFootprintFloor
            let reserve = (allowTightFit && runtime != .mlx)
                ? edgeHeadroomReserve
                : loadHeadroomReserve
            let needed = assumed + reserve
            let kernelBudget = availableMemoryForModel
            if kernelBudget > 0, needed > kernelBudget {
                return hardCeilingMessage(
                    neededBytes: needed,
                    ceilingBytes: processMemoryCeiling,
                    runtime: runtime,
                    lowMemoryEnabled: allowTightFit,
                    hasIncreasedMemoryEntitlement: false
                )
            }
        }

        // A user-confirmed experimental attempt bypasses memory-capacity
        // admission for this call only. Critical thermal protection remains:
        // adding a known-oversized allocation while iOS is already throttling
        // at its highest level is not a useful model test.
        if allowUnsafeMemoryLoad {
            return nil
        }

        // 1b. Post-pressure cooldown. iOS recently hit CRITICAL memory pressure
        //     and we dumped weights; refuse heavy reloads briefly so we don't
        //     race the kernel's reclaim into a Jetsam kill. A small model that
        //     comfortably fits the live headroom is still allowed to recover.
        if let remaining = pressureCooldownRemaining {
            let footprint = measuredFootprint
            let assumed = footprint > 0 ? footprint : unknownFootprintFloor
            // A model may load during the cooldown if it COMFORTABLY fits the
            // live headroom right now, full reserve kept. The earlier gate also
            // required `assumed <= smallRecoveryModelCeiling` (1 GB) — but every
            // shipping model is ≥1.4 GB, so NO real model could ever recover and
            // the assistant/lens was dead for the whole 20s window even after
            // the dump had freed plenty of memory. The full reserve is the
            // safety margin against the in-flight reclaim race; the size ceiling
            // added nothing but the dead-end.
            let fitsAsRecovery = assumed + loadHeadroomReserve <= availableMemoryForModel
            if !fitsAsRecovery {
                let secs = max(1, Int(remaining.rounded(.up)))
                return "iOS just reported a memory-pressure spike. Wait ~\(secs)s for it to recover memory, then retry."
            }
        }

        // 2. RAM verdict (physical-memory heuristic). Adjustable: when the
        //    user relaxes the strict gate, skip this conservative check but
        //    still enforce the hard per-process ceiling below — that one
        //    reflects memory the kernel will actually grant, so ignoring it
        //    means a near-certain crash.
        if AppSettings.shared.strictMemoryGate {
            if case .wontFit(let msg) = verdictWithCurrentlyLoaded(
                forFootprint: measuredFootprint,
                excludingModelID: modelID
            ) {
                let strictBudget = availableMemoryForModel
                let strictReserve = AppSettings.shared.showEdgeModels && runtime != .mlx
                    ? edgeHeadroomReserve
                    : loadHeadroomReserve
                let strictNeeded = measuredFootprint + strictReserve
                let shouldBlock = strictVerdictBlocks(
                    neededBytes: strictNeeded,
                    availableBytes: strictBudget,
                    measuredPeakBytes: measuredFootprint,
                    allowTightFit: allowTightFit,
                    runtime: runtime,
                    hasEntitlement: hasIncreasedMemoryLimitEntitlement,
                    peakIsCalibrated: false
                )
                if shouldBlock {
                    return msg
                }
                if strictNeeded > strictBudget {
                    Diagnostics.shared.breadcrumb(
                        "strict verdict near-fit override · deficit=\(strictNeeded - strictBudget) · peak=\(measuredFootprint) · available=\(strictBudget)",
                        category: "memory"
                    )
                }
            }
        }

        // 3. Live per-process ceiling — the real OOM killer on iOS. The verdict
        //    above reasons about *physical* RAM (70% of device total), but iOS
        //    caps each process well below that. A model can pass the physical
        //    heuristic (e.g. 3.8 GB on a "4.2 GB usable" 6 GB device) yet still
        //    exceed the ~3 GB the kernel will actually hand this process — that
        //    mismatch is the classic "passes the gate, crashes on load" case.
        //    This backstop refuses outright before the allocation that would
        //    SIGKILL the app. When the static footprint is unknown (custom /
        //    imported models that don't match a preset or an on-disk folder),
        //    fall back to a conservative floor so an unsized large model can't
        //    slip through with needed == 0.
        // Use the entitlement-aware live figure, NOT raw
        // os_proc_available_memory(): on entitled devices the latter
        // under-reports and refuses models that comfortably fit (the core
        // "device has RAM but the model won't load" report).
        let hardCeilingFootprint = measuredFootprint > 0
            ? measuredFootprint
            : unknownFootprintFloor
        // MLX low-memory mode bounds allocator/cache retention but cannot page
        // eagerly materialized weights. Do not drop the load reserve merely
        // because that mode is enabled; Build 116 demonstrated a jetsam during
        // local weight materialization before MLX could report progress.
        let hardCeilingReserve = ((AppSettings.shared.showEdgeModels || allowTightFit)
            && runtime != .mlx)
            ? edgeHeadroomReserve
            : loadHeadroomReserve
        let hardCeilingNeeded = hardCeilingFootprint + hardCeilingReserve
        let hardCeiling = processMemoryCeiling
        if hardCeiling > 0, hardCeilingNeeded > hardCeiling {
            let nearFit = entitledNearFitAllowed(
                neededBytes: hardCeilingNeeded,
                availableBytes: hardCeiling,
                measuredPeakBytes: hardCeilingFootprint,
                runtime: runtime,
                hasEntitlement: hasIncreasedMemoryLimitEntitlement,
                peakIsCalibrated: false
            )
            if !nearFit {
                return hardCeilingMessage(
                    neededBytes: hardCeilingNeeded,
                    ceilingBytes: hardCeiling,
                    runtime: runtime,
                    lowMemoryEnabled: allowTightFit,
                    hasIncreasedMemoryEntitlement: hasIncreasedMemoryLimitEntitlement
                )
            }
            Diagnostics.shared.breadcrumb(
                "entitled near-fit grace · deficit=\(hardCeilingNeeded - hardCeiling) · peak=\(hardCeilingFootprint) · ceiling=\(hardCeiling)",
                category: "memory"
            )
        }

        let procAvail = availableMemoryForModel
        if procAvail > 0 {
            let footprint = measuredFootprint
            let assumed = footprint > 0 ? footprint : unknownFootprintFloor
            // `assumed` is already the peak resident working set (it includes
            // the load transient — see `estimatedFootprint`). Add only a fixed
            // reserve for app/runtime/OS glue. Multiplying the peak again here
            // is what previously inflated every estimate ~2.5× and refused
            // models that comfortably fit the per-process ceiling.
            //
            // Edge / developer mode may drop the reserve for paging-capable
            // runtimes. MLX keeps it because weights cannot be paged and the
            // disk-derived peak is not a calibrated successful measurement.
            let reserve = ((AppSettings.shared.showEdgeModels || allowTightFit)
                && runtime != .mlx)
                ? edgeHeadroomReserve
                : loadHeadroomReserve
            let needed = assumed + reserve
            if procAvail < needed {
                let nearFit = entitledNearFitAllowed(
                    neededBytes: needed,
                    availableBytes: procAvail,
                    measuredPeakBytes: assumed,
                    runtime: runtime,
                    hasEntitlement: hasIncreasedMemoryLimitEntitlement,
                    peakIsCalibrated: false
                )
                if nearFit {
                    Diagnostics.shared.breadcrumb(
                        "entitled live near-fit grace · deficit=\(needed - procAvail) · peak=\(assumed) · available=\(procAvail)",
                        category: "memory"
                    )
                    return nil
                }
                // Distinguish "too big for this device, period" from "would
                // fit if you freed up what's resident right now." The hard
                // ceiling (`processMemoryCeiling`) is what this process can
                // ever allocate with nothing else loaded; if `needed` exceeds
                // even that, telling the user to "unload other models" is
                // wrong — no amount of freeing helps, they need a smaller
                // model. This is the common "Qwen2.5 7B won't load" case on
                // 6 GB devices whose per-process ceiling sits below ~5 GB.
                let ceiling = processMemoryCeiling
                if needed > ceiling {
                    return hardCeilingMessage(
                        neededBytes: needed,
                        ceilingBytes: ceiling,
                        runtime: runtime,
                        lowMemoryEnabled: allowTightFit,
                        hasIncreasedMemoryEntitlement: hasIncreasedMemoryLimitEntitlement
                    )
                }
                return String(
                    format: "Not enough memory to load this model safely (~%.1f GB needed, only ~%.1f GB available to the app). Unload other models or close some apps, then retry.",
                    Double(needed) / 1_000_000_000,
                    Double(procAvail) / 1_000_000_000
                )
            }
        }
        return nil
    }

    static nonisolated func entitledNearFitAllowed(
        neededBytes: Int64,
        availableBytes: Int64,
        measuredPeakBytes: Int64,
        runtime: ModelRuntime?,
        hasEntitlement: Bool,
        peakIsCalibrated: Bool
    ) -> Bool {
        guard runtime == .mlx,
              hasEntitlement,
              peakIsCalibrated,
              neededBytes > availableBytes,
              availableBytes > 0,
              measuredPeakBytes > 0,
              measuredPeakBytes <= availableBytes else {
            return false
        }
        return neededBytes - availableBytes <= entitledNearFitGrace
    }

    /// Resolves the conservative verdict gate before the hard/live ceiling
    /// checks run. This must mirror the same entitled near-fit policy used by
    /// those later checks; otherwise `verdictWithCurrentlyLoaded` returns its
    /// reserve-inclusive warning first and makes the near-fit path unreachable.
    ///
    /// `allowTightFit` retains its existing edge-mode behavior: the reserve may
    /// be dropped only when the measured model peak itself fits. Normal mode is
    /// narrower and permits only the bounded entitled reserve deficit.
    static nonisolated func strictVerdictBlocks(
        neededBytes: Int64,
        availableBytes: Int64,
        measuredPeakBytes: Int64,
        allowTightFit: Bool,
        runtime: ModelRuntime?,
        hasEntitlement: Bool,
        peakIsCalibrated: Bool
    ) -> Bool {
        guard neededBytes > availableBytes else { return false }

        let peakFits = availableBytes > 0
            && measuredPeakBytes > 0
            && measuredPeakBytes <= availableBytes
        if allowTightFit,
           peakFits,
           runtime != .mlx || peakIsCalibrated {
            return false
        }

        return !entitledNearFitAllowed(
            neededBytes: neededBytes,
            availableBytes: availableBytes,
            measuredPeakBytes: measuredPeakBytes,
            runtime: runtime,
            hasEntitlement: hasEntitlement,
            peakIsCalibrated: peakIsCalibrated
        )
    }

    /// Explains a hard process-ceiling refusal without implying that the
    /// low-memory switch can page every model format. MLX can reduce retained
    /// allocator/KV cache, but its full weights still count against the iOS
    /// process limit. Only llama.cpp's GGUF path can keep weights file-backed
    /// and let iOS reclaim clean pages.
    static nonisolated func hardCeilingMessage(
        neededBytes: Int64,
        ceilingBytes: Int64,
        runtime: ModelRuntime?,
        lowMemoryEnabled: Bool,
        hasIncreasedMemoryEntitlement: Bool = true
    ) -> String {
        let neededGB = Double(neededBytes) / 1_000_000_000
        let ceilingGB = Double(ceilingBytes) / 1_000_000_000
        if runtime == .mlx, !hasIncreasedMemoryEntitlement {
            return String(
                format: "This installed build is missing the iOS Increased Memory Limit entitlement. The MLX model needs ~%.1f GB, but iOS currently grants this app only ~%.1f GB. Re-sign with a provisioning profile that grants com.apple.developer.kernel.increased-memory-limit, or choose a smaller MLX/GGUF model.",
                neededGB,
                ceilingGB
            )
        }
        if runtime == .mlx, lowMemoryEnabled {
            return String(
                format: "This MLX model needs ~%.1f GB, but this app can use ~%.1f GB on this device. Low-memory mode cannot page MLX weights. Choose a smaller MLX model (4B or 1.7B), or import a GGUF quantization for storage-backed paging.",
                neededGB,
                ceilingGB
            )
        }
        return String(
            format: "This model is too large for this device (~%.1f GB needed, but the app can use at most ~%.1f GB here). Pick a smaller model — a 4B or 1.7B fits comfortably.",
            neededGB,
            ceilingGB
        )
    }

    /// Capacity failures cannot be fixed by retrying the same model. Keep the
    /// classification beside the messages it recognizes so recovery UI does
    /// not depend on a particular model name or byte estimate.
    static nonisolated func isHardCapacityFailure(_ message: String) -> Bool {
        let normalized = message.lowercased()
        return normalized.contains("this mlx model needs")
            || normalized.contains("model is too large for this device")
            || normalized.contains("missing the ios increased memory limit entitlement")
    }

    // MARK: - Device summary

    static var deviceSummary: String {
        "\(deviceTotalRAM.formattedBytes) total · ~\(availableRAM.formattedBytes) usable for ML"
    }
}
