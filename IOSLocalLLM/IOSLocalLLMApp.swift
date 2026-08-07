import SwiftUI
import UIKit

@main
struct OnDeviceLASApp: App {
    @UIApplicationDelegateAdaptor(LASAppDelegate.self) private var appDelegate
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ZStack {
                // Shared ambient sample field so clear Liquid Glass has live
                // pixels to refract (CodeLens transparent-glass pattern).
                LiquidPinkBackdrop()

                Group {
                    if settings.hasSeenOnboarding {
                        LocalAPIServerView()
                            .transition(.opacity)
                    } else {
                        LASOnboardingView()
                            .transition(.opacity)
                    }
                }
            }
            .lasTheme()
        }
        .onChange(of: scenePhase) { _, newPhase in
            LASLifecycleController.shared.handle(newPhase)
        }
    }
}

final class LASAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        CrashReporter.shared.install()
        Diagnostics.shared.breadcrumb("On Device: LAS launch", category: "lifecycle")
        return true
    }

    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == BackgroundDownloadCoordinator.sessionIdentifier else {
            completionHandler()
            return
        }
        RuntimeLogCenter.emit(
            "Reattaching background model-download session",
            subsystem: "download"
        )
        Task { @MainActor in
            let coordinator = BackgroundDownloadCoordinator.shared
            coordinator.systemCompletionHandler = completionHandler
            coordinator.reattach()
        }
    }
}

/// Lifecycle policy for the server-only product.
///
/// The listener is deliberately stopped when the app backgrounds. This keeps
/// the local API opt-in and matches iOS's foreground execution boundary. Any
/// resident model is drained before the process gives the system its memory
/// back; the user can load the selected model again after returning.
@MainActor
final class LASLifecycleController {
    static let shared = LASLifecycleController()

    private var epoch: UInt64 = 0

    func handle(_ phase: ScenePhase) {
        epoch &+= 1
        let currentEpoch = epoch
        Task { [weak self] in
            guard let self else { return }
            await self.apply(phase, epoch: currentEpoch)
        }
    }

    private func apply(_ phase: ScenePhase, epoch: UInt64) async {
        guard epoch == self.epoch else { return }

        switch phase {
        case .active:
            guard AppSettings.shared.hasSeenOnboarding else { return }
            if AppSettings.shared.localAPIEnabled {
                await LocalAPIManager.shared.start()
            }
            guard epoch == self.epoch else { return }
            if AppSettings.shared.localAPIAutoLoadModel {
                CodingAssistantService.shared.startLoad()
            }
            guard epoch == self.epoch else { return }
        case .background:
            await LocalAPIManager.shared.stop()
            guard epoch == self.epoch else { return }
            CodingAssistantService.shared.cancelLoad()
            CodingAssistantService.shared.stopGeneration()
            await CodingAssistantService.shared.unloadAndWaitForCleanup()
            guard epoch == self.epoch else { return }
            await MLXGenerationGate.shared.clearCacheWhenIdle()
        case .inactive:
            break
        @unknown default:
            break
        }
    }
}
