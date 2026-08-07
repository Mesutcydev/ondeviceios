import SwiftUI
import UIKit

@main
struct CoreAIOnDeviceLASApp: App {
    @UIApplicationDelegateAdaptor(CoreAIAppDelegate.self) private var appDelegate
    @ObservedObject private var settings = AppSettings.shared
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ZStack {
                LiquidPinkBackdrop()

                Group {
                    if settings.hasSeenOnboarding {
                        CoreAIDashboardView()
                            .transition(.opacity)
                    } else {
                        CoreAIOnboardingView()
                            .transition(.opacity)
                    }
                }
            }
            .lasTheme()
        }
        .onChange(of: scenePhase) { _, phase in
            CoreAILifecycleController.shared.handle(phase)
        }
    }
}

final class CoreAIAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        CrashReporter.shared.install()
        Diagnostics.shared.breadcrumb("Core AI: LAS launch", category: "lifecycle")
        RuntimeLogCenter.emit("Core AI: LAS launch", subsystem: "lifecycle")
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
            "Reattaching background Core AI model-download session",
            subsystem: "download"
        )
        Task { @MainActor in
            let coordinator = BackgroundDownloadCoordinator.shared
            coordinator.systemCompletionHandler = completionHandler
            coordinator.reattach()
        }
    }
}

@MainActor
final class CoreAILifecycleController {
    static let shared = CoreAILifecycleController()
    private var epoch: UInt64 = 0

    func handle(_ phase: ScenePhase) {
        epoch &+= 1
        let current = epoch
        Task { [weak self] in
            guard let self else { return }
            await self.apply(phase, epoch: current)
        }
    }

    private func apply(_ phase: ScenePhase, epoch: UInt64) async {
        guard epoch == self.epoch else { return }
        switch phase {
        case .active:
            CrashReporter.shared.markRunning()
            guard AppSettings.shared.hasSeenOnboarding else { return }
            if AppSettings.shared.localAPIEnabled {
                await LocalAPIManager.shared.start()
            }
            if AppSettings.shared.localAPIAutoLoadModel {
                CodingAssistantService.shared.startLoad()
            }
        case .background:
            Diagnostics.shared.breadcrumb("scene background · epoch=\(epoch)", category: "lifecycle")
            // Clear the crash marker before the first suspension point. iOS
            // may suspend or terminate a backgrounded process while an
            // awaited shutdown is still pending; leaving the foreground
            // marker armed past that await misclassifies an ordinary
            // background termination as a jetsam/watchdog crash next launch.
            CrashReporter.shared.markCleanShutdown()
            await LocalAPIManager.shared.stop()
            CodingAssistantService.shared.cancelLoad()
            CodingAssistantService.shared.stopGeneration()
            await CodingAssistantService.shared.unloadAndWaitForCleanup()
            RuntimeLogCenter.emit(
                "Backgrounded · local API stopped and Core AI model unloaded",
                subsystem: "lifecycle"
            )
        case .inactive:
            break
        @unknown default:
            break
        }
    }
}
