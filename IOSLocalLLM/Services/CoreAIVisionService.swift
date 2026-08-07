import Foundation
import UIKit

#if canImport(CoreAIImageSegmenter)
import CoreAIImageSegmenter
#endif

/// Vision boundary for Core AI model packages. Image segmentation is kept
/// separate from language generation because Core AI ships task-specific
/// model functions rather than one universal VLM contract.
@MainActor
final class CoreAIVisionService {
    static let shared = CoreAIVisionService()
    private init() {}

    func isAvailable() -> Bool {
        #if canImport(CoreAIImageSegmenter)
        if #available(iOS 27.0, *) { return true }
        #endif
        return false
    }

    func analyze(image: UIImage, prompt: String) async throws -> String {
        #if canImport(CoreAIImageSegmenter)
        if #available(iOS 27.0, *) {
            guard let url = CoreAIModelStore.shared.modelResourcesURL,
                  let manifest = CoreAIModelStore.shared.manifest else {
                throw CoreAIInferenceError.modelMissing
            }
            guard manifest.capabilities.imageInput else {
                throw CoreAIInferenceError.unsupportedCapability("image input")
            }
            guard let cgImage = image.cgImage else {
                throw CoreAIInferenceError.invalidImage("The image has no Core Graphics representation.")
            }
            let segmenter = try await ImageSegmenter(resourcesAt: url.path)
            let response = try await segmenter.segment(image: cgImage, prompt: prompt)
            return "Core AI returned \(response.segments.count) matching image segment(s)."
        }
        #endif
        throw CoreAIInferenceError.unavailable("The Core AI vision runtime is unavailable in this build.")
    }
}
