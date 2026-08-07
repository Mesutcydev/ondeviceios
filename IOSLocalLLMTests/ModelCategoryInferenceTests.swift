import XCTest
@testable import IOSLocalLLM

// MARK: - ModelCategoryInferenceTests
//
// Pins the category inference that decides whether a model is an assistant,
// a vision/VLM, a voice, or an image-generation (diffusion) model — the logic
// behind "recognize image models at download and when downloaded" and the
// per-category headers in the Models tab.

final class ModelCategoryInferenceTests: XCTestCase {
    func test_explicitConversationSwitchPreservesItsLoadTarget() {
        let savedDefault = AssistantModel(
            id: "downloaded:MercuriusDream/Nanbeige4.2-3B-mlx-4bit",
            repoID: "MercuriusDream/Nanbeige4.2-3B-mlx-4bit",
            displayName: "Nanbeige 3B",
            subtitle: "4-bit",
            approxRAMBytes: 3_000_000_000,
            tags: [],
            contextWindowTokens: 4_096
        )
        let conversationSelection = AssistantModel(
            id: "downloaded:Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
            repoID: "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
            displayName: "Qwen3.5 9B",
            subtitle: "4-bit",
            approxRAMBytes: 6_500_000_000,
            tags: [],
            contextWindowTokens: 4_096
        )

        let resolved = CodingAssistantService.loadTarget(
            activeModel: conversationSelection,
            savedDefault: savedDefault,
            reselectFromSettings: false
        )

        XCTAssertEqual(resolved.id, conversationSelection.id)
        XCTAssertNotEqual(resolved.id, savedDefault.id)
    }

    func test_normalLoadStillUsesSavedDefault() {
        let savedDefault = AssistantModel(
            id: "downloaded:MercuriusDream/Nanbeige4.2-3B-mlx-4bit",
            repoID: "MercuriusDream/Nanbeige4.2-3B-mlx-4bit",
            displayName: "Nanbeige 3B",
            subtitle: "4-bit",
            approxRAMBytes: 3_000_000_000,
            tags: [],
            contextWindowTokens: 4_096
        )
        let currentSelection = AssistantModel(
            id: "downloaded:Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
            repoID: "Jackrong/MLX-Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
            displayName: "Qwen3.5 9B",
            subtitle: "4-bit",
            approxRAMBytes: 6_500_000_000,
            tags: [],
            contextWindowTokens: 4_096
        )

        let resolved = CodingAssistantService.loadTarget(
            activeModel: currentSelection,
            savedDefault: savedDefault,
            reselectFromSettings: true
        )

        XCTAssertEqual(resolved.id, savedDefault.id)
    }

    @MainActor
    func test_installedRegistryModelIsReconciledIntoModelsTabSource() throws {
        let repoID = "Jackrong/test-\(UUID().uuidString)"
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer {
            try? FileManager.default.removeItem(at: directory)
            ModelDownloadCenter.shared.unregisterCustom(repoID: repoID)
        }
        try Data(#"{"architectures":["Qwen3ForCausalLM"]}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        try Data(#"{"model_max_length":4096}"#.utf8)
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try Data([0x01])
            .write(to: directory.appendingPathComponent("model.safetensors"))

        let record = try XCTUnwrap(
            InstalledModelRegistry.validateDirectory(directory, repoID: repoID)
        )
        XCTAssertTrue(record.validationState.isActivatable)

        ModelDownloadCenter.shared.reconcileInstalledRegistry(records: [record])

        let installed = ModelDownloadCenter.shared.models.first {
            $0.sourceRepoID.caseInsensitiveCompare(repoID) == .orderedSame
        }
        XCTAssertNotNil(installed)
        XCTAssertTrue(installed?.isReady == true)
        XCTAssertEqual(installed?.category, .assistant)
    }


    private func cat(_ repo: String, pipeline: String? = nil, tags: [String] = []) -> DownloadableModel.Category {
        LocalModelRegistry.category(repoID: repo, pipelineTag: pipeline, tags: tags)
    }

    // MARK: Pipeline tags are authoritative

    func test_textToImagePipeline_isImageGen() {
        XCTAssertEqual(cat("some/repo", pipeline: "text-to-image"), .imageGen)
        XCTAssertEqual(cat("some/repo", pipeline: "image-to-image"), .imageGen)
        XCTAssertEqual(cat("some/repo", pipeline: "unconditional-image-generation"), .imageGen)
    }

    func test_visionPipeline_isVLM() {
        XCTAssertEqual(cat("some/repo", pipeline: "image-text-to-text"), .vlm)
        XCTAssertEqual(cat("some/repo", pipeline: "image-to-text"), .vlm)
        XCTAssertEqual(cat("some/repo", pipeline: "visual-question-answering"), .vlm)
    }

    func test_speechPipeline_isVoice() {
        XCTAssertEqual(cat("some/repo", pipeline: "text-to-speech"), .voice)
        XCTAssertEqual(cat("some/repo", pipeline: "automatic-speech-recognition"), .voice)
    }

    // MARK: Keyword / tag fallback when pipeline is missing

    func test_diffusionKeywords_isImageGen() {
        XCTAssertEqual(cat("stabilityai/stable-diffusion-2-1-base"), .imageGen)
        XCTAssertEqual(cat("stabilityai/sdxl-turbo"), .imageGen)
        XCTAssertEqual(cat("black-forest-labs/FLUX.1-schnell"), .imageGen)
    }

    func test_diffusersTag_isImageGen() {
        XCTAssertEqual(cat("anyorg/anymodel", tags: ["diffusers"]), .imageGen)
    }

    func test_visionKeywords_isVLM() {
        XCTAssertEqual(cat("Qwen/Qwen2.5-VL-3B-Instruct"), .vlm)
        XCTAssertEqual(cat("HuggingFaceTB/SmolVLM-Instruct"), .vlm)
    }

    func test_voiceKeywords_isVoice() {
        XCTAssertEqual(cat("openai/whisper-base"), .voice)
        XCTAssertEqual(cat("KittenML/kitten-tts"), .voice)
    }

    func test_plainLLM_isAssistant() {
        XCTAssertEqual(cat("Qwen/Qwen2.5-Coder-1.5B-Instruct"), .assistant)
        XCTAssertEqual(cat("meta-llama/Llama-3.2-1B-Instruct"), .assistant)
    }

    // MARK: Image-gen must not be misread as a vision (VLM) model

    func test_imageGenDoesNotCollideWithVision() {
        // A diffusion repo must land in imageGen, never vlm — the bug this fixes.
        XCTAssertNotEqual(cat("stabilityai/stable-diffusion-2-1-base"), .vlm)
        XCTAssertNotEqual(cat("stabilityai/stable-diffusion-2-1-base"), .assistant)
    }

    // MARK: Role mapping stays consistent

    func test_roleMapping() {
        XCTAssertEqual(LocalModelRegistry.role(for: .assistant), .assistant)
        XCTAssertEqual(LocalModelRegistry.role(for: .vlm), .vision)
        XCTAssertEqual(LocalModelRegistry.role(for: .voice), .voice)
        XCTAssertEqual(LocalModelRegistry.role(for: .imageGen), .image)
    }

    // MARK: - Capability inference (Tools / Coder / Fast / Multilingual pills)

    func test_inferred_supportsTools_addsToolsPill() {
        let caps = ModelCapability.inferred(repoID: "mlx-community/Qwen3-4B-4bit",
                                            supportsTools: true)
        XCTAssertTrue(caps.contains(.tools))
    }

    func test_inferred_withoutTools_omitsToolsPill() {
        let caps = ModelCapability.inferred(repoID: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                                            supportsTools: false)
        XCTAssertFalse(caps.contains(.tools))
    }

    func test_inferred_coderRepo_addsCoderPill() {
        let caps = ModelCapability.inferred(repoID: "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
        XCTAssertTrue(caps.contains(.coder))
    }

    func test_inferred_codeTag_addsCoderPill() {
        let caps = ModelCapability.inferred(repoID: "some/model", tags: ["code"])
        XCTAssertTrue(caps.contains(.coder))
    }

    func test_inferred_fastTag_addsFastPill() {
        XCTAssertTrue(ModelCapability.inferred(repoID: "some/model", tags: ["fast"]).contains(.fast))
        XCTAssertTrue(ModelCapability.inferred(repoID: "some/model", tags: ["light"]).contains(.fast))
    }

    func test_inferred_multilingualFamilies() {
        XCTAssertTrue(ModelCapability.inferred(repoID: "Qwen/Qwen3-4B").contains(.multilingual))
        XCTAssertTrue(ModelCapability.inferred(repoID: "prism-ml/Bonsai-27B-mlx-1bit").contains(.multilingual))
        XCTAssertTrue(ModelCapability.inferred(repoID: "google/gemma-2-2b-it").contains(.multilingual))
    }

    func test_bonsaiCatalog_containsAllPublishedMLXVariants() {
        let bonsai = AssistantModelCatalog.presets.filter { $0.repoID.lowercased().contains("bonsai") }
        XCTAssertEqual(bonsai.count, 8)
        XCTAssertEqual(Set(bonsai.map(\.id)), [
            "bonsai-27b-1bit", "bonsai-27b-ternary",
            "bonsai-8b-1bit", "bonsai-8b-ternary",
            "bonsai-4b-1bit", "bonsai-4b-ternary",
            "bonsai-1.7b-1bit", "bonsai-1.7b-ternary",
        ])
        XCTAssertTrue(bonsai.allSatisfy { $0.downloadSizeBytes != nil })
        XCTAssertTrue(bonsai.allSatisfy { $0.supportsThinking })
    }

    func test_ornithIsAFirstClassHighMemoryAssistantPreset() throws {
        let ornith = try XCTUnwrap(
            AssistantModelCatalog.model(forID: "ornith-1.0-9b-4bit")
        )
        XCTAssertEqual(ornith.repoID, "mlx-community/Ornith-1.0-9B-4bit")
        XCTAssertEqual(ornith.platformCompatibility, .highMemoryMobileAndMac)
        XCTAssertEqual(ornith.contextWindowTokens, 4_096)
        XCTAssertLessThanOrEqual(
            ornith.approxRAMBytes + MemoryAdvisor.loadHeadroomReserve,
            MemoryAdvisor.maximumHighMemoryIPhoneProcessCeiling
        )
    }

    func test_bonsaiMetadata_resolvesVendorFamilyAndTemplates() {
        XCTAssertEqual(ModelVendor.infer(from: "prism-ml/Bonsai-27B-mlx-1bit"), .prism)
        XCTAssertEqual(ModelFamily.inferID(from: "prism-ml/Ternary-Bonsai-8B-mlx-2bit"), "bonsai")
        XCTAssertEqual(ModelFamily.displayName(forID: "bonsai"), "Bonsai")
        XCTAssertEqual(ChatTemplate.detect(for: "prism-ml/Bonsai-27B-mlx-1bit").format, "qwen35")
        XCTAssertEqual(ChatTemplate.detect(for: "prism-ml/Bonsai-4B-mlx-1bit").format, "chatml")
    }

    func test_chatTemplateUsesModelContextWithoutExposingItAsDisplayText() {
        let message = ChatMessage(
            role: .user,
            content: "Summarize this.",
            modelContent: "[FILE]\nSource text\nSummarize this."
        )

        let prompt = ChatTemplate.chatML.format(messages: [message])

        XCTAssertTrue(prompt.contains("[FILE]\nSource text\nSummarize this."))
        XCTAssertEqual(message.content, "Summarize this.")
    }

    func test_assistantOutputSanitizer_removesKnownQwenBoundaryLeaks() {
        XCTAssertEqual(
            AssistantOutputSanitizer.clean("\nassistant: off.\n\n### Result\n**Done**"),
            "### Result\n**Done**"
        )
        XCTAssertEqual(
            AssistantOutputSanitizer.clean(".\n\nThis is the answer."),
            "This is the answer."
        )
    }

    func test_assistantOutputSanitizer_preservesNormalAnswerText() {
        let answer = "Assistant: configure the setting to off.\n\nThen retry."
        XCTAssertEqual(AssistantOutputSanitizer.clean(answer), answer)
    }

    func test_assistantMarkdown_preservesSingleLineBreaks() {
        let source = AssistantOutputSanitizer.preservingLineBreaksForMarkdown(
            "Whether it's:\nSolving a problem\nCalculations\n\nDone"
        )
        XCTAssertEqual(
            source,
            "Whether it's:  \nSolving a problem  \nCalculations\n\nDone"
        )

        let rendered = try? AttributedString(
            markdown: source,
            options: .init(interpretedSyntax: .full)
        )
        XCTAssertTrue(
            String(rendered?.characters ?? AttributedString().characters)
                .hasPrefix("Whether it's:\nSolving a problem\nCalculations")
        )
    }

    func test_assistantMarkdown_repairsRunTogetherProseBoundaries() {
        let source = AssistantOutputSanitizer.preservingLineBreaksForMarkdown(
            "Please provide:Your name (full name)Any other detail.Ask again"
        )

        XCTAssertEqual(
            source,
            "Please provide:  \nYour name (full name)  \nAny other detail.\n\nAsk again"
        )
    }

    func test_assistantMarkdown_repairsRunTogetherHeadingsFromLocalModels() {
        let source = AssistantOutputSanitizer.preservingLineBreaksForMarkdown(
            "Document Title: OpusJakePurpose: A workflowDraft Removals — instructionsSubmit Them For MeKeep Me Gone"
        )

        XCTAssertEqual(
            source,
            "Document Title: OpusJake  \nPurpose: A workflow  \nDraft Removals — instructions  \nSubmit Them For Me  \nKeep Me Gone"
        )
    }

    func test_assistantMarkdown_repairsImageDescriptionLabels() {
        let source = AssistantOutputSanitizer.preservingLineBreaksForMarkdown(
            "Image DescriptionThis appears to be a listing.What's Visible:Product: Apple Mac Studio M4 MaxModel number: MU963Chip: 14-core CPU / 32-core GPUMemory: 36GB RAMStorage: 512GB SSDPricing:Original price: EGP 320,600Current price: EGP 153,950 (52% off)Stock Status: 1 item in stockWebsite Elements:Instagram iconNavigation menu: Home.Note: No reviews yet.——The image shows an online store."
        )

        XCTAssertFalse(source.contains("DescriptionThis"))
        XCTAssertFalse(source.contains("MaxModel"))
        XCTAssertFalse(source.contains("MU963Chip"))
        XCTAssertFalse(source.contains("GPUMemory"))
        XCTAssertFalse(source.contains("RAMStorage"))
        XCTAssertFalse(source.contains("SSDPricing"))
        XCTAssertFalse(source.contains("stockWebsite"))
        XCTAssertTrue(source.contains("Description  \nThis"))
        XCTAssertTrue(source.contains("RAM  \nStorage"))
        XCTAssertTrue(source.contains("yet.\n\n—The image"))
    }

    func test_imageGroundingSanitizer_removesCrossModelControlTokens() {
        let raw = """
        <|im_start|>assistant
        Assistant: The image shows a magnetic phone holder.<end_of_utterance>
        It includes a product listing.<|im_end|>
        """

        XCTAssertEqual(
            ImageGroundingSanitizer.clean(raw),
            "The image shows a magnetic phone holder.\nIt includes a product listing."
        )
    }

    func test_bonsai27_isTheOnlyDualTextAndVisionBonsaiFamily() throws {
        let binary27 = try XCTUnwrap(AssistantModelCatalog.model(forID: "bonsai-27b-1bit"))
        let ternary27 = try XCTUnwrap(AssistantModelCatalog.model(forID: "bonsai-27b-ternary"))
        let smaller = AssistantModelCatalog.presets.filter {
            $0.id.hasPrefix("bonsai-") && !$0.id.hasPrefix("bonsai-27b")
        }

        XCTAssertTrue(binary27.capabilities.contains(.vision))
        XCTAssertTrue(ternary27.capabilities.contains(.vision))
        XCTAssertTrue(DualRoleModelPolicy.isTextAndVision(repoID: binary27.repoID))
        XCTAssertTrue(DualRoleModelPolicy.isTextAndVision(repoID: ternary27.repoID))
        XCTAssertTrue(smaller.allSatisfy { !$0.capabilities.contains(.vision) })
        XCTAssertTrue(smaller.allSatisfy {
            !DualRoleModelPolicy.isTextAndVision(repoID: $0.repoID)
        })
    }

    @MainActor
    func test_bonsai27_downloadEntry_keepsOnePackageForBothCatalogRoles() throws {
        let model = try XCTUnwrap(ModelDownloadCenter.shared.models.first {
            $0.id == "bonsai-27b-1bit"
        })
        XCTAssertEqual(model.sourceRepoID, "prism-ml/Bonsai-27B-mlx-1bit")
        XCTAssertTrue(model.supportsCategory(.assistant))
        XCTAssertTrue(model.supportsCategory(.vlm))
    }

    func test_bonsaiCatalog_declaresPublishedPlatformCompatibility() throws {
        let binary27 = try XCTUnwrap(AssistantModelCatalog.model(forID: "bonsai-27b-1bit"))
        let ternary27 = try XCTUnwrap(AssistantModelCatalog.model(forID: "bonsai-27b-ternary"))
        let smaller = AssistantModelCatalog.presets.filter {
            $0.id.hasPrefix("bonsai-") && !$0.id.hasPrefix("bonsai-27b")
        }

        XCTAssertEqual(binary27.platformCompatibility, .highMemoryMobileAndMac)
        XCTAssertEqual(ternary27.platformCompatibility, .macOnly)
        XCTAssertTrue(smaller.allSatisfy { $0.platformCompatibility == .mobileAndMac })

        #if os(macOS) || targetEnvironment(macCatalyst)
        XCTAssertTrue(ModelPlatformCompatibility.macOnly.supportsCurrentPlatform)
        #else
        XCTAssertFalse(ModelPlatformCompatibility.macOnly.supportsCurrentPlatform)
        #endif
    }

    func test_inferred_plainRepo_noFalsePositives() {
        // A bare non-coder, non-multilingual repo with no tools should get
        // no inferred capability pills at all.
        let caps = ModelCapability.inferred(repoID: "meta-llama/Llama-3.2-1B-Instruct")
        XCTAssertTrue(caps.isEmpty, "Expected no inferred pills, got \(caps)")
    }

    func test_displayCapabilities_surfacesToolsForToolModel() {
        // The default flagship supports tools → the picker row should show
        // the Tools pill without the preset literal listing it.
        let flagship = AssistantModelCatalog.presets.first { $0.supportsTools }
        let model = try? XCTUnwrap(flagship)
        XCTAssertTrue(model?.displayCapabilities.contains(.tools) ?? false)
    }

    func test_displayCapabilities_orderingStatusBeforeCapabilityBeforeGated() {
        // Synthetic model carrying a status, a capability, and gated — the
        // ordering contract: status → capability markers → gated last.
        let m = AssistantModel(
            id: "t", repoID: "Qwen/Qwen2.5-Coder-7B", displayName: "t", subtitle: "t",
            approxRAMBytes: 0, tags: ["fast"], contextWindowTokens: 0,
            capabilities: [.best, .gated, .vision], supportsTools: true
        )
        let order = m.displayCapabilities
        let idxBest = order.firstIndex(of: .best)
        let idxVision = order.firstIndex(of: .vision)
        let idxGated = order.firstIndex(of: .gated)
        XCTAssertNotNil(idxBest); XCTAssertNotNil(idxVision); XCTAssertNotNil(idxGated)
        XCTAssertLessThan(idxBest!, idxVision!)   // status before capability
        XCTAssertLessThan(idxVision!, idxGated!)  // capability before gated
        // Inferred markers present too.
        XCTAssertTrue(order.contains(.tools))
        XCTAssertTrue(order.contains(.coder))
        XCTAssertTrue(order.contains(.fast))
    }

    func test_allCapabilities_haveLabelAndSymbol() {
        for cap in ModelCapability.allCases {
            XCTAssertFalse(cap.label.isEmpty, "\(cap) missing label")
            XCTAssertFalse(cap.symbol.isEmpty, "\(cap) missing symbol")
        }
    }
}

@MainActor
final class HFSearchServiceTests: XCTestCase {
    func test_normalizesCopiedHubModelURLs() {
        XCTAssertEqual(
            HFSearchService.normalizedQuery(
                "https://huggingface.co/mlx-community/Qwen3-4B/tree/main"
            ),
            "mlx-community/Qwen3-4B"
        )
        XCTAssertEqual(
            HFSearchService.normalizedQuery("huggingface.co/owner/model.git/resolve/main"),
            "owner/model"
        )
        XCTAssertEqual(HFSearchService.normalizedQuery("  Qwen3 coder  "), "Qwen3 coder")
    }

    func test_mlxSearchUsesHubFilterAndEncodesRepositoryQuery() throws {
        let url = try XCTUnwrap(HFSearchService.searchURL(
            query: "mlx-community/Qwen 3",
            filter: .mlx,
            limit: 250
        ))
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let values = Dictionary(grouping: components.queryItems ?? [], by: \.name)

        XCTAssertEqual(values["search"]?.first?.value, "mlx-community/Qwen 3")
        XCTAssertEqual(values["filter"]?.first?.value, "mlx")
        XCTAssertNil(values["pipeline_tag"])
        XCTAssertEqual(values["limit"]?.first?.value, "100")
        XCTAssertEqual(values["full"]?.first?.value, "true")
    }

    func test_taskSearchUsesPipelineTagRatherThanGenericFilter() throws {
        let url = try XCTUnwrap(HFSearchService.searchURL(
            query: "vision",
            filter: .vlm,
            limit: 30
        ))
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let values = Dictionary(grouping: components.queryItems ?? [], by: \.name)

        XCTAssertEqual(values["pipeline_tag"]?.first?.value, "image-text-to-text")
        XCTAssertNil(values["filter"])
    }

    func test_decodesBothHubIdentifierShapesAndMetadata() throws {
        let json = #"[
          {"modelId":"mlx-community/Qwen3-4B","downloads":123,"likes":7,"tags":["mlx","license:apache-2.0"],"pipeline_tag":"text-generation"},
          {"id":"owner/vision-model","downloads":9,"likes":2,"tags":["coreml"],"pipeline_tag":"image-text-to-text"}
        ]"#
        let models = try HFSearchService.decodeResults(Data(json.utf8))

        XCTAssertEqual(models.map(\.id), ["mlx-community/Qwen3-4B", "owner/vision-model"])
        XCTAssertEqual(models[0].downloads, 123)
        XCTAssertEqual(models[0].licenseTag, "apache-2.0")
        XCTAssertTrue(models[0].isMLX)
        XCTAssertEqual(models[1].pipelineTag, "image-text-to-text")
        XCTAssertTrue(models[1].isCoreML)
    }
}
