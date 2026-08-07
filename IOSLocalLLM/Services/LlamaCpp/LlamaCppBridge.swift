import Foundation
import UIKit
import CoreImage
import os   // OSAllocatedUnfairLock

struct LlamaCppTextGenerationResult: Sendable, Equatable {
    enum StopReason: Sendable, Equatable {
        case stop
        case length
    }

    let tokensPerSecond: Double
    let promptTokenCount: Int
    let completionTokenCount: Int
    let stopReason: StopReason
}

// MARK: - LlamaCppBridge
//
// Swift wrapper for llama.cpp + mtmd (multimodal) C APIs. Loads
// a GGUF LLM + mmproj pair and runs full image-prompt → tokens
// inference. The C calls below are bridged via
// IOSLocalLLM-Bridging-Header.h.
//
// Threading: this type isn't Sendable; the convention is that
// `LlamaCppVLMService` constructs and drives one instance from a
// background `Task.detached`. The cancellation flag is written from
// the MainActor and read from the inference thread, so it's guarded
// by an unfair lock.

/// Errors surfaced by the bridge.
enum LlamaCppError: Error, LocalizedError {
    case backendInitFailed
    case modelLoadFailed(String)
    case mmprojLoadFailed(String)
    case contextInitFailed
    case tokenizeFailed(Int32)
    case invalidToken(Int32)
    case decodeFailed(Int32)
    case bitmapInitFailed
    case visionUnsupported
    case promptTooLong(Int)
    case generationInProgress
    case noOutput
    case cancelled

    var errorDescription: String? {
        switch self {
        case .backendInitFailed:           return "llama.cpp backend init failed"
        case .modelLoadFailed(let p):      return "Failed to load LLM at \(p)"
        case .mmprojLoadFailed(let p):     return "Failed to load mmproj at \(p)"
        case .contextInitFailed:           return "Failed to create llama context"
        case .tokenizeFailed(let code):    return "mtmd_tokenize failed with code \(code)"
        case .invalidToken(let token):     return "Tokenizer produced invalid token id \(token)"
        case .decodeFailed(let code):
            if code == -3 {
                return "llama.cpp backend compute failed while decoding (code -3)"
            }
            return "llama_decode failed with code \(code)"
        case .bitmapInitFailed:            return "Image RGB conversion failed (mtmd_bitmap_init)"
        case .visionUnsupported:           return "Loaded model does not support vision (mmproj mismatch?)"
        case .promptTooLong(let count):    return "Prompt is too long for the GGUF context (\(count) tokens)"
        case .generationInProgress:        return "A GGUF generation is already in progress"
        case .noOutput:                    return "Inference produced no output tokens"
        case .cancelled:                   return "Inference cancelled"
        }
    }
}

extension LlamaCppError: Equatable {
    static func == (lhs: LlamaCppError, rhs: LlamaCppError) -> Bool {
        switch (lhs, rhs) {
        case (.backendInitFailed, .backendInitFailed),
             (.contextInitFailed, .contextInitFailed),
             (.bitmapInitFailed, .bitmapInitFailed),
             (.visionUnsupported, .visionUnsupported),
             (.generationInProgress, .generationInProgress),
             (.noOutput, .noOutput),
             (.cancelled, .cancelled):
            return true
        case (.modelLoadFailed(let a), .modelLoadFailed(let b)):     return a == b
        case (.mmprojLoadFailed(let a), .mmprojLoadFailed(let b)):   return a == b
        case (.tokenizeFailed(let a), .tokenizeFailed(let b)):       return a == b
        case (.invalidToken(let a), .invalidToken(let b)):           return a == b
        case (.decodeFailed(let a), .decodeFailed(let b)):           return a == b
        case (.promptTooLong(let a), .promptTooLong(let b)):         return a == b
        default: return false
        }
    }
}

/// One loaded VLM, ready for image + prompt → token streaming.
final class LlamaCppVLM: @unchecked Sendable {

    // MARK: - C handles

    private var model: OpaquePointer?      // llama_model *
    private var ctx: OpaquePointer?        // llama_context *
    private var mtmdCtx: OpaquePointer?    // mtmd_context *
    // Swift bridges `llama_sampler` as a named C struct, so the
    // pointer type Swift expects is `UnsafeMutablePointer<llama_sampler>`,
    // not `OpaquePointer`. Same applies to llama_batch (handled
    // inline in describe).
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private let configuredThreads: Int32
    private let configuredContextSize: UInt32
    private let textGenerationActive = OSAllocatedUnfairLock(initialState: false)

    // MARK: - Cancellation

    private let cancelFlag = OSAllocatedUnfairLock(initialState: false)

    /// Request the in-flight `describe` call to exit at the next
    /// token boundary. Safe from any thread.
    func cancelCurrent() { cancelFlag.withLock { $0 = true } }

    // MARK: - Backend init (one-shot per process)

    private static var backendInitDone = false
    private static let backendInitLock = NSLock()

    private static func ensureBackendInit() {
        backendInitLock.lock(); defer { backendInitLock.unlock() }
        if !backendInitDone {
            llama_backend_init()
            backendInitDone = true
        }
    }

    // MARK: - Init / deinit

    /// Loads LLM + mmproj into RAM/Metal, creates the inference
    /// context, prepares the sampler chain. Heavy — caller should
    /// dispatch on a background Task. Throws and cleans up on any
    /// step's failure.
    init(
        llmPath: String,
        mmprojPath: String? = nil,
        nThreads: Int32 = Int32(ProcessInfo.processInfo.activeProcessorCount),
        contextSize: UInt32 = 4096,
        gpuLayers: Int32 = 999
    ) throws {
        self.configuredThreads = nThreads
        self.configuredContextSize = contextSize
        Self.ensureBackendInit()

        // 1. Load the text LLM. Default params enable mmap on iOS
        //    which lets the OS page weight bytes out under
        //    pressure — friendly to our memory gate.
        var modelParams = llama_model_default_params()
        modelParams.load_mode = LLAMA_LOAD_MODE_MMAP
        modelParams.n_gpu_layers = gpuLayers
        guard let m = llama_model_load_from_file(llmPath, modelParams) else {
            throw LlamaCppError.modelLoadFailed(llmPath)
        }
        self.model = m

        // 2. Create the inference context. n_batch is the max
        //    tokens per llama_decode call; image embeddings can
        //    arrive in larger batches than typical text, so we
        //    bump above the 32-token default.
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextSize
        ctxParams.n_batch = 512
        ctxParams.n_ubatch = 512
        ctxParams.n_threads = nThreads
        ctxParams.n_threads_batch = nThreads
        guard let c = llama_init_from_model(m, ctxParams) else {
            llama_model_free(m)
            self.model = nil
            throw LlamaCppError.contextInitFailed
        }
        self.ctx = c

        // A standalone text GGUF is complete without an mmproj. Keep the
        // llama context and build the normal sampler, then return before the
        // multimodal setup.
        guard let mmprojPath else {
            let samplerParams = llama_sampler_chain_default_params()
            let chain = llama_sampler_chain_init(samplerParams)
            llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7))
            llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9, 1))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 1...UInt32.max)))
            self.sampler = chain
            return
        }

        // 3. Load the multimodal projector. mtmd_init_from_file
        //    binds it to the LLM and returns a context that
        //    handles tokenization + image-to-embedding as one unit.
        var mtmdParams = mtmd_context_params_default()
        mtmdParams.use_gpu = true
        mtmdParams.print_timings = false
        mtmdParams.n_threads = nThreads
        mtmdParams.image_marker = nil           // deprecated; use media_marker
        mtmdParams.media_marker = mtmd_default_marker()
        mtmdParams.warmup = true                // run a dummy encode to flush Metal pipelines
        guard let mc = mtmd_init_from_file(mmprojPath, m, mtmdParams) else {
            llama_free(c)
            llama_model_free(m)
            self.ctx = nil; self.model = nil
            throw LlamaCppError.mmprojLoadFailed(mmprojPath)
        }
        self.mtmdCtx = mc

        // 4. Vision-support sanity check — catches the case where
        //    the user paired a wrong mmproj (e.g. audio) with the
        //    LLM. Better to fail load than to hit a confusing
        //    decode error later.
        guard mtmd_support_vision(mc) else {
            mtmd_free(mc)
            llama_free(c)
            llama_model_free(m)
            self.mtmdCtx = nil; self.ctx = nil; self.model = nil
            throw LlamaCppError.visionUnsupported
        }

        // 5. Sampler chain: temperature 0.7 + top-p 0.9 + dist
        //    seeding (random per-instance). Mirrors mtmd-cli's
        //    defaults — produces grounded, low-repetition captions
        //    without runaway top-k weirdness.
        let samplerParams = llama_sampler_chain_default_params()
        let chain = llama_sampler_chain_init(samplerParams)
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 1...UInt32.max)))
        self.sampler = chain
    }

    deinit {
        // Ensure all Metal work submitted by this context has retired before
        // freeing the sampler/model. Without this barrier a rapid unload →
        // reload can overlap the old backend's final command buffers with the
        // new model's allocations and fail the second load near the limit.
        if let c = ctx { llama_synchronize(c) }
        if let s = sampler { llama_sampler_free(s) }
        if let mc = mtmdCtx { mtmd_free(mc) }
        if let c = ctx { llama_free(c) }
        if let m = model { llama_model_free(m) }
    }

    // MARK: - Text generation

    /// Runs an already chat-formatted prompt through the resident GGUF. This
    /// also works when an mtmd projector is attached: text-only turns use the
    /// normal tokenizer/context, while image turns enter `describe` below.
    func generateText(
        prompt: String,
        maxTokens: Int,
        onToken: @escaping (String) -> Void
    ) throws -> LlamaCppTextGenerationResult {
        guard let samp = sampler, let m = model else {
            throw LlamaCppError.contextInitFailed
        }
        let acquired = textGenerationActive.withLock { active -> Bool in
            guard !active else { return false }
            active = true
            return true
        }
        guard acquired else { throw LlamaCppError.generationInProgress }
        defer { textGenerationActive.withLock { $0 = false } }
        cancelFlag.withLock { $0 = false }

        let vocab = llama_model_get_vocab(m)
        // The imported-model path already supplies explicit role prefixes.
        // Some community Gemma 4 GGUFs embed a tool-oriented template whose
        // generated control-token sequence is not valid for their quantized
        // per-layer embedding table. Tokenize the plain conversation just as
        // llama-simple does; add_special=true below still supplies BOS.
        let formattedPrompt = prompt

        let required: Int32 = formattedPrompt.withCString { ptr in
            llama_tokenize(vocab, ptr, Int32(strlen(ptr)), nil, 0, true, true)
        }
        guard required < 0 else { throw LlamaCppError.tokenizeFailed(required) }
        var tokens = [llama_token](repeating: 0, count: Int(-required))
        let written: Int32 = formattedPrompt.withCString { ptr in
            tokens.withUnsafeMutableBufferPointer { buffer in
                llama_tokenize(
                    vocab, ptr, Int32(strlen(ptr)), buffer.baseAddress,
                    Int32(buffer.count), true, true
                )
            }
        }
        guard written > 0 else { throw LlamaCppError.tokenizeFailed(written) }
        tokens.removeSubrange(Int(written)..<tokens.count)
        let tokenizedPromptCount = tokens.count
        Diagnostics.shared.breadcrumb(
            "GGUF tokenized · characters=\(formattedPrompt.count) · tokens=\(tokenizedPromptCount) · context=\(configuredContextSize)",
            category: "assistant"
        )

        // Retain the newest conversational tokens when an existing thread is
        // longer than this compact iOS context. Message-level trimming in the
        // service keeps normal requests small; this tokenizer-exact clamp is
        // the final guard because character estimates vary substantially by
        // language and community tokenizer. Preserve the first special token
        // (BOS for this Gemma tokenizer), then take the newest prompt suffix.
        let requestedOutputTokens = max(1, min(maxTokens, Int(configuredContextSize) / 4))
        let promptTokenLimit = max(1, Int(configuredContextSize) - requestedOutputTokens - 1)
        if tokens.count > promptTokenLimit {
            if promptTokenLimit > 1, let firstToken = tokens.first {
                tokens = [firstToken] + tokens.suffix(promptTokenLimit - 1)
            } else {
                tokens = Array(tokens.suffix(promptTokenLimit))
            }
            Diagnostics.shared.breadcrumb(
                "GGUF prompt trimmed · before=\(tokenizedPromptCount) · after=\(tokens.count) · outputReserve=\(requestedOutputTokens)",
                category: "assistant"
            )
        }
        let vocabularySize = llama_vocab_n_tokens(vocab)
        if let invalid = tokens.first(where: { $0 < 0 || $0 >= vocabularySize }) {
            throw LlamaCppError.invalidToken(invalid)
        }
        // The user's output preference may equal/exceed this GGUF's bounded
        // runtime context (for example 4096). Clamp output to the space left
        // after prompt tokenization instead of failing every first send.
        let generationLimit = min(requestedOutputTokens, Int(configuredContextSize) - tokens.count - 1)
        guard generationLimit > 0 else { throw LlamaCppError.promptTooLong(tokens.count) }

        // Gemma 4's recurrent per-layer input path is sensitive to physical
        // batches above 64 tokens even though the logical context is larger.
        // Recreate the cheap context per request (the loaded model remains
        // resident), cap n_batch/n_ubatch at 64, and submit prompt prefill in
        // matching chunks. A single 256–383-token llama_decode can otherwise
        // abort inside ggml_backend_blas_graph_compute before Swift can catch it.
        let physicalBatchSize = 64
        if let oldContext = ctx {
            llama_synchronize(oldContext)
            llama_free(oldContext)
            ctx = nil
        }
        var requestParams = llama_context_default_params()
        requestParams.n_ctx = configuredContextSize
        requestParams.n_batch = UInt32(physicalBatchSize)
        requestParams.n_ubatch = UInt32(physicalBatchSize)
        requestParams.n_threads = configuredThreads
        requestParams.n_threads_batch = configuredThreads
        guard let c = llama_init_from_model(m, requestParams) else {
            throw LlamaCppError.contextInitFailed
        }
        ctx = c

        // Use llama.cpp's single-sequence helper instead of assigning explicit
        // positions. Qwen 3.5 mixes recurrent/GDN and attention layers; current
        // llama.cpp tracks the recurrent sequence state internally, and a
        // manually restarted position batch can make the second decode graph
        // fail with code -3. The helper borrows the Swift token buffer only for
        // the duration of the synchronous decode below.
        let chunkCount = (tokens.count + physicalBatchSize - 1) / physicalBatchSize
        Diagnostics.shared.breadcrumb(
            "GGUF prefill start · tokens=\(tokens.count) · batch=\(physicalBatchSize) · chunks=\(chunkCount)",
            category: "assistant"
        )
        var promptOffset = 0
        var chunkIndex = 0
        while promptOffset < tokens.count {
            if cancelFlag.withLock({ $0 }) { throw LlamaCppError.cancelled }
            let count = min(physicalBatchSize, tokens.count - promptOffset)
            chunkIndex += 1
            Diagnostics.shared.breadcrumb(
                "GGUF prefill chunk · index=\(chunkIndex)/\(chunkCount) · tokens=\(count) · offset=\(promptOffset)",
                category: "assistant"
            )
            let promptStatus = tokens.withUnsafeMutableBufferPointer { buffer in
                let batch = llama_batch_get_one(
                    buffer.baseAddress?.advanced(by: promptOffset),
                    Int32(count)
                )
                return llama_decode(c, batch)
            }
            guard promptStatus == 0 else { throw LlamaCppError.decodeFailed(promptStatus) }
            promptOffset += count
        }
        Diagnostics.shared.breadcrumb(
            "GGUF prefill complete · tokens=\(tokens.count) · chunks=\(chunkCount)",
            category: "assistant"
        )

        let startTime = Date()
        var tokensGenerated = 0
        var reachedEndOfGeneration = false
        var pendingUTF8: [UInt8] = []
        pendingUTF8.reserveCapacity(16)
        var pieceBuffer = [Int8](repeating: 0, count: 256)
        llama_sampler_reset(samp)

        for _ in 0..<generationLimit {
            if cancelFlag.withLock({ $0 }) { throw LlamaCppError.cancelled }
            let tokenID = llama_sampler_sample(samp, c, -1)
            llama_sampler_accept(samp, tokenID)
            if llama_vocab_is_eog(vocab, tokenID) {
                reachedEndOfGeneration = true
                break
            }

            let pieceCount = pieceBuffer.withUnsafeMutableBufferPointer { buffer in
                llama_token_to_piece(vocab, tokenID, buffer.baseAddress, Int32(buffer.count), 0, true)
            }
            if pieceCount > 0 {
                pendingUTF8.reserveCapacity(pendingUTF8.count + Int(pieceCount))
                for index in 0..<Int(pieceCount) {
                    pendingUTF8.append(UInt8(bitPattern: pieceBuffer[index]))
                }
                let piece = Self.consumeCompleteUTF8Prefix(&pendingUTF8)
                if !piece.isEmpty { onToken(piece) }
            }
            tokensGenerated += 1

            var nextToken = tokenID
            let status = withUnsafeMutablePointer(to: &nextToken) { tokenPointer in
                llama_decode(c, llama_batch_get_one(tokenPointer, 1))
            }
            guard status == 0 else { throw LlamaCppError.decodeFailed(status) }
        }

        if !pendingUTF8.isEmpty {
            onToken(String(decoding: pendingUTF8, as: UTF8.self))
        }
        return LlamaCppTextGenerationResult(
            tokensPerSecond: Self.tokensPerSec(
                tokens: tokensGenerated,
                start: startTime
            ),
            promptTokenCount: tokens.count,
            completionTokenCount: tokensGenerated,
            stopReason: reachedEndOfGeneration ? .stop : .length
        )
    }

    // MARK: - Describe

    /// Encode the image, prepend the prompt, stream tokens via
    /// `onToken`, finish via `onComplete(tokensPerSec)`. Cancellable
    /// via `cancelCurrent()`. On error, throws and `onComplete`
    /// is NOT called.
    func describe(
        image: UIImage,
        prompt: String,
        maxTokens: Int = 256,
        onToken: @escaping (String) -> Void,
        onComplete: @escaping (Double) -> Void
    ) throws {
        guard let mc = mtmdCtx, let c = ctx, let samp = sampler, let m = model else {
            throw LlamaCppError.contextInitFailed
        }
        cancelFlag.withLock { $0 = false }

        // Reset the KV cache from the previous describe() call. mtmd_helper_
        // eval_chunks below feeds tokens starting at n_past = 0, but llama_
        // decode tracks the highest position it saw for each sequence and
        // refuses to accept a batch whose starting position isn't
        // (last_pos + 1). Without this clear, the second call fails with:
        //   "init: the tokens of sequence 0 in the input batch have
        //    inconsistent sequence positions … X = N, Y = 0"
        // and the caller is forced to tear down and reload the whole 780+ MB
        // model just to get a working context — observed in the crash log
        // before this fix. data: false drops only the position-tracking
        // metadata; the underlying KV tensor storage is reused.
        llama_memory_clear(llama_get_memory(c), false)

        // 1. Convert UIImage → RGB-packed bytes for mtmd_bitmap.
        let bitmap = try Self.makeRGBBitmap(from: image)
        let mtmdBitmap: OpaquePointer? = bitmap.rgb.withUnsafeBufferPointer { buf -> OpaquePointer? in
            guard let base = buf.baseAddress else { return nil }
            return mtmd_bitmap_init(UInt32(bitmap.width), UInt32(bitmap.height), base)
        }
        guard let mb = mtmdBitmap else { throw LlamaCppError.bitmapInitFailed }
        defer { mtmd_bitmap_free(mb) }

        // 2. Build a chat-template-shaped prompt with the media marker
        //    mtmd will replace with image-token placeholders during
        //    tokenize. SmolVLM2's GGUF weights carry a chat template
        //    (`<|im_start|>User: ...<end_of_utterance>\nAssistant:`).
        //    Feeding only "<__media__>\n<prompt>" used to limp along, but
        //    newer mtmd/template handling expects the real role wrapper.
        let marker = String(cString: mtmd_get_marker(mc))
        let fullPrompt = Self.formatPrompt(
            prompt,
            marker: marker,
            model: m
        )

        // 3. Tokenize text + bitmap into a chunks list.
        let chunks = mtmd_input_chunks_init()
        defer { mtmd_input_chunks_free(chunks) }

        let tokenizeStatus: Int32 = fullPrompt.withCString { cstr -> Int32 in
            var inputText = mtmd_input_text()
            inputText.text = cstr
            inputText.text_len = strlen(cstr)
            inputText.add_special = true
            inputText.parse_special = true
            var bitmaps: [OpaquePointer?] = [mb]
            return bitmaps.withUnsafeMutableBufferPointer { bp -> Int32 in
                mtmd_tokenize(mc, chunks, &inputText, bp.baseAddress, 1)
            }
        }
        guard tokenizeStatus == 0 else {
            throw LlamaCppError.tokenizeFailed(tokenizeStatus)
        }

        // 4. Eval the chunks. mtmd-helper handles the text+image
        //    interleave + non-causal attention setup for image
        //    chunks. n_past starts at 0 (fresh KV cache).
        var nPast: llama_pos = 0
        let evalStatus = mtmd_helper_eval_chunks(
            mc,
            c,
            chunks,
            /* n_past */ 0,
            /* seq_id */ 0,
            /* n_batch */ 512,
            /* logits_last */ true,
            &nPast
        )
        guard evalStatus == 0 else {
            throw LlamaCppError.decodeFailed(evalStatus)
        }

        // 5. Generate. One sample → decode → advance n_past per
        //    iteration. EOS or maxTokens or cancel ends the loop.
        let startTime = Date()
        var tokensGenerated = 0
        let vocab = llama_model_get_vocab(m)
        var batch = llama_batch_init(1, 0, 1)
        defer { llama_batch_free(batch) }

        // Raw UTF-8 bytes carried across tokens. A multi-byte character
        // (emoji, CJK) is routinely split across two pieces; decoding each
        // piece in isolation corrupts it. We accumulate bytes here, emit
        // the longest complete-UTF-8 prefix per token, and retain the
        // partial tail for the next piece.
        var pendingUTF8: [UInt8] = []
        pendingUTF8.reserveCapacity(16)
        var pieceBuf = [Int8](repeating: 0, count: 128)

        for _ in 0..<maxTokens {
            if cancelFlag.withLock({ $0 }) {
                // Contract (see doc-comment above): on throw, onComplete
                // is NOT called — the caller's catch handler owns the
                // finalisation in that case. Calling onComplete here
                // AND throwing caused a double-fire that crashed the
                // AnalysisService continuation ("SWIFT TASK CONTINUATION
                // MISUSE: tried to resume its continuation more than once").
                throw LlamaCppError.cancelled
            }

            let tokenID = llama_sampler_sample(samp, c, -1)
            llama_sampler_accept(samp, tokenID)

            if llama_vocab_is_eog(vocab, tokenID) {
                break
            }

            // Token → UTF-8 piece. 128-byte buffer covers any
            // single-token piece (most are <16).
            let n = pieceBuf.withUnsafeMutableBufferPointer { buf -> Int32 in
                llama_token_to_piece(vocab, tokenID, buf.baseAddress, Int32(buf.count), 0, true)
            }
            if n > 0 {
                // Take exactly `n` bytes. String(cString:) is wrong here on
                // both counts: it needs a NUL terminator (absent when the
                // piece fills the buffer — OOB read) and it mangles a
                // multi-byte sequence split across tokens.
                pendingUTF8.reserveCapacity(pendingUTF8.count + Int(n))
                for index in 0..<Int(n) {
                    pendingUTF8.append(UInt8(bitPattern: pieceBuf[index]))
                }
                let piece = Self.consumeCompleteUTF8Prefix(&pendingUTF8)
                if !piece.isEmpty { onToken(piece) }
            }
            tokensGenerated += 1

            // Next batch: single token at position nPast, logits
            // requested on this position so the next iteration
            // can sample from them.
            batch.n_tokens = 1
            batch.token[0] = tokenID
            batch.pos[0] = nPast
            batch.n_seq_id[0] = 1
            batch.seq_id[0]?.pointee = 0
            batch.logits[0] = 1

            let decRet = llama_decode(c, batch)
            if decRet != 0 {
                throw LlamaCppError.decodeFailed(decRet)
            }
            nPast += 1
        }

        // Flush whatever bytes are still held back (a trailing split
        // sequence that never completed) — decode lossily so nothing is
        // silently dropped.
        if !pendingUTF8.isEmpty {
            let tail = String(decoding: pendingUTF8, as: UTF8.self)
            pendingUTF8.removeAll()
            if !tail.isEmpty { onToken(tail) }
        }

        onComplete(Self.tokensPerSec(tokens: tokensGenerated, start: startTime))
    }

    // MARK: - Helpers

    /// Splits `bytes` into the longest prefix that ends on a complete
    /// UTF-8 character and the trailing bytes of an incomplete multi-byte
    /// sequence (0–3 bytes). llama.cpp tokens routinely split a character
    /// across pieces, so the tail is carried over to the next piece.
    private static func consumeCompleteUTF8Prefix(_ bytes: inout [UInt8]) -> String {
        guard !bytes.isEmpty else { return "" }
        // Walk back over at most 3 trailing continuation bytes to find the
        // lead byte of the final character, then check whether the sequence
        // it starts is complete.
        var cut = bytes.count
        var i = bytes.count - 1
        var continuations = 0
        while i >= 0, continuations < 3, bytes[i] & 0b1100_0000 == 0b1000_0000 {
            continuations += 1
            i -= 1
        }
        if i >= 0 {
            let lead = bytes[i]
            let expected: Int
            if lead & 0b1000_0000 == 0 { expected = 1 }
            else if lead & 0b1110_0000 == 0b1100_0000 { expected = 2 }
            else if lead & 0b1111_0000 == 0b1110_0000 { expected = 3 }
            else if lead & 0b1111_1000 == 0b1111_0000 { expected = 4 }
            else { expected = 1 }   // invalid lead — let lossy decoding handle it
            if continuations + 1 < expected {
                cut = i   // final character incomplete — hold it back
            }
        }
        let prefix = String(decoding: bytes[..<cut], as: UTF8.self)
        if cut == bytes.count {
            bytes.removeAll(keepingCapacity: true)
        } else if cut > 0 {
            bytes.removeFirst(cut)
        }
        return prefix
    }

    private static func tokensPerSec(tokens: Int, start: Date) -> Double {
        let elapsed = Date().timeIntervalSince(start)
        return elapsed > 0 ? Double(tokens) / elapsed : 0
    }

    /// Apply the GGUF's own chat template around the user turn, keeping the
    /// mtmd media marker inside the user content. This mirrors
    /// llama-mtmd-cli's single-turn path and is especially important for
    /// SmolVLM/SmolVLM2, whose `<|im_start|>` token is BOS rather than ChatML.
    private static func formatPrompt(
        _ prompt: String,
        marker: String,
        model: OpaquePointer
    ) -> String {
        let userContent: String = {
            if prompt.contains(marker) { return prompt }
            if prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return marker
            }
            return "\(marker)\n\(prompt)"
        }()

        guard let template = llama_model_chat_template(model, nil),
              let rendered = applyChatTemplate(template, userContent: userContent) else {
            return userContent
        }
        return rendered
    }

    private static func applyChatTemplate(
        _ template: UnsafePointer<CChar>,
        userContent: String
    ) -> String? {
        "user".withCString { rolePtr in
            userContent.withCString { contentPtr in
                var message = llama_chat_message(role: rolePtr, content: contentPtr)
                let required = llama_chat_apply_template(
                    template,
                    &message,
                    1,
                    true,
                    nil,
                    0
                )
                guard required >= 0 else { return nil }

                var buffer = [CChar](repeating: 0, count: Int(required) + 1)
                let written = buffer.withUnsafeMutableBufferPointer { buf in
                    llama_chat_apply_template(
                        template,
                        &message,
                        1,
                        true,
                        buf.baseAddress,
                        Int32(buf.count)
                    )
                }
                guard written >= 0 else { return nil }
                let byteCount = min(Int(written), buffer.count - 1)
                let bytes = buffer.prefix(byteCount).map { UInt8(bitPattern: $0) }
                return String(decoding: bytes, as: UTF8.self)
            }
        }
    }

    /// UIImage → row-major RGB bytes (3 bytes per pixel) at the
    /// image's native pixel dimensions. mtmd resizes internally
    /// per the mmproj's metadata, so we don't pre-resize. We
    /// rasterize through a CGContext to flatten orientation and
    /// drop any alpha channel — mtmd_bitmap_init expects exactly
    /// `width * height * 3` bytes with no padding.
    ///
    /// Path: render into a 32bpp BGRA bitmap, then repack down to
    /// 24bpp RGB. We CANNOT render directly into 24bpp RGB —
    /// CGBitmapContext on iOS rejects packed 8-bit-per-component
    /// formats without an alpha channel and logs
    ///   "CGBitmapContextCreate: unsupported parameter combination:
    ///    RGB | 8 bits/component, integer | <N> bytes/row.
    ///    kCGImageAlphaNone | kCGImageByteOrderDefault | kCGImagePixelFormatPacked"
    /// before returning nil. The repack step is one O(w*h) pass over
    /// already-resident bytes, cheaper than even a single Metal upload.
    private struct RGBBitmap {
        let rgb: [UInt8]
        let width: Int
        let height: Int
    }

    private static func makeRGBBitmap(from image: UIImage) throws -> RGBBitmap {
        let cg: CGImage
        if let cgImage = image.cgImage {
            cg = cgImage
        } else if let ciImage = image.ciImage {
            let renderCtx = CIContext()
            guard let made = renderCtx.createCGImage(ciImage, from: ciImage.extent) else {
                throw LlamaCppError.bitmapInitFailed
            }
            cg = made
        } else {
            throw LlamaCppError.bitmapInitFailed
        }

        let width = cg.width
        let height = cg.height
        let bgraBytesPerRow = width * 4
        var bgra = [UInt8](repeating: 0, count: bgraBytesPerRow * height)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ok: Bool = bgra.withUnsafeMutableBufferPointer { buf -> Bool in
            guard let ctx = CGContext(
                data: buf.baseAddress,
                width: width, height: height,
                bitsPerComponent: 8,
                bytesPerRow: bgraBytesPerRow,
                space: cs,
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue |
                            CGImageAlphaInfo.premultipliedFirst.rawValue
            ) else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        if !ok { throw LlamaCppError.bitmapInitFailed }

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        rgb.withUnsafeMutableBufferPointer { rgbBuf in
            bgra.withUnsafeBufferPointer { bgraBuf in
                guard let rgbBase = rgbBuf.baseAddress,
                      let bgraBase = bgraBuf.baseAddress else { return }
                for i in 0..<(width * height) {
                    let s = bgraBase + i * 4
                    let d = rgbBase + i * 3
                    d[0] = s[2]   // R (from B-position in little-endian BGRA)
                    d[1] = s[1]   // G
                    d[2] = s[0]   // B
                }
            }
        }
        return RGBBitmap(rgb: rgb, width: width, height: height)
    }
}
