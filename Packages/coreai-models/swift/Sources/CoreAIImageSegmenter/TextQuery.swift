// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Three levels of text input, from raw text down to pre-computed embeddings.
///
/// Choose the earliest stage your caller can supply:
///
/// - `.prompt` — Use this at the `ImageSegmenter` level. `ImageSegmenter.segment(image:textQuery:)`
///   tokenizes the string via `CLIPTokenizer` and forwards `.tokens` to the engine.
///   **Do not pass `.prompt` directly to a `SegmentationEngine`** — engines will throw
///   `SegmentationRuntimeError.invalidConfiguration` if they receive it.
///
/// - `.tokens` — Pass pre-tokenized `Int32` IDs when you want to reuse a tokenization
///   result across calls or run your own tokenizer. Shape: `(batch_size, contextLength)`.
///
/// - `.embeddings` — Pass pre-computed Float embeddings to skip both tokenization and
///   the text encoder inside the model. Only supported by models with a dedicated embeddings
///   input tensor (discovered at init by `CoreAISegmentationEngine`). Shape:
///   `(batch_size, sequence_length, hidden_size)`.
///
/// **Adding a new input modality** (e.g. image crops, audio): add a new case here and a
/// corresponding branch in `CoreAISegmentationEngine.segment(image:textQuery:parameters:)`.
/// `SegmentationParameters` can hold any extra hyper-parameters the new case requires.
public enum TextQuery: Sendable {
    /// Raw text prompt — tokenized by `ImageSegmenter` before reaching the engine.
    case prompt(String)
    /// Pre-tokenized token IDs, shape `[batch_size, contextLength]`.
    case tokens([[Int32]])
    /// Pre-computed text embeddings, shape `[batch_size, sequence_length, hidden_size]`.
    case embeddings([[[Float]]])
}
