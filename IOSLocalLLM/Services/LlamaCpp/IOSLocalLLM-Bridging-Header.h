//
// IOSLocalLLM-Bridging-Header.h
//
// Objective-C / C bridging header for the llama.cpp integration.
// Exposes the llama.cpp and mtmd (multimodal) C APIs to Swift via
// the LlamaCppBridge.swift wrapper.
//
// The llama / ggml / mtmd headers and implementations come from the
// prebuilt llama.xcframework (built via
// ThirdParty/llama.cpp/build-xcframework.sh). Source-tree header paths
// remain configured in project.yml for deterministic dependency scanning.
//
// Adding new symbols: add the include here, then reference the C
// symbol from Swift via its bare name (Swift sees C globals as
// top-level functions).
//

#ifndef IOS_LOCAL_LLM_BRIDGING_HEADER_H
#define IOS_LOCAL_LLM_BRIDGING_HEADER_H

// Core llama.cpp text-generation API.
//   - llama_backend_init / llama_backend_free
//   - llama_model_load_from_file / llama_model_free
//   - llama_init_from_model (context) / llama_free
//   - llama_tokenize / llama_token_to_piece / llama_detokenize
//   - llama_decode / llama_get_logits
//   - llama_sampler_* (chained samplers)
//   - llama_n_vocab / llama_n_ctx / etc.
#import "llama.h"

// ggml — needed by mtmd.h's struct declarations.
#import "ggml.h"

// mtmd — multimodal helpers. Experimental upstream API; the
// "WARNING: This API is experimental and subject to many BREAKING
// CHANGES" notice in mtmd.h is real. We pin to a specific llama.cpp
// commit (recorded in ThirdParty/llama.cpp/.git/HEAD when cloned)
// and explicitly track upstream changes via LENS_PIPELINE.md.
#import "mtmd.h"

// mtmd-helper — high-level convenience wrappers. Lets us avoid
// hand-managing batch / token-counting / non-causal-attention setup.
// Key symbol: mtmd_helper_eval_chunks() runs the entire text+image
// decode in one call.
#import "mtmd-helper.h"

#endif  // IOS_LOCAL_LLM_BRIDGING_HEADER_H
