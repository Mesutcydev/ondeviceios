# On Device: LAS

On Device: LAS is a server-only iOS build of the local runtime. It keeps the
local model loading, memory, thermal-safety, authentication, and API
compatibility paths, while removing the assistant, lens, voice, and share
extension surfaces.

For the **iOS 27 / Apple Core AI** product (no MLX or llama.cpp), see
`Docs/CORE_AI_INTEGRATION.md` and the `CoreAIOnDeviceLAS` scheme.

## Build

From this directory:

```sh
xcodegen generate
pod install
open OnDeviceLAS.xcworkspace
```

Use the `OnDeviceLAS` scheme. Before installing on a phone, select your Apple
team and a signing method in Xcode. The bundle identifier is
`com.mesutcydev.ondevicelas`, so it is separate from the source app.

The repository intentionally does not contain model weights. Load or import a
compatible model on the device, then use the server homepage to make it
resident. The model menu keeps the original model-selection path available
without restoring the assistant UI.

## Local API

The server starts automatically on port `11434` and shows its LAN URLs and
bearer key in the app. Rotate the key after installing, then update clients
with the new value. It supports the existing OpenAI-compatible,
Anthropic-compatible, and Ollama-compatible routes implemented by
`LocalAPIServer.swift`.

The homepage exposes compatibility controls for:

- hidden model reasoning (kept out of final protocol text);
- OpenAI/Anthropic/Ollama tool calls;
- parallel tool calls;
- strict JSON-schema validation for tool arguments;
- optional model auto-loading when the app becomes active.

Tool responses are emitted in the provider-native shapes, including preserved
call IDs, structured arguments, multiple calls, and streaming SSE frames.
Ornith 1.0 responses are normalized on the adapter boundary: reasoning before
`</think>` is exposed separately as `reasoning_content` where the provider
supports it, while `content` contains only the final answer. The same parser
handles split streaming markers and runs before tool-call extraction.
Provider-specific fields the local runtime does not use (for example
Anthropic `thinking` or `output_config`) are ignored instead of causing a
400 response. Invalid tool names or arguments are rejected before a client
can execute them.

When MLX handles a request with tools, the runtime uses a token-level response
grammar rather than relying only on prompt instructions. It constrains the
private decision envelope to either a text response or declared tool calls,
keeps arguments syntactically valid JSON, and preserves the normal KV-cache
quantization policy. The API still performs schema validation after decoding.
If a tokenizer cannot expose a usable vocabulary entry for the grammar, the
request falls back to the existing parser-and-validation path instead of
deadlocking generation.

The listener stops when iOS backgrounds the app. Requests made before a model
is resident receive the server's normal service-unavailable response.
