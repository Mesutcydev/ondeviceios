# AI and agent integration

iOS Local LLM can act as a private local inference endpoint for development
tools and AI agents. These interfaces are opt-in and remain separate from the
app's normal on-device UI.

This document describes the implemented compatibility surface. It does not
claim complete compatibility with every option in the upstream APIs.

## Local API server

Enable the server in the app's Mac/bridge settings. The default port is
`11434`, and the UI displays the reachable LAN address and generated API key.

Every request requires either:

```http
Authorization: Bearer YOUR_IOS_LOCAL_LLM_KEY
```

or:

```http
x-api-key: YOUR_IOS_LOCAL_LLM_KEY
```

The key is generated locally, stored in Keychain, and can be rotated from the
app. Traffic is plain HTTP, so use the server only on a trusted network. The
listener stops when iOS backgrounds the app.

### Implemented routes

| Dialect | Method and route | Notes |
| --- | --- | --- |
| OpenAI | `GET /v1/models` | Lists locally available assistant models |
| OpenAI | `POST /v1/chat/completions` | Text/image chat, streaming, and supported tool calls |
| OpenAI | `POST /v1/responses` | Text/image input and supported tool calls |
| Anthropic | `POST /v1/messages` | Text/base64-image messages and supported tool-use blocks |
| Ollama | `GET /api/tags` | Lists available models |
| Ollama | `POST /api/show` | Returns local model information |
| Ollama | `POST /api/chat` | Chat, base64 images, and supported tool definitions |
| Ollama | `POST /api/generate` | Prompt-based generation with optional base64 images |

Unsupported request options are rejected explicitly rather than silently
ignored. The route implementation and decoding rules live in
`IOSLocalLLM/Services/Bridge/LocalAPIServer.swift`.

Image input is intentionally local-only. Send an embedded `data:image/...;base64`
URL for OpenAI-compatible requests, an Anthropic base64 image source, or an
Ollama `images` base64 value. The server does not fetch HTTP(S) image URLs,
because network access must remain explicit and visible in the app. The active
model must be a native MLX VLM or a GGUF model loaded beside its matching
`mmproj-*.gguf` projector.

### OpenAI-compatible example

```bash
export OPENAI_BASE_URL="http://IPHONE_IP:11434/v1"
export OPENAI_API_KEY="YOUR_IOS_LOCAL_LLM_KEY"

curl "$OPENAI_BASE_URL/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

curl "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "LOCAL_MODEL_ID",
    "messages": [{"role": "user", "content": "Explain this Swift error."}],
    "max_completion_tokens": 4096,
    "reasoning_effort": "none",
    "stream": false
  }'
```

The requested model must be installed and must match a model ID or repository
ID known to the app. `max_tokens` and `max_completion_tokens` both set the
visible completion budget; when both are present, `max_completion_tokens`
wins. The accepted value is clamped to the active runtime's effective limit,
which is published as `max_output_tokens` by `GET /v1/models`. Prompt/context
tokens are budgeted separately from completion tokens.

For Qwen3-family models, `reasoning_effort: "none"` is the recommended
OpenAI-compatible no-thinking option. The compatibility decoder also accepts
`enable_thinking: false`, `think: false`, and
`chat_template_kwargs.enable_thinking: false`; these are normalized to the
single model-aware chat-template option used by the runtime. Options are not
forwarded blindly to model families that do not support them.

OpenAI Chat Completions streaming uses `text/event-stream`. Every successful
stream ends with a chunk containing `finish_reason` (`stop`, `length`, or
`tool_calls`) followed by `data: [DONE]`. Set
`stream_options.include_usage` to `true` to receive the final usage-only chunk.
Non-stream responses always include exact `prompt_tokens`,
`completion_tokens`, and `total_tokens` when the backend reports them. A
deadline or disconnected client cancels its generation and does not emit a
false successful stop.

Reasoning models (Qwen3 / DeepSeek-R1-style parsers advertised as
`reasoning_parser` on `/v1/models`) keep the same split in both modes:
`reasoning_content` holds the chain of thought, and `content` holds only the
final answer. Streaming emits those fields on `choices[0].delta` as tokens
arrive; raw markers such as `<think>`, `</think>`, `<reasoning>`, and
`</reasoning>` are stripped once parsing has started. Clients that only read
`delta.content` therefore see the same clean answer they would get from a
non-streaming completion.

Memory and thermal safety gates may refuse a request. Because those limits can
change while the device heats or cools, clients should query `/v1/models`
again after a refusal rather than assuming a fixed output limit.

## Tool calling

The OpenAI chat/Responses and Anthropic message decoders accept supported function/tool
definitions and normalize them into the app's internal tool-call format.
Tool-choice and parallel-call behavior differ by dialect. Embedded image parts
remain attached while tool instructions are prepared, allowing a compatible
vision model to inspect an image and emit native tool calls in the same turn.

Treat a model's emitted tool call as untrusted input:

- validate the tool name and JSON arguments;
- preserve risk levels and approval requirements;
- do not execute shell or filesystem mutations merely because a model asks;
- return structured errors rather than fabricating successful results.

`IOSLocalLLM/Services/ToolRunner.swift` contains local output parsing.
`IOSLocalLLM/Services/Bridge/LocalAPIServer.swift` contains HTTP tool schemas and
response encoding.

## Paired Mac agent channel

The Mac bridge is a separate, pairing-authenticated channel for grounding an
iPhone assistant in the paired Mac's current context.

The wire protocol uses versioned JSON envelopes for:

- `tool_call`
- `tool_result`
- `tools_list`
- `heartbeat`
- protocol-level `error`

Risk levels are:

- `SAFE_READ`
- `LOW_RISK_ACTION`
- `MEDIUM_RISK_ACTION`
- `HIGH_RISK_ACTION`

The current iPhone flow focuses on safe read-only context, such as the
frontmost app, window title, selected text, and workspace path. A user can
request that context per message with `/mac`; the app does not continuously
observe the Mac.

The canonical iOS protocol types are in
`IOSLocalLLM/Services/Bridge/Agent/AgentProtocol.swift`. Changes must remain
compatible with the matching LocalCoderBridge implementation.

## Privacy and security boundaries

- All agent/local API features are disabled until the user enables or pairs
  them.
- The local API uses a distinct bearer key from Mac pairing.
- The paired agent channel pins identities and uses its own credentials.
- Prompt content, tool arguments, selected text, and model output must not be
  written to ordinary logs.
- Network compatibility does not imply cloud inference. Generation remains
  local unless a separately identified feature explicitly says otherwise.
- Do not expose the local HTTP listener to an untrusted Wi-Fi network or the
  public internet.

## Guidance for integrations

An external agent should:

1. Query `/v1/models` before selecting a model.
2. Expect model availability to change as users download, unload, or delete
   models.
3. Handle `401`, `404`, validation failures, memory refusals, cancellation,
   and thermal throttling.
4. Prefer streaming for long responses and tolerate an early terminal event.
5. Avoid retry loops that could worsen memory or thermal pressure.
6. Keep user approval outside the model for any consequential tool action.
