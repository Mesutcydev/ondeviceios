# Core AI: LAS (iOS 27)

`CoreAIOnDeviceLAS` is a separate iPhone product that keeps the authenticated
local API server while routing text generation through Apple Core AI.

It does **not** compile MLX, llama.cpp, voice, camera, or Mac-bridge code.
If Core AI is unavailable, the app fails explicitly — it never silently falls
back to a remote service.

## Requirements

- **Xcode 27** (or newer) with the iOS 27 SDK  
  Example: `/Applications/Xcode-beta.app`
- Physical iPhone on iOS 27+ for Core AI inference  
  (`CoreAI.framework` is device-SDK only; Simulator builds will not resolve it)
- A trusted Core AI model resource directory (`.aimodel` + supporting files)

## Build

```sh
cd /Users/m/Desktop/OnDeviceLAS
xcodegen generate
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
  open OnDeviceLAS.xcworkspace
```

Select the **CoreAIOnDeviceLAS** scheme, your Apple team, and an iOS 27 device.

Command-line device build:

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
xcodebuild \
  -workspace OnDeviceLAS.xcworkspace \
  -scheme CoreAIOnDeviceLAS \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  build
```

Bundle ID: `com.mesutcydev.coreaiondevicelas`  
Display name: **Core AI: LAS**

## Models

1. Import a model pack via Files, or download from a trusted HTTPS URL in
   **Model** management.
2. Packs live under Application Support/`CoreAIModels`.
3. The first successful load specializes the model through Core AI.
4. Backgrounding unloads the model and stops the local API listener.

## Local API

Same compatibility surface as On Device: LAS:

- OpenAI `GET /v1/models`, `POST /v1/chat/completions`
- OpenAI Responses `POST /v1/responses`
- Anthropic Messages `POST /v1/messages`
- Ollama routes under `/api`

Authentication is bearer-key only. Rotate the key after install.
