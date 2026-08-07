# coreai-models (OnDeviceLAS pin)

Vendored from https://github.com/apple/coreai-models.git
at revision `938d0b8943b942ce66438b94ab017c5631d1aef4`.

## Local patch

`swift/Sources/CoreAILanguageModels/LanguageModel/CoreAILanguageModel.swift`
no longer calls
`LanguageModelExecutorGenerationChannel.Response.Action.updateUsage`.

**Why:** On iPhone OS 27.0 (24A5390f) the system `FoundationModels`
framework does not export that symbol, while Xcode 27's SDK still
declares it. Linking the upstream call aborts at process load
(`dyld` / "Symbol not found") before the Core AI: LAS UI appears.

Remove this pin and return to the remote package once the device OS
and Xcode SDK export matching FoundationModels executor APIs.
