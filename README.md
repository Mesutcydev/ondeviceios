# OnDevice Local AI Studio

**Run open-source LLMs on iPhone offline** — chat, vision, voice, RAG, and an
authenticated local API. Built in SwiftUI with MLX, llama.cpp, whisper.cpp, and
Core ML. No account. No telemetry. No cloud inference by default.

[![Validate](https://github.com/Mesutcydev/ios-local-llm/actions/workflows/validate.yml/badge.svg)](https://github.com/Mesutcydev/ios-local-llm/actions/workflows/validate.yml)
[![CodeQL](https://github.com/Mesutcydev/ios-local-llm/actions/workflows/codeql.yml/badge.svg)](https://github.com/Mesutcydev/ios-local-llm/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Mesutcydev/ios-local-llm/badge)](https://scorecard.dev/viewer/?uri=github.com/Mesutcydev/ios-local-llm)
[![License: MIT](https://img.shields.io/badge/original%20code-MIT-2ea44f.svg)](LICENSE)
[![Swift](https://img.shields.io/badge/Swift-5.10-F05138.svg)](https://www.swift.org/)
[![Platform](https://img.shields.io/badge/platform-iOS%2018%2B%20%7C%20Mac%20Catalyst-blue.svg)](SETUP_INSTRUCTIONS.md)
[![Website](https://img.shields.io/badge/website-mesut.uk-0A66C2.svg)](https://mesut.uk/apps/codelens)

> Part of the **OnDevice** product line. Independent open-source project — not
> affiliated with or endorsed by Apple Inc.

**OnDevice Local AI Studio** (this repo) is the main local-first AI workbench
for iPhone and Apple silicon Macs. It runs language, vision, speech, and
image-generation models on the device, and exposes opt-in OpenAI-, Anthropic-,
and Ollama-compatible local APIs with structured tool calling and a paired Mac
agent channel.

| Product | Status | What it is |
| --- | --- | --- |
| **OnDevice Local AI Studio** | Main · open source | Full iOS/macCatalyst workbench (this repository) |
| [**OnDevice Local API Server**](https://mesutcydev.github.io/ios-local-llm/) | Beta | Dedicated local API surface for agents and LAN clients — [project page](https://mesutcydev.github.io/ios-local-llm/) |
| **OnDevice CoreAI Local API Server** | Beta | CoreAI-backed local API server variant |

Repository slug remains `ios-local-llm` for stable links. Product name is
**OnDevice Local AI Studio**. Site: [mesut.uk/apps/codelens](https://mesut.uk/apps/codelens).

The app was previously distributed through the App Store. This repository is
now the canonical source distribution. Sideload builds may appear under
Releases; there is currently no official App Store binary from this repo.

[![OnDevice Local AI Studio specification chart](Docs/Images/ios-local-llm-spec-chart.png)](Docs/Images/ios-local-llm-spec-chart.svg)

## App screenshots

| Home | Assistant |
| --- | --- |
| <img src="Docs/Images/Screenshots/home.png" width="320" alt="OnDevice Local AI Studio home dashboard"> | <img src="Docs/Images/Screenshots/assistant.png" width="320" alt="On-device assistant chat screen"> |
| Models | Local vision onboarding |
| <img src="Docs/Images/Screenshots/models.png" width="320" alt="Model discovery and management screen"> | <img src="Docs/Images/Screenshots/local-vision.png" width="320" alt="Local vision onboarding screen"> |

These are unedited iPhone Simulator captures from the open-source source build.
They contain no user content, accounts, tokens, or model weights. Simulator UI
validation does not substitute for physical-device thermal, memory, or
performance evidence; see [validation policy](Docs/VALIDATION.md).

## Highlights

- On-device chat and code assistance with MLX models
- Live camera and image analysis
- Local voice activity detection, transcription, and speech synthesis
- Local OpenAI-compatible API server and Mac bridge
- Model discovery and downloads from Hugging Face
- iPhone and Apple-silicon Mac Catalyst targets

## AI and agent integration

External development tools and agents can use the opt-in local server through:

- OpenAI-style models, chat completions, Responses, streaming, and supported
  tool calls
- Anthropic Messages compatibility
- Ollama-compatible model, chat, and generation routes
- A versioned, pairing-authenticated iPhone-to-Mac tool protocol with explicit
  risk levels

The local API is bearer-authenticated HTTP intended for a trusted LAN and
stops when iOS backgrounds the app. Compatibility is intentionally scoped;
unsupported options fail explicitly. See
[AI and agent integration](Docs/AGENT_INTEGRATION.md) for routes, examples,
security boundaries, and integration guidance.

## For coding agents

This repository includes an agent-readable discovery layer:

- [AGENTS.md](AGENTS.md) — architecture, commands, constraints, and editing
  rules
- [llms.txt](llms.txt) — concise documentation and capability index
- [codemeta.json](codemeta.json) — structured software metadata
- [SBOM.spdx.json](SBOM.spdx.json) — SPDX source dependency inventory
- [.github/copilot-instructions.md](.github/copilot-instructions.md) —
  repository instructions for GitHub Copilot
- [Coding-agent implementation handoff](Docs/CODING_AGENT_IMPLEMENTATION.md) —
  a copy-paste brief for implementing selected components in another app

Agents should treat `project.yml` as the Xcode project source of truth and must
not add model weights, credentials, signing files, or generated native
frameworks to Git.

## Use parts in your own iOS local-LLM app

You do not need to adopt the whole application. The repository now includes:

- [Architecture](ARCHITECTURE.md) — system layers, runtime lifecycle,
  subsystem ownership, and safety invariants
- [Reusable component catalog](Docs/REUSABLE_COMPONENTS.md) — exact source
  files, dependencies, tests, portability level, and extraction guidance
- [Coding-agent implementation handoff](Docs/CODING_AGENT_IMPLEMENTATION.md) —
  a ready-to-share prompt, workflow, and definition of done
- [LocalAIRuntimeFoundation notes](IOSLocalLLM/LocalAIRuntimeFoundation/README.md)
  — the closest existing boundary to a future standalone Swift package

Components are labeled **Extractable**, **Adaptable**, or **Integrated** so
developers and coding agents can distinguish small portable utilities from
services that require app-specific adapters or native inference frameworks.

## Project status

OnDevice Local AI Studio is usable but is a large, evolving application. Some
features require recent Apple hardware, optional model downloads, or native
frameworks that must be built locally. Contributions that improve first-run
setup, tests, accessibility, and documentation are especially welcome.

## Requirements

- macOS with Apple silicon
- Xcode 26 or newer
- iOS 18 or newer
- [XcodeGen](https://github.com/yonaskolb/XcodeGen)
- [CocoaPods](https://cocoapods.org/)
- CMake

An Apple Developer Program membership is not required for Simulator builds.
Running on a physical device uses your own signing identity and bundle
identifier.

## Build

Clone the repository and its native dependencies:

```bash
git clone --recurse-submodules https://github.com/Mesutcydev/ios-local-llm.git
cd ios-local-llm
```

Install project tools if needed:

```bash
brew install xcodegen cocoapods cmake
```

Build the native inference frameworks:

```bash
./scripts/build_native_frameworks.sh
```

For the optional Apple-Silicon Mac Catalyst target, add
`--with-catalyst`. The tracked root scripts build only the required slices
from the pinned submodules; no untracked submodule edits are required.

Generate the Xcode project and install CocoaPods:

```bash
xcodegen generate
pod install
open IOSLocalLLM.xcworkspace
```

Select the `IOSLocalLLM` scheme and an iOS Simulator. For a physical device,
change the bundle identifiers and select your own development team in Xcode.
See [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) and
[fork configuration](Docs/FORK_CONFIGURATION.md) for every identifier,
capability, and optional model step.

## Releases

Official releases are source-only. Starting with `v3.2.6`, each release
includes a reproducible source archive, SHA-256 checksum, and GitHub/Sigstore
provenance attestation. See
[release verification](Docs/RELEASE_VERIFICATION.md) for the exact download
and verification commands.

## Models and large files

No AI model weights, compiled Core ML models, generated XCFrameworks, or app
installers are distributed in this repository. They are intentionally ignored
because they are large and often have terms different from the OnDevice Local
AI Studio license.

OnDevice Local AI Studio downloads supported models only after a user chooses them. Always
review a model's license before downloading or redistributing it. In
particular, Apple FastVLM weights use a research-only license and are not part
of this open-source distribution.

## Privacy

Inference and user data are local by default. Network access is used for
explicit actions such as searching for or downloading models, optional web
search, and communication with a paired local bridge. Review
[PRIVACY_POLICY.md](PRIVACY_POLICY.md) and the app's privacy manifest before
shipping a modified build.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Please use
GitHub Issues for reproducible bugs and focused feature proposals. Security
reports should follow [SECURITY.md](SECURITY.md).

## License

Original project code and documentation are available under the
[MIT License](LICENSE). Third-party code, data, models, and dependencies remain
under their respective terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Names and compatibility
references are explained in [TRADEMARKS.md](TRADEMARKS.md).

Project decisions and contribution roles are documented in
[GOVERNANCE.md](GOVERNANCE.md), [ROADMAP.md](ROADMAP.md), and
[MAINTAINERS.md](MAINTAINERS.md).
