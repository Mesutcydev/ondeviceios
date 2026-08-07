# Validation and evidence policy

This document separates implemented safeguards from verified behavior. A code
path or a user report is not automatically a benchmark.

## Current evidence

The maintainer has confirmed manual debugging and user testing of model
management, RAM admission, memory-warning handling, and thermal-management
behavior. The repository contains the corresponding policy and lifecycle code,
but does not yet include enough shareable raw logs to claim a comprehensive
device benchmark.

Automated, repeatable checks currently include:

| Scope | Command | Environment |
| --- | --- | --- |
| Repository hygiene | `./scripts/validate_open_source.sh` | Linux or macOS |
| VoiceAgentOrb package | `swift test --package-path Packages/VoiceAgentOrb` | macOS CI |
| XcodeGen consistency | `xcodegen generate` followed by a clean diff | macOS |
| App workspace build | `xcodebuild build -workspace OnDeviceLAS.xcworkspace -scheme OnDeviceLAS -destination 'platform=iOS Simulator,name=iPhone 17'` | iPhone 17 simulator, iOS 27 |
| App unit tests | `xcodebuild test -workspace OnDeviceLAS.xcworkspace -scheme OnDeviceLAS -destination 'platform=iOS Simulator,name=iPhone 17'` | iPhone 17 simulator, iOS 27 |

On 29 July 2026, the open-source workspace built successfully and the
first-launch legal flow, onboarding, Home, Assistant, and Models screens were
manually exercised on the simulator. Assistant model preparation is
intentionally skipped on simulator builds because the simulated GPU cannot
provide valid MLX inference evidence; the real inference path remains a
physical-device validation requirement.

Do not pass `CODE_SIGNING_ALLOWED=NO` to the test command. Simulator tests are
ad-hoc signed without a paid developer account, and the Keychain-backed
pairing and credential-wipe tests require the resulting Keychain entitlement.

## Required physical-device matrix

Before making a quantified reliability or performance claim, record:

| Field | Required evidence |
| --- | --- |
| Hardware | exact device model and RAM tier |
| Software | iOS, Xcode, app commit, model ID and quantization |
| Model lifecycle | load, generate, cancel, switch, unload |
| Memory pressure | admission decision, warning handling, recovery |
| Thermal | nominal/fair/serious/critical transition behavior |
| Backgrounding | generation stop and resource release |
| Duration | warm-up, run length, prompt/context/output sizes |
| Result | logs, MetricKit/signpost evidence, failures and limitations |

Simulator tests are useful for UI and deterministic logic, but are not evidence
for Jetsam limits, Metal residency, sustained thermals, or battery behavior.
Use synthetic prompts and redact tokens, device names, addresses, and user data.

## Claim language

Use “implemented” for a code path, “manually observed” for a recorded manual
test, and “validated” only when the device, commit, procedure, and outcome are
documented. Never generalize one device result to all supported hardware.
