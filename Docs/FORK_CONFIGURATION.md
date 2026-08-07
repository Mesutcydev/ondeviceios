# Fork and signing configuration

A fork must use identifiers owned by its Apple development team. The current
`OnDeviceLAS` target is server-only and contains one app target plus its unit
tests; the legacy assistant, share-extension, App Group, and CloudKit
surfaces remain in the source tree for reference but are not linked by this
build.

## Required replacements

Choose a reverse-DNS prefix such as `com.example.localllm`, then replace:

| Existing value | Where it must remain consistent |
| --- | --- |
| `com.mesutcydev.ondevicelas` | app target bundle ID in `project.yml` |
| `com.mesutcydev.ondevicelas.tests` | unit-test bundle ID in `project.yml` |

Make target-level changes in `project.yml`, then update the tracked entitlement
and Swift files. Regenerate with:

```bash
xcodegen generate
pod install
```

Internal Keychain service names, queue labels, logging subsystems, Spotlight
domains, and background-session identifiers can remain stable for a private
development build. Public forks should replace them to avoid data/keychain
collisions with another installed distribution.

## Capabilities

The current target does not require an App Group or CloudKit container. If your
signing profile does not grant increased memory limit or extended virtual
addressing, remove only the unavailable entitlement and test the resulting
limits. Do not publish a profile, certificate, `.p8`, `.p12`,
`.mobileprovision`, team ID, or private CloudKit credential.

Simulator builds do not require a paid Apple Developer Program membership and
are the safest first verification target.
