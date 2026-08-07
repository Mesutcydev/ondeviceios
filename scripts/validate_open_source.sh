#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

required_files=(
  AGENTS.md
  ARCHITECTURE.md
  ASSET_PROVENANCE.md
  CHANGELOG.md
  CITATION.cff
  GOVERNANCE.md
  LICENSE
  MAINTAINERS.md
  PROVENANCE.md
  README.md
  REUSE.toml
  ROADMAP.md
  SBOM.spdx.json
  CONTRIBUTING.md
  SECURITY.md
  THIRD_PARTY_NOTICES.md
  Docs/AGENT_INTEGRATION.md
  Docs/CODING_AGENT_IMPLEMENTATION.md
  Docs/FORK_CONFIGURATION.md
  Docs/OPEN_SOURCE_PROGRAM_READINESS.md
  Docs/RELEASE_VERIFICATION.md
  Docs/REUSABLE_COMPONENTS.md
  Docs/SECURITY_MODEL.md
  Docs/VALIDATION.md
  Docs/XCODE_SECURITY_SETTINGS.md
  IOSLocalLLM/LocalAIRuntimeFoundation/README.md
  codemeta.json
  llms.txt
  .github/CODEOWNERS
  .github/copilot-instructions.md
  .github/workflows/codeql.yml
  .github/workflows/scorecard.yml
  .github/workflows/source-release.yml
  Packages/VoiceAgentOrb/LICENSE
  Packages/VoiceAgentOrb/NOTICE
  Packages/VoiceAgentOrb/README.md
  IOSLocalLLM/Vendor/StableDiffusion/LICENSE
  IOSLocalLLM/Resources/Voice/LICENSE
  LICENSES/Apple-Sample-Code-License.txt
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "$required_file" ]]; then
    echo "error: required file is missing: $required_file" >&2
    exit 1
  fi
done

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: run this check inside the Git repository" >&2
  exit 1
fi

blocked_pattern='(^|/)(Pods|vendor|DerivedData|XcodePublisher_Logs|xcuserdata|BundledVLM|BundledVoiceModels|FastVLM-0\.5B-fp16)(/|$)|\.(p8|p12|mobileprovision|ipa|xcarchive|pkg|gguf|safetensors|onnx|npz)$|(^|/)(ExportOptions\.plist|MEMORY\.md|CLAUDE\.rtf)$'
blocked_files="$(git ls-files | grep -E "$blocked_pattern" || true)"
if [[ -n "$blocked_files" ]]; then
  echo "error: publish-unsafe files are tracked:" >&2
  echo "$blocked_files" >&2
  exit 1
fi

oversized_files="$(
  git ls-files -z |
    xargs -0 -I{} sh -c 'if [ -f "$1" ]; then size=$(wc -c < "$1"); if [ "$size" -gt 50000000 ]; then printf "%s %s\n" "$size" "$1"; fi; fi' _ {} |
    sort -nr
)"
if [[ -n "$oversized_files" ]]; then
  echo "error: tracked files exceed the 50 MB repository limit:" >&2
  echo "$oversized_files" >&2
  exit 1
fi

secret_pattern='AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{50,}|glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'
secret_matches="$(git grep -Il -E "$secret_pattern" -- . ':!scripts/validate_open_source.sh' || true)"
if [[ -n "$secret_matches" ]]; then
  echo "error: possible credentials found in tracked files:" >&2
  echo "$secret_matches" >&2
  exit 1
fi

if ! cmp -s \
  IOSLocalLLM.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved \
  IOSLocalLLM.xcworkspace/xcshareddata/swiftpm/Package.resolved; then
  echo "error: committed Swift package lockfiles differ" >&2
  exit 1
fi

python3 -m json.tool codemeta.json >/dev/null
python3 -m json.tool SBOM.spdx.json >/dev/null

python3 - <<'PY'
import json
import plistlib
import re
from pathlib import Path

project_text = Path("project.yml").read_text(encoding="utf-8")
project_versions = set(
    re.findall(r'CFBundleShortVersionString:\s*"([^"]+)"', project_text)
)
if len(project_versions) != 1:
    raise SystemExit(
        f"error: project.yml has inconsistent marketing versions: {sorted(project_versions)}"
    )

version = project_versions.pop()
for plist_path in (
    Path("IOSLocalLLM/Info.plist"),
    Path("IOSLocalLLMShareExtension/Info.plist"),
):
    with plist_path.open("rb") as plist_file:
        plist_version = plistlib.load(plist_file)["CFBundleShortVersionString"]
    if plist_version != version:
        raise SystemExit(
            f"error: {plist_path} version {plist_version} does not match {version}"
        )

citation_text = Path("CITATION.cff").read_text(encoding="utf-8")
if not re.search(rf"^version:\s*{re.escape(version)}\s*$", citation_text, re.MULTILINE):
    raise SystemExit(f"error: CITATION.cff does not declare version {version}")

with Path("SBOM.spdx.json").open(encoding="utf-8") as sbom_file:
    sbom = json.load(sbom_file)
root_package = next(
    package
    for package in sbom["packages"]
    if package["SPDXID"] == "SPDXRef-Package-ios-local-llm"
)
if root_package["versionInfo"] != version:
    raise SystemExit(
        "error: SBOM root version "
        f"{root_package['versionInfo']} does not match {version}"
    )

changelog_text = Path("CHANGELOG.md").read_text(encoding="utf-8")
if f"## [{version}]" not in changelog_text:
    raise SystemExit(f"error: CHANGELOG.md has no {version} release section")
PY

unpinned_actions="$(
  grep -RHE '^[[:space:]]*uses:' .github/workflows |
    grep -Ev '@[0-9a-f]{40}([[:space:]]|$)' || true
)"
if [[ -n "$unpinned_actions" ]]; then
  echo "error: GitHub Actions must be pinned to full commit SHAs:" >&2
  echo "$unpinned_actions" >&2
  exit 1
fi

retired_brand_pattern='CodeLens|code lens|CODELENS|LOCAL AI STUDIO|Local AI Studio|LOCAL_AI_STUDIO'
retired_brand_matches="$(
  git grep -In -E "$retired_brand_pattern" -- \
    . \
    ':!ThirdParty/**' \
    ':!Pods/**' \
    ':!scripts/validate_open_source.sh' \
    ':!*.pbxproj' || true
)"
if [[ -n "$retired_brand_matches" ]]; then
  echo "error: retired project branding remains in tracked source:" >&2
  echo "$retired_brand_matches" >&2
  exit 1
fi

for executable_script in \
  scripts/build_native_frameworks.sh \
  scripts/native/build_catalyst_slice.sh \
  scripts/native/build_whisper_ios.sh; do
  if [[ ! -x "$executable_script" ]]; then
    echo "error: build script is not executable: $executable_script" >&2
    exit 1
  fi
  bash -n "$executable_script"
done

echo "Repository hygiene checks passed."
