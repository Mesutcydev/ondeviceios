#!/usr/bin/env bash

# Build the iOS device and Simulator slices of whisper.cpp without modifying
# the pinned submodule. The helper functions are sourced from whisper.cpp's
# upstream Apple XCFramework script at the pinned commit.

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
whisper_root="${repository_root}/ThirdParty/whisper.cpp"

if [[ ! -f "${whisper_root}/build-xcframework.sh" ]]; then
  echo "error: whisper.cpp submodule is missing; run git submodule update --init --recursive" >&2
  exit 1
fi

cd "$whisper_root"

IOS_MIN_OS_VERSION="${IOS_MIN_OS_VERSION:-16.4}"
BUILD_SHARED_LIBS=OFF
WHISPER_BUILD_EXAMPLES=OFF
WHISPER_BUILD_TESTS=OFF
WHISPER_BUILD_SERVER=OFF
GGML_METAL=ON
GGML_METAL_EMBED_LIBRARY=ON
GGML_BLAS_DEFAULT=ON
GGML_METAL_USE_BF16=ON
GGML_OPENMP=OFF

common_c_flags="-Wno-macro-redefined -Wno-shorten-64-to-32 -Wno-unused-command-line-argument -g"
common_cxx_flags="$common_c_flags"
common_cmake_args=(
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
  -DCMAKE_XCODE_ATTRIBUTE_DEBUG_INFORMATION_FORMAT=dwarf-with-dsym
  -DCMAKE_XCODE_ATTRIBUTE_GCC_GENERATE_DEBUGGING_SYMBOLS=YES
  -DCMAKE_XCODE_ATTRIBUTE_COPY_PHASE_STRIP=NO
  -DCMAKE_XCODE_ATTRIBUTE_STRIP_INSTALLED_PRODUCT=NO
  -DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=ggml
  -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
  -DWHISPER_BUILD_EXAMPLES=${WHISPER_BUILD_EXAMPLES}
  -DWHISPER_BUILD_TESTS=${WHISPER_BUILD_TESTS}
  -DWHISPER_BUILD_SERVER=${WHISPER_BUILD_SERVER}
  -DWHISPER_COREML=ON
  -DWHISPER_COREML_ALLOW_FALLBACK=ON
  -DGGML_METAL_EMBED_LIBRARY=${GGML_METAL_EMBED_LIBRARY}
  -DGGML_BLAS_DEFAULT=${GGML_BLAS_DEFAULT}
  -DGGML_METAL=${GGML_METAL}
  -DGGML_METAL_USE_BF16=${GGML_METAL_USE_BF16}
  -DGGML_OPENMP=${GGML_OPENMP}
)

# Materialize the two upstream packaging helpers before sourcing them. Process
# substitution is unreliable when CodeQL's macOS build tracer is injected.
helper_file="$(mktemp "${TMPDIR:-/tmp}/ios-local-llm-whisper-helpers.XXXXXX")"
trap 'unlink "$helper_file" 2>/dev/null || true' EXIT
awk '
  /^setup_framework_structure\(\)/ { copying = 1 }
  /^# Create dynamic libraries/ && copying { copying = 0 }
  /^combine_static_libraries\(\)/ { copying = 1 }
  /^echo "Building for iOS simulator/ && copying { copying = 0 }
  copying { print }
' build-xcframework.sh >"$helper_file"

# shellcheck disable=SC1090
source "$helper_file"
if ! declare -F setup_framework_structure >/dev/null ||
  ! declare -F combine_static_libraries >/dev/null; then
  echo "error: could not load XCFramework packaging helpers from whisper.cpp" >&2
  exit 1
fi

echo "==> Building whisper.cpp for iOS Simulator"
cmake -B build-ios-sim -G Xcode \
  "${common_cmake_args[@]}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN_OS_VERSION" \
  -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphonesimulator \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator \
  -DCMAKE_C_FLAGS="$common_c_flags" \
  -DCMAKE_CXX_FLAGS="$common_cxx_flags" \
  -S .
cmake --build build-ios-sim --config Release -- -quiet

echo "==> Building whisper.cpp for iOS devices"
cmake -B build-ios-device -G Xcode \
  "${common_cmake_args[@]}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN_OS_VERSION" \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
  -DCMAKE_C_FLAGS="$common_c_flags" \
  -DCMAKE_CXX_FLAGS="$common_cxx_flags" \
  -S .
cmake --build build-ios-device --config Release -- -quiet

setup_framework_structure build-ios-sim "$IOS_MIN_OS_VERSION" ios
setup_framework_structure build-ios-device "$IOS_MIN_OS_VERSION" ios
combine_static_libraries build-ios-sim Release-iphonesimulator ios true
combine_static_libraries build-ios-device Release-iphoneos ios false

mkdir -p build-apple
if [[ -d build-apple/whisper.xcframework ]]; then
  echo "error: build-apple/whisper.xcframework already exists; remove that generated artifact and retry" >&2
  exit 1
fi

xcrun xcodebuild -create-xcframework \
  -framework "$(pwd)/build-ios-sim/framework/whisper.framework" \
  -debug-symbols "$(pwd)/build-ios-sim/dSYMs/whisper.dSYM" \
  -framework "$(pwd)/build-ios-device/framework/whisper.framework" \
  -debug-symbols "$(pwd)/build-ios-device/dSYMs/whisper.dSYM" \
  -output "$(pwd)/build-apple/whisper.xcframework"

echo "==> Created ThirdParty/whisper.cpp/build-apple/whisper.xcframework"
