#!/bin/bash
#
# Build sortformer.xcframework for Apple platforms
# Produces a DYNAMIC framework with CoreML/ANE support and dSYMs.
# GGML symbols are NOT compiled in — they come from whisper.xcframework at runtime
# (resolved via -undefined dynamic_lookup).
#
set -e

# ============================================================================
# Options
# ============================================================================
IOS_MIN_OS_VERSION=16.4
MACOS_MIN_OS_VERSION=13.3

FRAMEWORK_NAME="sortformer"
BUNDLE_ID="org.ggml.sortformer"

# Source paths (relative to repo root)
SORTFORMER_SRC="streaming-sortformer/src"
SORTFORMER_CPP="${SORTFORMER_SRC}/sortformer.cpp"
SORTFORMER_H="${SORTFORMER_SRC}/sortformer.h"
COREML_MM="${SORTFORMER_SRC}/coreml/sortformer-coreml.mm"
COREML_H="${SORTFORMER_SRC}/coreml/sortformer-coreml.h"

# Include paths
GGML_INCLUDE="ggml/include"
SORTFORMER_INCLUDE="${SORTFORMER_SRC}"
COREML_INCLUDE="${SORTFORMER_SRC}/coreml"

# Common compiler flags (-g for debug symbols / dSYM generation)
COMMON_FLAGS="-O2 -g -DSORTFORMER_USE_COREML -DGGML_MAX_NAME=128"
COMMON_WARNS="-Wno-shorten-64-to-32 -Wno-unused-command-line-argument"
CXX_STD="-std=c++17"

BASE_DIR="$(pwd)"

echo "=== Building sortformer.xcframework ==="
echo "Base directory: ${BASE_DIR}"

# ============================================================================
# Verify source files exist
# ============================================================================
for f in "${SORTFORMER_CPP}" "${SORTFORMER_H}" "${COREML_MM}" "${COREML_H}"; do
    if [ ! -f "$f" ]; then
        echo "Error: Required source file not found: $f"
        exit 1
    fi
done

if [ ! -d "${GGML_INCLUDE}" ]; then
    echo "Error: GGML include directory not found: ${GGML_INCLUDE}"
    exit 1
fi

# ============================================================================
# Clean previous sortformer build artifacts
# ============================================================================
echo "Cleaning previous sortformer build artifacts..."
rm -rf build-sortformer-macos
rm -rf build-sortformer-ios-device
rm -rf build-sortformer-ios-sim

# ============================================================================
# Helper: compile sortformer for a given platform
# ============================================================================
compile_sortformer() {
    local build_dir="$1"
    local sdk="$2"
    local archs="$3"
    local min_version_flag="$4"

    echo "--- Compiling sortformer for ${build_dir} (sdk=${sdk}, archs=${archs}) ---"
    mkdir -p "${build_dir}"

    local sysroot
    sysroot="$(xcrun --sdk "${sdk}" --show-sdk-path)"

    # Build arch flags
    local arch_flags=""
    for arch in ${archs}; do
        arch_flags="${arch_flags} -arch ${arch}"
    done

    local include_flags="-I${BASE_DIR}/${GGML_INCLUDE} -I${BASE_DIR}/${SORTFORMER_INCLUDE} -I${BASE_DIR}/${COREML_INCLUDE}"

    # 1) Compile sortformer.cpp -> sortformer.o
    echo "  Compiling sortformer.cpp..."
    xcrun -sdk "${sdk}" clang++ \
        -isysroot "${sysroot}" \
        ${arch_flags} \
        ${min_version_flag} \
        ${CXX_STD} \
        ${COMMON_FLAGS} \
        ${COMMON_WARNS} \
        ${include_flags} \
        -c "${BASE_DIR}/${SORTFORMER_CPP}" \
        -o "${build_dir}/sortformer.o"

    # 2) Compile sortformer-coreml.mm -> sortformer-coreml.o
    echo "  Compiling sortformer-coreml.mm..."
    xcrun -sdk "${sdk}" clang++ \
        -isysroot "${sysroot}" \
        ${arch_flags} \
        ${min_version_flag} \
        ${CXX_STD} \
        ${COMMON_FLAGS} \
        ${COMMON_WARNS} \
        -fobjc-arc \
        ${include_flags} \
        -c "${BASE_DIR}/${COREML_MM}" \
        -o "${build_dir}/sortformer-coreml.o"

    # 3) Merge into static library (intermediate — will be converted to dynamic)
    echo "  Creating static library..."
    libtool -static \
        -o "${build_dir}/libsortformer.a" \
        "${build_dir}/sortformer.o" \
        "${build_dir}/sortformer-coreml.o" \
        2>/dev/null

    echo "  Done: ${build_dir}/libsortformer.a"
}

# ============================================================================
# Helper: create framework bundle (structure only — binary added by finalize)
# ============================================================================
setup_framework() {
    local build_dir="$1"
    local platform="$2"       # "macos" or "ios"
    local min_os_version="$3"

    local fw_dir="${build_dir}/framework/${FRAMEWORK_NAME}.framework"

    echo "--- Creating ${platform} framework bundle ---"

    if [[ "${platform}" == "macos" ]]; then
        # macOS versioned structure
        mkdir -p "${fw_dir}/Versions/A/Headers"
        mkdir -p "${fw_dir}/Versions/A/Modules"
        mkdir -p "${fw_dir}/Versions/A/Resources"

        ln -sf A "${fw_dir}/Versions/Current"
        ln -sf Versions/Current/Headers "${fw_dir}/Headers"
        ln -sf Versions/Current/Modules "${fw_dir}/Modules"
        ln -sf Versions/Current/Resources "${fw_dir}/Resources"
        ln -sf "Versions/Current/${FRAMEWORK_NAME}" "${fw_dir}/${FRAMEWORK_NAME}"

        local header_path="${fw_dir}/Versions/A/Headers"
        local module_path="${fw_dir}/Versions/A/Modules"
        local plist_path="${fw_dir}/Versions/A/Resources/Info.plist"
    else
        # iOS flat structure
        mkdir -p "${fw_dir}/Headers"
        mkdir -p "${fw_dir}/Modules"

        local header_path="${fw_dir}/Headers"
        local module_path="${fw_dir}/Modules"
        local plist_path="${fw_dir}/Info.plist"
    fi

    # Copy header (only sortformer.h — NOT ggml headers)
    cp "${BASE_DIR}/${SORTFORMER_H}" "${header_path}/"

    # Create module map
    cat > "${module_path}/module.modulemap" << 'MODULEMAP'
framework module sortformer {
    header "sortformer.h"
    link "c++"
    link framework "CoreML"
    link framework "Foundation"
    link framework "Accelerate"
    export *
}
MODULEMAP

    # Platform-specific Info.plist values
    local platform_name=""
    local sdk_name=""
    local supported_platform=""
    local device_family=""

    case "${platform}" in
        "macos")
            platform_name="macosx"
            sdk_name="macosx${min_os_version}"
            supported_platform="MacOSX"
            ;;
        "ios")
            platform_name="iphoneos"
            sdk_name="iphoneos${min_os_version}"
            supported_platform="iPhoneOS"
            device_family='
    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>'
            ;;
    esac

    cat > "${plist_path}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${FRAMEWORK_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>${min_os_version}</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>${supported_platform}</string>
    </array>${device_family}
    <key>DTPlatformName</key>
    <string>${platform_name}</string>
    <key>DTSDKName</key>
    <string>${sdk_name}</string>
</dict>
</plist>
EOF

    echo "  Framework structure created: ${fw_dir}"
}

# ============================================================================
# Helper: static -> dynamic library + dSYM + strip
# ============================================================================
finalize_framework() {
    local build_dir="$1"
    local sdk="$2"
    local archs="$3"
    local min_version_flag="$4"
    local platform="$5"       # "macos" or "ios"
    local is_simulator="$6"   # "true" or "false"

    echo "--- Finalizing ${platform} framework (dynamic lib + dSYM) ---"

    # Determine output path and install_name
    local output_lib=""
    local install_name=""
    if [[ "${platform}" == "macos" ]]; then
        output_lib="${build_dir}/framework/${FRAMEWORK_NAME}.framework/Versions/A/${FRAMEWORK_NAME}"
        install_name="@rpath/${FRAMEWORK_NAME}.framework/Versions/Current/${FRAMEWORK_NAME}"
    else
        output_lib="${build_dir}/framework/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}"
        install_name="@rpath/${FRAMEWORK_NAME}.framework/${FRAMEWORK_NAME}"
    fi

    local sysroot
    sysroot="$(xcrun --sdk "${sdk}" --show-sdk-path)"

    local arch_flags=""
    for arch in ${archs}; do
        arch_flags="${arch_flags} -arch ${arch}"
    done

    local temp_dir="${build_dir}/temp"
    mkdir -p "${temp_dir}"

    # 1) Create dynamic library from static library
    #    -undefined dynamic_lookup: ggml symbols resolved at runtime from whisper.framework
    echo "  Creating dynamic library..."
    xcrun -sdk "${sdk}" clang++ -dynamiclib \
        -isysroot "${sysroot}" \
        ${arch_flags} \
        ${min_version_flag} \
        -Wl,-force_load,"${BASE_DIR}/${build_dir}/libsortformer.a" \
        -framework CoreML -framework Foundation -framework Accelerate \
        -undefined dynamic_lookup \
        -install_name "${install_name}" \
        -o "${output_lib}"

    # 2) Mark binary as framework binary for device builds (vtool)
    if [[ "${is_simulator}" == "false" ]]; then
        if command -v xcrun vtool &>/dev/null; then
            case "${platform}" in
                "ios")
                    echo "  Marking binary as iOS framework..."
                    xcrun vtool -set-build-version ios ${IOS_MIN_OS_VERSION} ${IOS_MIN_OS_VERSION} -replace \
                        -output "${output_lib}" "${output_lib}"
                    ;;
            esac
        fi
    fi

    # 3) Generate dSYM
    echo "  Generating dSYM..."
    mkdir -p "${build_dir}/dSYMs"
    xcrun dsymutil "${output_lib}" -o "${build_dir}/dSYMs/${FRAMEWORK_NAME}.dSYM"

    # 4) Strip debug symbols from binary (dSYM already extracted)
    echo "  Stripping debug symbols..."
    cp "${output_lib}" "${temp_dir}/binary_to_strip"
    xcrun strip -S "${temp_dir}/binary_to_strip" -o "${temp_dir}/stripped_lib"
    mv "${temp_dir}/stripped_lib" "${output_lib}"

    # 5) Remove any auto-generated dSYM in framework dir (causes Invalid Bundle Structure)
    if [ -d "${output_lib}.dSYM" ]; then
        rm -rf "${output_lib}.dSYM"
    fi

    # Clean up
    rm -rf "${temp_dir}"
    echo "  Done: ${output_lib}"
}

# ============================================================================
# Build for macOS (arm64 + x86_64)
# ============================================================================
compile_sortformer \
    "build-sortformer-macos" \
    "macosx" \
    "arm64 x86_64" \
    "-mmacosx-version-min=${MACOS_MIN_OS_VERSION}"

setup_framework \
    "build-sortformer-macos" \
    "macos" \
    "${MACOS_MIN_OS_VERSION}"

finalize_framework \
    "build-sortformer-macos" \
    "macosx" \
    "arm64 x86_64" \
    "-mmacosx-version-min=${MACOS_MIN_OS_VERSION}" \
    "macos" \
    "false"

# ============================================================================
# Build for iOS device (arm64)
# ============================================================================
compile_sortformer \
    "build-sortformer-ios-device" \
    "iphoneos" \
    "arm64" \
    "-mios-version-min=${IOS_MIN_OS_VERSION}"

setup_framework \
    "build-sortformer-ios-device" \
    "ios" \
    "${IOS_MIN_OS_VERSION}"

finalize_framework \
    "build-sortformer-ios-device" \
    "iphoneos" \
    "arm64" \
    "-mios-version-min=${IOS_MIN_OS_VERSION}" \
    "ios" \
    "false"

# ============================================================================
# Build for iOS simulator (arm64 + x86_64)
# ============================================================================
compile_sortformer \
    "build-sortformer-ios-sim" \
    "iphonesimulator" \
    "arm64 x86_64" \
    "-mios-simulator-version-min=${IOS_MIN_OS_VERSION}"

setup_framework \
    "build-sortformer-ios-sim" \
    "ios" \
    "${IOS_MIN_OS_VERSION}"

finalize_framework \
    "build-sortformer-ios-sim" \
    "iphonesimulator" \
    "arm64 x86_64" \
    "-mios-simulator-version-min=${IOS_MIN_OS_VERSION}" \
    "ios" \
    "true"

# ============================================================================
# Create XCFramework (with dSYMs)
# ============================================================================
echo "=== Creating sortformer.xcframework ==="

# Remove previous xcframework if it exists (but don't nuke all of build-apple)
rm -rf build-apple/sortformer.xcframework

# Ensure build-apple directory exists
mkdir -p build-apple

xcodebuild -create-xcframework \
    -framework "$(pwd)/build-sortformer-macos/framework/${FRAMEWORK_NAME}.framework" \
    -debug-symbols "$(pwd)/build-sortformer-macos/dSYMs/${FRAMEWORK_NAME}.dSYM" \
    -framework "$(pwd)/build-sortformer-ios-device/framework/${FRAMEWORK_NAME}.framework" \
    -debug-symbols "$(pwd)/build-sortformer-ios-device/dSYMs/${FRAMEWORK_NAME}.dSYM" \
    -framework "$(pwd)/build-sortformer-ios-sim/framework/${FRAMEWORK_NAME}.framework" \
    -debug-symbols "$(pwd)/build-sortformer-ios-sim/dSYMs/${FRAMEWORK_NAME}.dSYM" \
    -output "$(pwd)/build-apple/${FRAMEWORK_NAME}.xcframework"

echo ""
echo "=== sortformer.xcframework created successfully ==="
echo "Location: build-apple/sortformer.xcframework/"
echo ""

# ============================================================================
# Verification
# ============================================================================
echo "=== Verifying symbols ==="

MACOS_LIB="build-apple/sortformer.xcframework/macos-arm64_x86_64/sortformer.framework/Versions/A/sortformer"

if [ ! -f "${MACOS_LIB}" ]; then
    echo "Error: macOS library not found at ${MACOS_LIB}"
    exit 1
fi

# Check for required sortformer symbols (should be defined = T)
echo "Checking sortformer API symbols..."
REQUIRED_SYMBOLS=(
    "_sortformer_init"
    "_sortformer_diarize"
    "_sortformer_to_rttm"
    "_sortformer_free"
)
for sym in "${REQUIRED_SYMBOLS[@]}"; do
    if nm -arch arm64 "${MACOS_LIB}" 2>/dev/null | grep -q " T ${sym}$"; then
        echo "  ✓ ${sym} (defined)"
    else
        echo "  ✗ ${sym} NOT FOUND as defined symbol"
        exit 1
    fi
done

# Check for required CoreML symbols (should be defined = T)
echo "Checking CoreML bridge symbols..."
COREML_SYMBOLS=(
    "_sortformer_coreml_init"
    "_sortformer_coreml_encode"
    "_sortformer_coreml_free"
)
for sym in "${COREML_SYMBOLS[@]}"; do
    if nm -arch arm64 "${MACOS_LIB}" 2>/dev/null | grep -q " T ${sym}$"; then
        echo "  ✓ ${sym} (defined)"
    else
        echo "  ✗ ${sym} NOT FOUND as defined symbol"
        exit 1
    fi
done

# Check that NO ggml symbols are DEFINED (T = text/code)
echo "Checking for ggml symbol leakage..."
DEFINED_GGML=$(nm -arch arm64 "${MACOS_LIB}" 2>/dev/null | grep " T _ggml_" | head -5)
if [ -n "${DEFINED_GGML}" ]; then
    echo "  ✗ ERROR: Found defined _ggml_ symbols (should only be undefined references):"
    echo "${DEFINED_GGML}"
    exit 1
else
    echo "  ✓ No _ggml_ symbols defined (good — they come from whisper.xcframework)"
fi

# Check that ggml symbols ARE referenced (U = undefined, resolved at runtime via whisper.framework)
UNDEF_GGML_COUNT=$(nm -arch arm64 "${MACOS_LIB}" 2>/dev/null | grep " U _ggml_" | wc -l | tr -d ' ')
echo "  ✓ ${UNDEF_GGML_COUNT} undefined _ggml_ references (resolved at runtime from whisper.framework)"

# Verify dSYMs are included in xcframework
echo ""
echo "Checking dSYMs in xcframework..."
for platform_dir in "macos-arm64_x86_64" "ios-arm64" "ios-arm64_x86_64-simulator"; do
    DSYM_PATH="build-apple/sortformer.xcframework/${platform_dir}/dSYMs/${FRAMEWORK_NAME}.dSYM"
    if [ -d "${DSYM_PATH}" ]; then
        echo "  ✓ dSYM present for ${platform_dir}"
    else
        echo "  ✗ dSYM MISSING for ${platform_dir}"
        exit 1
    fi
done

echo ""
echo "=== All verifications passed ==="
echo ""

# Clean up intermediate build directories
echo "Cleaning up intermediate build directories..."
rm -rf build-sortformer-macos
rm -rf build-sortformer-ios-device
rm -rf build-sortformer-ios-sim

echo "Done!"
