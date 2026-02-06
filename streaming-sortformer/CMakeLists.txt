cmake_minimum_required(VERSION 3.14)
project(sortformer-ggml C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable LTO for Release builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# Compiler warnings
if (NOT MSVC)
    add_compile_options(-Wall -Wextra)
endif()

# Force static libraries for this project
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Model has tensor names up to 67 chars; ggml default is 64
add_compile_definitions(GGML_MAX_NAME=128)

# Add ggml
add_subdirectory(ggml)

# Find threads
find_package(Threads REQUIRED)

# --- CoreML support (macOS/iOS only) ---
option(SORTFORMER_COREML "Enable CoreML acceleration on Apple platforms" OFF)

if (SORTFORMER_COREML)
    if (APPLE)
        find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
        find_library(COREML_FRAMEWORK CoreML REQUIRED)
        message(STATUS "CoreML support enabled")
        
        # Build CoreML bridge library
        add_library(sortformer-coreml STATIC
            src/coreml/sortformer-coreml.mm
        )
        
        target_include_directories(sortformer-coreml PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/coreml
        )
        
        target_link_libraries(sortformer-coreml PRIVATE
            ${FOUNDATION_FRAMEWORK}
            ${COREML_FRAMEWORK}
        )
        
        # Enable ARC for Objective-C++
        set_target_properties(sortformer-coreml PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
        )
    else()
        message(FATAL_ERROR "CoreML is only available on Apple platforms")
    endif()
endif()

# --- libsortformer (static library) ---
add_library(sortformer STATIC
    src/sortformer.cpp
)

target_include_directories(sortformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)

target_compile_options(sortformer PRIVATE -march=native)

target_link_libraries(sortformer PUBLIC
    ggml
    Threads::Threads
)

# Link CoreML if enabled
if (SORTFORMER_COREML)
    target_compile_definitions(sortformer PUBLIC SORTFORMER_USE_COREML)
    target_include_directories(sortformer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/coreml)
    target_link_libraries(sortformer PUBLIC sortformer-coreml)
endif()

# --- sortformer-diarize (CLI binary) ---
add_executable(sortformer-diarize
    src/sortformer-cli.cpp
)

target_compile_options(sortformer-diarize PRIVATE -march=native)

target_link_libraries(sortformer-diarize PRIVATE
    sortformer
)

# --- sortformer-quantize (quantization tool) ---
add_executable(sortformer-quantize
    tools/quantize.cpp
)
target_include_directories(sortformer-quantize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
)
target_compile_options(sortformer-quantize PRIVATE -march=native)
target_link_libraries(sortformer-quantize PRIVATE
    ggml
)

# --- test-streaming-api (streaming API test binary) ---
add_executable(test-streaming-api
    src/test-streaming-api.cpp
)

target_compile_options(test-streaming-api PRIVATE -march=native)

target_link_libraries(test-streaming-api PRIVATE
    sortformer
)
