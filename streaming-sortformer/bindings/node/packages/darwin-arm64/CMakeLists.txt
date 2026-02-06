cmake_minimum_required(VERSION 3.15)
project(sortformer-addon)

# NAPI version
add_compile_definitions(NAPI_VERSION=7)

# Find node-addon-api include path
execute_process(COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})

# Paths to main project
set(SORTFORMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(SORTFORMER_LIB "${SORTFORMER_ROOT}/build/libsortformer.a")
set(SORTFORMER_COREML_LIB "${SORTFORMER_ROOT}/build/libsortformer-coreml.a")
set(GGML_LIB "${SORTFORMER_ROOT}/build/ggml/src/libggml.a")
set(GGML_BASE_LIB "${SORTFORMER_ROOT}/build/ggml/src/libggml-base.a")
set(GGML_CPU_LIB "${SORTFORMER_ROOT}/build/ggml/src/libggml-cpu.a")
set(SORTFORMER_INCLUDE "${SORTFORMER_ROOT}/src")
set(GGML_INCLUDE "${SORTFORMER_ROOT}/ggml/include")

# Source files
add_library(${PROJECT_NAME} SHARED
    src/addon.cpp
    src/SortformerModel.cpp
    src/DiarizeWorker.cpp
    src/StreamingSession.cpp
    ${CMAKE_JS_SRC}
)

# Output as .node file
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_JS_INC}
    ${NODE_ADDON_API_DIR}
    ${SORTFORMER_INCLUDE}
    ${GGML_INCLUDE}
)

# C++17 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Link Node.js library
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_JS_LIB})

# Link against libsortformer.a and its dependencies
# Order matters: sortformer depends on ggml, ggml depends on ggml-base and ggml-cpu
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${SORTFORMER_LIB}
    ${GGML_LIB}
    ${GGML_BASE_LIB}
    ${GGML_CPU_LIB}
)

# Apple-specific: CoreML integration
if(APPLE)
    # Enable CoreML
    target_compile_definitions(${PROJECT_NAME} PRIVATE SORTFORMER_USE_COREML)
    
    # Find Apple frameworks
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)
    
    # Link CoreML bridge library and frameworks
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ${SORTFORMER_COREML_LIB}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    
    # Enable Objective-C ARC for .mm files
    set_target_properties(${PROJECT_NAME} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    )
endif()

# Windows-specific: Generate node.lib
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
                    /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
