cmake_minimum_required(VERSION 3.20)
project(elasticapp VERSION 0.0.4 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
add_compile_options(-Wno-unused)

# Find Python and pybind11 (temp)
# pybind11 will find Python and NumPy automatically
# Set Python interpreter to ../.venv
set(Python3_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.venv")
set(PYTHON_EXECUTABLE "${Python3_ROOT_DIR}/bin/python")
# set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")

# Find pybind11 and ensure it uses the Python interpreter from our venv
set(PYBIND11_FINDPYTHON "ON")
set(PYBIND11_PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "" FORCE)
find_package(pybind11 REQUIRED)

# ------------------------------------------------------------ #
# External dependency management #
# Include FetchContent for Catch2 and Eigen3
include(FetchContent)

# Save current BUILD_TESTING state and disable it for dependencies
set(BUILD_TESTING_SAVED ${BUILD_TESTING})
set(BUILD_TESTING OFF)

# Fetch Catch2 for C++ unit testing
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
)
set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CATCH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Catch2)

# Fetch Eigen3 for numerical computations
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
# Disable Eigen3 tests and documentation
set(EIGEN_BUILD_DOC OFF)
FetchContent_MakeAvailable(Eigen3)

# Restore BUILD_TESTING state for our tests
set(BUILD_TESTING ${BUILD_TESTING_SAVED})

# ------------------------------------------------------------ #

# Define directories #
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/py")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Option to enable SIMD optimizations
option(ELASTICAPP_USE_SIMD "Enable SIMD optimizations" ON)

# Option to enable threading support
option(ELASTICAPP_USE_THREADING "Enable threading support" ON)

# Option to set default number of threads (0 = use OpenMP default, typically all CPU cores)
# Note: This is a compile-time constant, set via -DELASTICAPP_NUM_THREADS=N at configure time
# For runtime control, use the Python/C++ API or OMP_NUM_THREADS environment variable
set(ELASTICAPP_NUM_THREADS "0" CACHE STRING "Default number of OpenMP threads (0 = use OpenMP default, typically all CPU cores)")
mark_as_advanced(ELASTICAPP_NUM_THREADS)

# Configure threading support (OpenMP) - check early but apply per-target
if(ELASTICAPP_USE_THREADING)
    # On macOS with Homebrew, help CMake find libomp
    # This is needed for both Makefile builds and scikit-build-core builds
    if(APPLE)
        # Try to find Homebrew-installed libomp
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(HOMEBREW_LIBOMP_PREFIX)
            message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
            # Set OpenMP variables for clang on macOS
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
                set(OpenMP_CXX_LIBRARIES "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
                message(STATUS "Configured OpenMP for clang with Homebrew libomp")
            endif()
        endif()
    endif()

    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
        message(STATUS "Threading support enabled")
        if(ELASTICAPP_NUM_THREADS)
            message(STATUS "Default number of threads: ${ELASTICAPP_NUM_THREADS}")
        else()
            message(STATUS "Number of threads: use OpenMP default (typically all CPU cores)")
        endif()
    else()
        message(FATAL_ERROR
            "ELASTICAPP_USE_THREADING is ON but OpenMP was not found.\n"
            "Please install OpenMP:\n"
            "  - On macOS: brew install libomp\n"
            "  - On Ubuntu/Debian: sudo apt-get install libomp-dev\n"
            "  - On Fedora/RHEL: sudo dnf install libgomp-devel\n"
            "Or set ELASTICAPP_USE_THREADING=OFF to disable threading support."
        )
    endif()
endif()

# Helper function to apply optimization flags to a target
# This function must be defined before targets are created
function(apply_optimization_flags target_name)
    if(ELASTICAPP_USE_SIMD)
        # Enable Eigen vectorization
        target_compile_definitions(${target_name} PRIVATE EIGEN_VECTORIZE ELASTICAPP_USE_SIMD)

        # Add architecture-specific optimization flags
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            # -march=native optimizes for the current CPU
            target_compile_options(${target_name} PRIVATE -march=native)

            # AVX2 and FMA are only available on x86_64, not ARM
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
                target_compile_options(${target_name} PRIVATE -mavx2 -mfma)
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            # MSVC flags for AVX2
            target_compile_options(${target_name} PRIVATE /arch:AVX2)
        endif()
        message(STATUS "SIMD optimizations enabled for ${target_name}")
    endif()

    if(ELASTICAPP_USE_THREADING AND OpenMP_CXX_FOUND)
        target_compile_definitions(${target_name} PRIVATE ELASTICAPP_USE_THREADING)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        # Set default number of threads if specified (non-zero)
        if(ELASTICAPP_NUM_THREADS AND NOT ELASTICAPP_NUM_THREADS STREQUAL "0")
            target_compile_definitions(${target_name} PRIVATE ELASTICAPP_NUM_THREADS=${ELASTICAPP_NUM_THREADS})
        endif()
        message(STATUS "Threading support enabled for ${target_name}")
    endif()
endfunction()

# Static library for common C++ code
# This allows both Python modules and test executables to link against it
# add_library(elasticapp_core STATIC
# )
# target_include_directories(elasticapp_core PUBLIC ${SRC_DIR})
# target_link_libraries(elasticapp_core PUBLIC Eigen3::Eigen)

# Extension module: _memory_block (Python module name)
pybind11_add_module(_memory_block
    "${SRC_DIR}/_api.cpp"
)
target_compile_definitions(_memory_block PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(_memory_block PRIVATE ${SRC_DIR})
target_link_libraries(_memory_block PRIVATE Eigen3::Eigen)

# Apply optimization flags to _memory_block
apply_optimization_flags(_memory_block)

# Install _memory_block module
install(TARGETS _memory_block
        LIBRARY DESTINATION "elasticapp"
        COMPONENT python)

# Extension module: version (Python module name)
pybind11_add_module(version
    "${SRC_DIR}/_version.cpp"
)
target_compile_definitions(version PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(version PRIVATE ${SRC_DIR})

# Install version module
install(TARGETS version
        LIBRARY DESTINATION "elasticapp"
        COMPONENT python)

# Extension module: _testing
# pybind11_add_module(_testing
#     "${TEST_DIR}/cpp/_testing.cpp"
#     "${TEST_DIR}/cpp/mock_block_system.h"
# )
# target_compile_definitions(_testing PRIVATE VERSION_INFO=${PROJECT_VERSION})
# target_include_directories(_testing PRIVATE ${TEST_DIR} ${SRC_DIR})
# target_link_libraries(_testing PRIVATE Eigen3::Eigen elasticapp_core)

# Option to control matrix storage order (default: column-major)
option(ELASTICAPP_USE_ROW_MAJOR "Use row-major storage for Eigen matrices" OFF)

# Install elasticapp_core module
# install(TARGETS elasticapp_core
#         LIBRARY DESTINATION "elasticapp"
#         COMPONENT python)

# Install Python package files
install(FILES
        "${PYTHON_PACKAGE_DIR}/__init__.py"
        "${PYTHON_PACKAGE_DIR}/memory_block_rod.py"
        DESTINATION "elasticapp"
        COMPONENT python)

# C++ unit tests with Catch2
enable_testing()
include(CTest)

# Test executable
add_executable(cpp_tests
    "${TEST_DIR}/cpp/test_version.cpp"
    "${TEST_DIR}/cpp/test_block.cpp"
    "${TEST_DIR}/cpp/test_traits.cpp"
    "${TEST_DIR}/cpp/test_system.cpp"
    "${TEST_DIR}/cpp/test_system_variable_validation.cpp"
    "${TEST_DIR}/cpp/test_block_view.cpp"
    "${TEST_DIR}/cpp/test_block_operations.cpp"
)

target_compile_definitions(cpp_tests PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(cpp_tests PRIVATE ${SRC_DIR} ${TEST_DIR}/cpp/mock)
target_link_libraries(cpp_tests PRIVATE Catch2::Catch2WithMain Eigen3::Eigen)

# Apply options to test executable
if(ELASTICAPP_USE_ROW_MAJOR)
    target_compile_definitions(cpp_tests PRIVATE ELASTICAPP_USE_ROW_MAJOR=1)
endif()

# Apply optimization flags to cpp_tests
apply_optimization_flags(cpp_tests)

# Use Catch2's test discovery to automatically register tests with CTest
# Add Catch2 extras directory to module path (FetchContent uses uppercase)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(cpp_tests)
