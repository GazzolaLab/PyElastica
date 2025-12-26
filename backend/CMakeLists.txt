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

# Fetch Eigen3 for numerical computations
FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
# Disable Eigen3 tests and documentation
set(EIGEN_BUILD_DOC OFF)
# Set Eigen's default storage order to row-major
# This ensures Eigen's default behavior matches our explicit MatrixType template parameter
set(EIGEN_DEFAULT_TO_ROW_MAJOR ON CACHE BOOL "Use row-major as default matrix storage order" FORCE)
FetchContent_MakeAvailable(Eigen3)

# Restore BUILD_TESTING state for our tests
set(BUILD_TESTING ${BUILD_TESTING_SAVED})

# ------------------------------------------------------------ #

# Define directories #
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/py")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

option(ELASTICAPP_USE_THREADING "Enable threading support" ON)
option(ELASTICAPP_VECTORIZATION_REPORTS "Enable compiler vectorization reports" OFF)
option(ELASTICAPP_BUILD_TESTS "Build C++ unit tests" OFF)

# Option to set number of threads per component in update_dynamics parallel sections
# This controls how many threads are used for each spatial component (x, y, z) in the parallel sections
# Default is 3 (one thread per component), but can be adjusted for different hardware configurations
set(ELASTICAPP_COMPONENT_THREADS "2" CACHE STRING "Number of threads per component in update_dynamics parallel sections (default: 3)")
mark_as_advanced(ELASTICAPP_COMPONENT_THREADS)

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

    find_package(OpenMP COMPONENTS CXX REQUIRED)
endif()

# Helper function to apply optimization flags to a target
# This function must be defined before targets are created
function(apply_optimization_flags target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE -O3 -march=native)
        target_compile_options(${target_name} PRIVATE -ffunction-sections -fdata-sections)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${target_name} PRIVATE /O2)
    endif()
    # Add NDEBUG flag to disable assertions in release builds
    target_compile_definitions(${target_name} PRIVATE NDEBUG)

    # Enable Eigen vectorization
    target_compile_definitions(${target_name} PRIVATE EIGEN_VECTORIZE)

    # Add architecture-specific optimization flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # AVX2 and FMA are only available on x86_64, not ARM
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            target_compile_options(${target_name} PRIVATE -mavx2 -mfma)
        endif()

        # Add vectorization report flags if requested
        if(ELASTICAPP_VECTORIZATION_REPORTS)
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # Clang vectorization reports
                target_compile_options(${target_name} PRIVATE
                    -Rpass=loop-vectorize
                    -Rpass-missed=loop-vectorize
                    -Rpass-analysis=loop-vectorize
                )
                message(STATUS "Clang vectorization reports enabled for ${target_name}")
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                # GCC vectorization reports
                target_compile_options(${target_name} PRIVATE
                    -fopt-info-vec
                    -fopt-info-vec-missed
                    -fopt-info-vec-all
                )
                message(STATUS "GCC vectorization reports enabled for ${target_name}")
            endif()
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC flags for AVX2
        target_compile_options(${target_name} PRIVATE /arch:AVX2)
    endif()

    if(ELASTICAPP_USE_THREADING AND OpenMP_CXX_FOUND)
        target_compile_definitions(${target_name} PRIVATE ELASTICAPP_USE_THREADING)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "Threading support enabled for ${target_name}")
    endif()
endfunction()

# Extension module: _memory_block (Python module name)
pybind11_add_module(_memory_block
    "${SRC_DIR}/_api.cpp"
)
target_compile_definitions(_memory_block PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(_memory_block PRIVATE ${SRC_DIR})
target_link_libraries(_memory_block PRIVATE Eigen3::Eigen)

if(APPLE)
    target_link_options(_memory_block PRIVATE -Wl,-dead_strip)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_options(_memory_block PRIVATE -Wl,--gc-sections)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_options(_memory_block PRIVATE -Wl,--gc-sections)
endif()


# Apply component threads option to _memory_block (only if explicitly set and non-empty)
# If not set, the code will use OpenMP's default thread count
if(ELASTICAPP_COMPONENT_THREADS AND NOT ELASTICAPP_COMPONENT_THREADS STREQUAL "" AND NOT ELASTICAPP_COMPONENT_THREADS STREQUAL "0")
    target_compile_definitions(_memory_block PRIVATE ELASTICAPP_COMPONENT_THREADS=${ELASTICAPP_COMPONENT_THREADS})
endif()

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

# Install Python package files
install(FILES
        "${PYTHON_PACKAGE_DIR}/__init__.py"
        "${PYTHON_PACKAGE_DIR}/memory_block_rod.py"
        DESTINATION "elasticapp"
        COMPONENT python)

# C++ unit tests with Catch2
if(ELASTICAPP_BUILD_TESTS)
    enable_testing()
    include(CTest)

    # Fetch Catch2 for C++ unit testing
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4
    )
    set(CATCH_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(CATCH_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Catch2)

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

    # Apply optimization flags to cpp_tests
    apply_optimization_flags(cpp_tests)

    # Use Catch2's test discovery to automatically register tests with CTest
    # Add Catch2 extras directory to module path (FetchContent uses uppercase)
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(cpp_tests)
else()
    message(STATUS "C++ tests disabled (ELASTICAPP_BUILD_TESTS=OFF)")
endif()
