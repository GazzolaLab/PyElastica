# Distributed under the MIT License. See LICENSE.txt for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

option(USE_SLEEF "Use Sleef to add more vectorized instructions." OFF)

log_info("Finding Blaze")

find_package(Blaze 3.9 REQUIRED QUIET)

log_debug("BLAZE_INCLUDE_DIR: ${BLAZE_INCLUDE_DIR}")
log_debug("BLAZE_VERSION: ${BLAZE_VERSION}")

file(APPEND "${CMAKE_BINARY_DIR}/BuildInformation.txt"
        "Blaze Version:  ${BLAZE_VERSION}\n")

#elastica_include_directories("${BLAZE_INCLUDE_DIR}")
add_library(Blaze INTERFACE IMPORTED)
set_property(TARGET Blaze PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_INCLUDE_DIR})

set(_BLAZE_USE_SLEEF 0)

if (USE_SLEEF)
    # Try to find Sleef to increase vectorization
    include(SetupSleef)
endif ()

if (SLEEF_FOUND)
    target_link_libraries(
            Blaze
            INTERFACE
            Sleef
    )
    set(_BLAZE_USE_SLEEF 1)
endif ()

# Configure Blaze. Some of the Blaze configuration options could be optimized
# for the machine we are running on. See documentation:
# https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20and%20Installation#!step-2-configuration
target_compile_definitions(Blaze
        INTERFACE
        # - Enable external BLAS kernels
        BLAZE_BLAS_MODE=0
        # - Set default matrix storage order to column-major, since many of our
        #   functions are implemented for column-major layout. This default reduces
        #   conversions.
        BLAZE_DEFAULT_STORAGE_ORDER=blaze::rowMajor
        # - Disable SMP parallelization. This disables SMP parallelization for all
        #   possible backends (OpenMP, C++11 threads, Boost, HPX):
        #   https://bitbucket.org/blaze-lib/blaze/wiki/Serial%20Execution#!option-3-deactivation-of-parallel-execution
        BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0
        # - Disable MPI parallelization
        BLAZE_MPI_PARALLEL_MODE=0
        # - Using the default cache size, which may have been configured automatically
        #   by the Blaze CMake configuration for the machine we are running on. We
        #   could override it here explicitly to tune performance.
        # BLAZE_CACHE_SIZE
        BLAZE_USE_PADDING=1
        # - Always enable non-temporal stores for cache optimization of large data
        #   structures: https://bitbucket.org/blaze-lib/blaze/wiki/Configuration%20Files#!streaming-non-temporal-stores
        BLAZE_USE_STREAMING=1
        # - Initializing default-constructed structures for fundamental types
        BLAZE_USE_DEFAULT_INITIALIZATON=1
        # Use Sleef for vectorization of more math functions
        BLAZE_USE_SLEEF=${_BLAZE_USE_SLEEF}
        # Set inlining settings
        BLAZE_USE_STRONG_INLINE=1
        BLAZE_USE_ALWAYS_INLINE=1
        # Set vectorization (leave to 1, else there is no use for blaze)
        BLAZE_USE_VECTORIZATION=1
        )

add_interface_lib_headers(
        TARGET Blaze
        HEADERS
        blaze/math/DynamicMatrix.h
        blaze/math/DynamicVector.h
        blaze/math/StaticVector.h
        blaze/system/Version.h
)

set_property(
        GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
        Blaze
)
