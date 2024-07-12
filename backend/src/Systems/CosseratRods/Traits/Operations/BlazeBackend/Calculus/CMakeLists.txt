# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CalculusKernels)

add_elastica_library(${LIBRARY} INTERFACE)

add_subdirectory(KernelGenerators)
add_subdirectory(Quadrature)
add_subdirectory(Difference)

set(LIBRARY_SOURCES
        Kernels.hpp
        )

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        ${LIBRARY_SOURCES}
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        ${LIBRARY_SOURCES}
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        CalculusKernelGenerators
        CalculusDifference
        CalculusQuadrature
)
