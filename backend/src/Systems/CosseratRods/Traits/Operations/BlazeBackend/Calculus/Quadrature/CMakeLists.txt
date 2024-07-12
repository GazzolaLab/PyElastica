# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CalculusQuadrature)

add_elastica_library(${LIBRARY} INTERFACE)

set(LIBRARY_SOURCES
        BaseTemplate.hpp
        Blaze.hpp
        Kernels.hpp
        Operation.hpp
        Scalar.hpp
        SIMD.hpp
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
        ModuleSettings
        CalculusKernelGenerators
        Utilities
        Blaze
        Stl
)
