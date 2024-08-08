# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY InvRotateDivideKernel)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        BaseTemplate.hpp
        Blaze.hpp
        Scalar.hpp
        SIMD.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        BaseTemplate.hpp
        Blaze.hpp
        Scalar.hpp
        SIMD.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        Blaze
)
