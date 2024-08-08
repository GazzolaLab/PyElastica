# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY StateExpressionBackends)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        blaze.hpp
        Declarations.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        blaze.hpp
        Declarations.hpp
)

add_subdirectory(blaze)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        SO3PrimitiveAddAssignKernel
)
