# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY StateExpressionOrderTags)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        OrderTags.hpp
        Types.hpp
        TypeTraits.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        OrderTags.hpp
        Types.hpp
        TypeTraits.hpp
)
