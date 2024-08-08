# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY StateExpressionTemplates)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        StateAddExpr.hpp
        StateExpr.hpp
        StateTimeDeltaMultExprBase.hpp
        StateTimeMultExpr.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        StateAddExpr.hpp
        StateExpr.hpp
        StateTimeDeltaMultExprBase.hpp
        StateTimeMultExpr.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        ErrorHandling
        StateTypeTraits
)
