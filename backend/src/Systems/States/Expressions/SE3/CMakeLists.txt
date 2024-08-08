# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY SE3Expressions)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        SE3.hpp
        SE3Base.hpp
        SE3SE3AddExpr.hpp
        SE3TimeDeltaMultExpr.hpp
        Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        SE3.hpp
        SE3Base.hpp
        SE3SE3AddExpr.hpp
        SE3TimeDeltaMultExpr.hpp
        Types.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        Time
        StateTypeTraits
        StateExpressionOrderTags
        StateExpressionTemplates
)
