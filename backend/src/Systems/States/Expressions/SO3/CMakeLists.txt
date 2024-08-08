# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY SO3Expressions)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        SO3.hpp
        SO3AddExpr.hpp
        SO3Base.hpp
        SO3RotRotAddExpr.hpp
        SO3SO3AddExpr.hpp
        SO3TimeDeltaMultExpr.hpp
        Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        SO3.hpp
        SO3AddExpr.hpp
        SO3Base.hpp
        SO3RotRotAddExpr.hpp
        SO3SO3AddExpr.hpp
        SO3TimeDeltaMultExpr.hpp
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
        StateExpressionBackends
)
