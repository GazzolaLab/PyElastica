# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY StateTemplates)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        States.hpp
        StatesBase.hpp
        StatesStatesAddExpr.hpp
        StatesTimeDeltaMultExpr.hpp
        Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        States.hpp
        StatesBase.hpp
        StatesStatesAddExpr.hpp
        StatesTimeDeltaMultExpr.hpp
        Types.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        ErrorHandling
        Time
        StateExpressionTemplates
        StateExpressionOrderTags
        StateTypeTraits
)
