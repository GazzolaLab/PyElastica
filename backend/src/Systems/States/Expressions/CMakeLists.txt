# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY StateExpressions)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        backends.hpp
        OrderTags.hpp
        SE3.hpp
        SO3.hpp
        States.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        backends.hpp
        OrderTags.hpp
        SE3.hpp
        SO3.hpp
        States.hpp
)

add_subdirectory(backends)
add_library(${LIBRARY}::Backends ALIAS StateExpressionBackends)

add_subdirectory(Expr)
add_library(${LIBRARY}::Expr ALIAS StateExpressionTemplates)

add_subdirectory(OrderTags)
add_library(${LIBRARY}::OrderTags ALIAS StateExpressionOrderTags)

add_subdirectory(SE3)
add_library(${LIBRARY}::SE3 ALIAS SE3Expressions)

add_subdirectory(SO3)
add_library(${LIBRARY}::SO3 ALIAS SO3Expressions)

add_subdirectory(States)
add_library(${LIBRARY}::States ALIAS StateTemplates)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        StateExpressionOrderTags
        StateExpressionTemplates
        StateExpressionBackends
        SE3Expressions
        SO3Expressions
        StateTemplates
)
