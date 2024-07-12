# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY States)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        States.hpp
        Types.hpp
        TypeTraits.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        States.hpp
        Types.hpp
        TypeTraits.hpp
)

add_subdirectory(Expressions)
add_library(${LIBRARY}::Expressions ALIAS StateExpressions)

add_subdirectory(TypeTraits)
add_library(${LIBRARY}::TypeTraits ALIAS StateTypeTraits)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        StateExpressions
        StateTypeTraits
)
