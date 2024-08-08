# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY Materials)

add_elastica_library(${LIBRARY})

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        PUBLIC
        Materials.hpp
        Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        PRIVATE
        Materials.cpp
)

target_link_libraries(
        ${LIBRARY}
        PUBLIC
        Blaze
        ErrorHandling
        Utilities
)
