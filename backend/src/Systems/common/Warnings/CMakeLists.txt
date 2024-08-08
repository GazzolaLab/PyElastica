# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY SystemsWarnings)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}
        HEADERS
        UserWarnings.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        UserWarnings.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        ModuleSettings
)
