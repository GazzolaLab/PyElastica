# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY BlockConcepts)

add_elastica_library(${LIBRARY} INTERFACE)

set(LIBRARY_SOURCES
        Gettable.hpp
        Sliceable.hpp
        Spannable.hpp
        Subdividable.hpp
        Types.hpp
        Viewable.hpp
        )

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        ${LIBRARY_SOURCES}
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        ${LIBRARY_SOURCES}
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        Stl
)
