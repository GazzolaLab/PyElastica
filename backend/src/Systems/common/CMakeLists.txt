# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY SystemsCommon)

add_elastica_library(${LIBRARY} INTERFACE)

set(TARGET_SOURCES
        Access.hpp
        Cardinality.hpp
        Components.hpp
        IndexCheck.hpp
        Initialization.hpp
        Protocols.hpp
        SymplecticStepperAdapter.hpp
        Tags.hpp
        Types.hpp
        )

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        ${TARGET_SOURCES}
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        ${TARGET_SOURCES}
)

add_subdirectory(Components)
add_subdirectory(Warnings)
add_subdirectory(Python)
add_subdirectory(Traits)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Components
        SystemsWarnings
        SystemsCommonTraits
)
