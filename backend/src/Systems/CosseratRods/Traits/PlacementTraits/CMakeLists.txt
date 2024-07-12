# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodPlacementTraits)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        PlacementTraits.hpp
        Protocols.hpp
        TypeTraits.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        PlacementTraits.hpp
        Protocols.hpp
        TypeTraits.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        ErrorHandling
)
