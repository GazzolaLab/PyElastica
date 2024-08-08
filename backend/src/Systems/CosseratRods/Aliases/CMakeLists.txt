# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodAliases)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        CosseratRod.hpp
        CosseratRodWithoutDamping.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        CosseratRod.hpp
        CosseratRodWithoutDamping.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        CosseratRodComponents
        CosseratRodTraits
)
