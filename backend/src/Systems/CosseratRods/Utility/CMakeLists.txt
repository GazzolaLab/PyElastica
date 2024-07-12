# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodUtility)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        CenterOfMass.hpp
        Energy.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        CenterOfMass.hpp
        Energy.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Blaze
        BlazeTensor
        Utilities
)
