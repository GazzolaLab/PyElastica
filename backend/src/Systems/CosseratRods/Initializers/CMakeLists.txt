# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodInitializers)
add_elastica_library(${LIBRARY} INTERFACE)

add_subdirectory(detail)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        ComposeInitializers.hpp
        Initializers.hpp
        StraightCosseratRod.hpp
        UserDefinedCosseratRod.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        ComposeInitializers.hpp
        Initializers.hpp
        StraightCosseratRod.hpp
        UserDefinedCosseratRod.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Options
        Materials
        CosseratRods
        Utilities
)
