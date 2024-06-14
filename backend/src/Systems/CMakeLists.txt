# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY Systems)

add_elastica_library(${LIBRARY} INTERFACE)

# add_subdirectory(Block)
# add_library(${LIBRARY}::Block ALIAS Block)

# add_subdirectory(common)
# add_library(${LIBRARY}::Common ALIAS SystemsCommon)

add_subdirectory(CosseratRods)
add_library(${LIBRARY}::CosseratRods ALIAS CosseratRods)

# add_subdirectory(Customization)
# add_library(${LIBRARY}::Customization ALIAS SystemsCustomization)

# add_subdirectory(Python)

# add_subdirectory(RigidBody)
# add_library(${LIBRARY}::RigidBody ALIAS RigidBody)

# add_subdirectory(States)
# add_library(${LIBRARY}::States ALIAS States)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        # Block.hpp
        CosseratRods.hpp
        # Protocols.hpp
        # RigidBody.hpp
        # Sphere.hpp
        # Systems.hpp
        # Tags.hpp
        # Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        # Block.hpp
        CosseratRods.hpp
        # Protocols.hpp
        # RigidBody.hpp
        # Sphere.hpp
        # Systems.hpp
        # Tags.hpp
        # Types.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        # Block
        # SystemsCommon
        CosseratRods
        # SystemsCustomization
        # RigidBody
        # States
)
