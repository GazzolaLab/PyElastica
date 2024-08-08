# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodOperationTraits)

add_elastica_library(${LIBRARY} INTERFACE)

add_subdirectory(BlazeBackend)
add_library(${LIBRARY}::Blaze ALIAS CosseratRodBlazeOperations)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        CosseratRodBlazeOperations
)
