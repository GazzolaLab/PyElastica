# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodTraits)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        DataOpsTraits.hpp
        # PlacementTrait.hpp
        Protocols.hpp
        Types.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        DataOpsTraits.hpp
        # PlacementTrait.hpp
        Protocols.hpp
        Types.hpp
)

add_subdirectory(DataType)
add_library(${LIBRARY}::DataTraits ALIAS CosseratRodDataTraits)

# add_subdirectory(PlacementTraits)
# add_library(${LIBRARY}::PlacementTraits ALIAS CosseratRodPlacementTraits)

add_subdirectory(Operations)
add_library(${LIBRARY}::OperationTraits ALIAS CosseratRodOperationTraits)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        CosseratRodDataTraits
        # CosseratRodPlacementTraits
        CosseratRodOperationTraits
)

