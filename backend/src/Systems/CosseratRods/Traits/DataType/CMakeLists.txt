# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodDataTraits)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        BlazeBackend/DataTypeTraits.hpp
        BlazeBackend/MatrixTag.hpp
        BlazeBackend/TensorTag.hpp
        BlazeBackend/VectorTag.hpp
        Protocols.hpp
        Rank.hpp
        ScalarTag.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        BlazeBackend/DataTypeTraits.hpp
        BlazeBackend/MatrixTag.hpp
        BlazeBackend/TensorTag.hpp
        BlazeBackend/VectorTag.hpp
        Protocols.hpp
        Rank.hpp
        ScalarTag.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        ModuleSettings
        Utilities
        Blaze
        BlazeTensor
)

