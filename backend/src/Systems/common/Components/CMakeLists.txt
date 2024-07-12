# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY Components)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        Component.hpp
        GeometryComponent.hpp
        KinematicsComponent.hpp
        NameComponent.hpp
        NameAdapter.hpp
        Types.hpp
        TypeTraits.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        Component.hpp
        GeometryComponent.hpp
        KinematicsComponent.hpp
        NameComponent.hpp
        NameAdapter.hpp
        Types.hpp
        TypeTraits.hpp
)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
)
