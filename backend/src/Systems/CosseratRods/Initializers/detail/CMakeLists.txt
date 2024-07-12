# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY CosseratRodInitializers)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        Affordances.hpp
        Common.hpp
        Elasticity.hpp
        Geometry.hpp
        Kinematics.hpp
        Transformations.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        Affordances.hpp
        Common.hpp
        Elasticity.hpp
        Geometry.hpp
        Kinematics.hpp
        Transformations.hpp
)
