# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY SO3PrimitiveAddAssignKernel)

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        SO3PrimitiveAddAssign/BaseTemplate.hpp
        SO3PrimitiveAddAssign/Scalar.hpp
        SO3PrimitiveAddAssign/SIMD.hpp
        Resize.hpp
        Size.hpp
        SO3PrimitiveAddAssign.hpp
        SO3PrimitiveAssign.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        SO3PrimitiveAddAssign/BaseTemplate.hpp
        SO3PrimitiveAddAssign/Scalar.hpp
        SO3PrimitiveAddAssign/SIMD.hpp
        Resize.hpp
        Size.hpp
        SO3PrimitiveAddAssign.hpp
        SO3PrimitiveAssign.hpp
)


target_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        Blaze
        Frames
        ModuleSettings
)
