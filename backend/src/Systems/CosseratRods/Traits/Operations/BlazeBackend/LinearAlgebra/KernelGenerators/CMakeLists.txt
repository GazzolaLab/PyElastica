# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY LinearAlgebraKernelGenerators)

add_elastica_library(${LIBRARY} INTERFACE)

add_subdirectory(MatVec)
add_subdirectory(VecScalar)
add_subdirectory(VecVec)

target_link_libraries(
        ${LIBRARY}
        INTERFACE
        MatVecKernelGenerators
        VecScalarKernelGenerators
        VecVecKernelGenerators
)
