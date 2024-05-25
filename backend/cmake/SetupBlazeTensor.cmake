# Distributed under the MIT License. See LICENSE.txt for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

log_info("Finding Blaze Tensor")

find_package(BlazeTensor 0.1 REQUIRED QUIET)

log_debug("BLAZE_TENSOR_INCLUDE_DIR: ${BLAZE_TENSOR_INCLUDE_DIR}")
log_debug("BLAZE_TENSOR_VERSION: ${BLAZE_TENSOR_VERSION}")

file(APPEND "${CMAKE_BINARY_DIR}/BuildInformation.txt"
        "BlazeTensor Version:  ${BLAZE_TENSOR_VERSION}\n")

#elastica_include_directories("${BLAZE_INCLUDE_DIR}")
add_library(BlazeTensor INTERFACE IMPORTED)
set_property(TARGET BlazeTensor PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${BLAZE_TENSOR_INCLUDE_DIR})

add_interface_lib_headers(
        TARGET BlazeTensor
        HEADERS
        blaze_tensor/Blaze.h
        blaze_tensor/math/DynamicTensor.h
        blaze_tensor/math/StaticTensor.h
        blaze_tensor/math/SubTensor.h
        blaze_tensor/math/views/ColumnSlice.h
        # for the version number in CMakelists
        blaze_tensor/system/Version.h
)

set_property(
        GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
        BlazeTensor
)
