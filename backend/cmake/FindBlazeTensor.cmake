# Distributed under the MIT License. See LICENSE.txt for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

include(GetEnvPath)

getenv_path(BLAZE_TENSOR_ROOT)

find_path(BLAZE_TENSOR_INCLUDE_DIR
        PATH_SUFFIXES include
        NAMES blaze_tensor/Blaze.h
        HINTS ${BLAZE_TENSOR_ROOT} ${ENV_BLAZE_TENSOR_ROOT}
        DOC "Blaze Tensor include directory. Used BLAZE_TENSOR_ROOT to set a search dir.")

set(BLAZE_TENSOR_INCLUDE_DIRS ${BLAZE_TENSOR_INCLUDE_DIR})
set(BLAZE_TENSOR_VERSION "")

if (EXISTS "${BLAZE_TENSOR_INCLUDE_DIRS}/blaze_tensor/system/Version.h")
    # Extract version info from header
    file(READ "${BLAZE_TENSOR_INCLUDE_DIRS}/blaze_tensor/system/Version.h"
            BLAZE_TENSOR_FIND_HEADER_CONTENTS)

    string(REGEX MATCH
            "#define BLAZE_TENSOR_MAJOR_VERSION [0-9]+"
            BLAZE_TENSOR_MAJOR_VERSION
            "${BLAZE_TENSOR_FIND_HEADER_CONTENTS}")
    string(REPLACE "#define BLAZE_TENSOR_MAJOR_VERSION "
            ""
            BLAZE_TENSOR_MAJOR_VERSION
            ${BLAZE_TENSOR_MAJOR_VERSION})

    string(REGEX MATCH
            "#define BLAZE_TENSOR_MINOR_VERSION [0-9]+"
            BLAZE_TENSOR_MINOR_VERSION
            "${BLAZE_TENSOR_FIND_HEADER_CONTENTS}")
    string(REPLACE "#define BLAZE_TENSOR_MINOR_VERSION "
            ""
            BLAZE_TENSOR_MINOR_VERSION
            ${BLAZE_TENSOR_MINOR_VERSION})

    set(BLAZE_TENSOR_VERSION "${BLAZE_TENSOR_MAJOR_VERSION}.${BLAZE_TENSOR_MINOR_VERSION}")

else ()
    message(WARNING "Failed to find file "
            "'${BLAZE_TENSOR_INCLUDE_DIRS}/blaze_tensor/system/Version.h' "
            "while detecting the BlazeTensor version.")
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BlazeTensor
        FOUND_VAR
        BLAZETENSOR_FOUND
        REQUIRED_VARS
        BLAZE_TENSOR_INCLUDE_DIR
        BLAZE_TENSOR_INCLUDE_DIRS
        VERSION_VAR
        BLAZE_TENSOR_VERSION)
mark_as_advanced(BLAZE_TENSOR_INCLUDE_DIR
        BLAZE_TENSOR_INCLUDE_DIRS
        BLAZE_TENSOR_VERSION
        BLAZE_TENSOR_MAJOR_VERSION
        BLAZE_TENSOR_MINOR_VERSION)
