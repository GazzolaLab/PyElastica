# Distributed under the MIT License.
# See LICENSE.txt for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

# Adds the header files to the target.
#
# Usage:
#
#   add_interface_lib_headers(
#     TARGET TARGET_NAME
#     HEADERS
#     A.hpp
#     B.hpp
#     C.hpp
#     )
#
# This function is intended to be used with libraries added using add_library
# or added by CMake's provided find_package (e.g. Boost). The
# add_spectre_library handles adding header files for targets correctly and
# so this function does not need to be used for libraries added with
# add_spectre_library.
function(add_interface_lib_headers)
    cmake_parse_arguments(
            ARG "" "TARGET" "HEADERS"
            ${ARGN})

    if (NOT TARGET ${ARG_TARGET})
        message(FATAL_ERROR
                "Unknown target '${ARG_TARGET}'"
                )
    endif (NOT TARGET ${ARG_TARGET})

    get_target_property(
            TARGET_TYPE
            ${ARG_TARGET}
            TYPE
    )
    if (NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
        message(FATAL_ERROR
                "The target '${ARG_TARGET}' is not an INTERFACE library and so "
                "add_interface_lib_headers should not be used to add header files "
                "to it."
                )
    endif (NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)

    get_property(
            ELASTICA_INTERFACE_LIBRARY_HEADERS
            GLOBAL
            PROPERTY ELASTICA_INTERFACE_LIBRARY_HEADERS
    )

    # Switch to delimiting the headers by ':' because CMake uses ';' to delimit
    # elements of a list.
    string(REPLACE ";" ":" TARGET_HEADERS
            "${ARG_TARGET}=${ARG_HEADERS}")
    list(APPEND ELASTICA_INTERFACE_LIBRARY_HEADERS ${TARGET_HEADERS})

    set_property(
            GLOBAL PROPERTY ELASTICA_INTERFACE_LIBRARY_HEADERS
            ${ELASTICA_INTERFACE_LIBRARY_HEADERS}
    )
endfunction(add_interface_lib_headers)

# Returns a list of all the header files for the target `TARGET`
# by setting the variable with the name `${RESULT_NAME}`
#
# Usage:
#   get_target_headers(MyTarget MY_TARGET_HEADERS)
function(get_target_headers TARGET RESULT_NAME)
    get_target_property(
            TARGET_TYPE
            ${TARGET}
            TYPE
    )
    if (${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
        get_property(
                _ELASTICA_INTERFACE_LIBRARY_HEADERS
                GLOBAL
                PROPERTY ELASTICA_INTERFACE_LIBRARY_HEADERS
        )
        foreach (_LIB ${_ELASTICA_INTERFACE_LIBRARY_HEADERS})
            string(REPLACE "${TARGET}=" "" _LIB_HEADERS ${_LIB})
            if (NOT ${_LIB} STREQUAL ${_LIB_HEADERS})
                string(REPLACE ":" ";" _LIB_HEADERS ${_LIB_HEADERS})
                set(${RESULT_NAME} ${_LIB_HEADERS} PARENT_SCOPE)
                break()
            endif (NOT ${_LIB} STREQUAL ${_LIB_HEADERS})
        endforeach (_LIB ${_ELASTICA_INTERFACE_LIBRARY_HEADERS})

    else (${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
        get_property(
                _HEADER_FILES
                TARGET ${TARGET}
                PROPERTY PUBLIC_HEADER
        )
        set(${RESULT_NAME} ${_HEADER_FILES} PARENT_SCOPE)
    endif (${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
endfunction(get_target_headers TARGET)
