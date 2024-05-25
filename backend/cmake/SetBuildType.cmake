# Set type of build
# Distributed under the MIT License. See LICENSE for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

# CMake lets the user define CMAKE_BUILD_TYPE on the command line and recognizes
# the values "Debug", "Release", "RelWithDebInfo", and "MinSizeRel", which add
# specific compiler or linker flags.  CMake's default behavior is to add no
# additional compiler or linker flags if the user does not define
# CMAKE_BUILD_TYPE on the command line, or passes an unrecognized value.
# We add a sanity check that checks if CMAKE_BUILD_TYPE is one of the recognized
# values, and we also set the CMAKE_BUILD_TYPE to "Debug" if the user does not
# specify it on the command line.  In addition, we add "None" as a valid value
# for CMAKE_BUILD_TYPE whose behavior is to add no additional compiler or linker
# flags. This is done by defining the following flags analagous to those used by
# the other build types. Additional build types can be defined in a similar
# manner by defining the appropriate flags, and adding the name of the build
# type to CMAKE_BUILD_TYPES below.
set(CMAKE_CXX_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the compiler for Build type None."
        FORCE)

set(CMAKE_C_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the compiler for Build type None."
        FORCE)

set(CMAKE_EXE_LINKER_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the linker for Build type None."
        FORCE)

set(CMAKE_MODULE_LINKER_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the linker for Build type None."
        FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the linker for Build type None."
        FORCE)

set(CMAKE_STATIC_LINKER_FLAGS_NONE "" CACHE STRING
        "Additional flags used by the linker for Build type None."
        FORCE)

mark_as_advanced(
        CMAKE_CXX_FLAGS_NONE
        CMAKE_C_FLAGS_NONE
        CMAKE_EXE_LINKER_FLAGS_NONE
        CMAKE_MODULE_LINKER_FLAGS_NONE
        CMAKE_SHARED_LINKER_FLAGS_NONE
        CMAKE_STATIC_LINKER_FLAGS_NONE
)


set(CMAKE_BUILD_TYPES
        "Debug"
        "Release"
        "None"
        "RelWithDebInfo"
        "MinSizeRel")

if (NOT CMAKE_BUILD_TYPE)
    log_info("-> CMAKE_BUILD_TYPE not specified, setting to 'Debug'")
    set(CMAKE_BUILD_TYPE
            Debug
            CACHE STRING "Choose the type of build: ${CMAKE_BUILD_TYPES}" FORCE)
else ()
    if (NOT ${CMAKE_BUILD_TYPE} IN_LIST CMAKE_BUILD_TYPES)
        message(
                FATAL_ERROR
                "\n" "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}\n"
                "Valid values: ${CMAKE_BUILD_TYPES}\n")
    endif ()
    # message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif ()

# cmake-format: off
log_info("CMAKE_BUILD_TYPE                ${CMAKE_BUILD_TYPE}")
# cmake-format: on
