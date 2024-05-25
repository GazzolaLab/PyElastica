# Checks supported compiler versions for C++14

# Distributed under the MIT License. See LICENSE for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.2)
        message(FATAL_ERROR "GCC version must be at least 5.2")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
        message(FATAL_ERROR "Clang version must be at least 3.5")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(FATAL_ERROR "Intel compiler is not supported.")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "AppleClang version must be at least 6.0")
    endif()
else()
    message(
        WARNING
            "The compiler ${CMAKE_CXX_COMPILER_ID} is unsupported compiler! "
            "Compilation has only been tested with Clang, and GCC.")
endif()

# cmake-format: off
set(CXX_COMPILER_CONFIG ${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION})
string(TOLOWER ${CXX_COMPILER_CONFIG} CXX_COMPILER_CONFIG)
log_info("CXX compiler info")
log_info("------------------")
log_info("CMAKE_CXX_COMPILER              ${CMAKE_CXX_COMPILER}")
log_info("CMAKE_CXX_COMPILER_ID           ${CMAKE_CXX_COMPILER_ID}")
log_info("CMAKE_CXX_COMPILER_VERSION      ${CMAKE_CXX_COMPILER_VERSION}")
log_info("CXX_COMPILER_CONFIG             ${CXX_COMPILER_CONFIG}")
# cmake-format: on
