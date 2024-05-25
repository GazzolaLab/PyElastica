# Setup TBB in elastica environment

# Distributed under the MIT License. See LICENSE for details.

log_info("Finding TBB")

# TBB >= 2021 required because variadic support was added in this version
# for parallel invoke, which is needed by some of RunTests.
find_package(TBB 2021.0 REQUIRED)

file(APPEND "${CMAKE_BINARY_DIR}/BuildInformation.txt"
        "TBB Version:  ${TBB_VERSION}\n")

add_library(TBB INTERFACE IMPORTED)
set_property(TARGET TBB
        APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${TBB_INCLUDE_DIRS} ${TBB_MALLOC_INCLUDE_DIRS})
set_property(TARGET TBB
        APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${TBB_LIBRARIES} ${TBB_MALLOC_LIBRARIES})

set_property(
        GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
        TBB
)

#elastica_include_directories("${TBB_INCLUDE_DIRS}")
#elastica_include_directories("${TBB_MALLOC_INCLUDE_DIRS}")
#list(APPEND ELASTICA_LIBRARIES ${TBB_LIBRARIES})
#list(APPEND ELASTICA_LIBRARIES ${TBB_MALLOC_LIBRARIES})

log_debug("TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
log_debug("TBB_LIBRARIES: ${TBB_LIBRARIES}")
log_debug("TBB_MALLOC_INCLUDE_DIRS: ${TBB_MALLOC_INCLUDE_DIRS}")
log_debug("TBB_MALLOC_LIBRARIES: ${TBB_MALLOC_LIBRARIES}")
log_debug("TBB_VERSION: ${TBB_VERSION}")
