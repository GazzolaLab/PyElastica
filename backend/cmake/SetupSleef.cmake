log_info("Finding Sleef")

find_package(Sleef QUIET)

if (SLEEF_FOUND)
    file(APPEND "${CMAKE_BINARY_DIR}/BuildInformation.txt"
            "Sleef Version:  ${SLEEF_VERSION}\n")

    add_library(Sleef INTERFACE IMPORTED)
    set_property(TARGET Sleef PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${SLEEF_INCLUDE_DIRS})
    set_property(TARGET Sleef
            APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${SLEEF_LIBRARIES})
    add_interface_lib_headers(
            TARGET Sleef
            HEADERS
            sleef.h
    )

    set_property(
            GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
            Sleef
    )

    log_debug("SLEEF_INCLUDE_DIRS: ${SLEEF_INCLUDE_DIRS}")
    log_debug("SLEEF_LIBRARIES: ${SLEEF_LIBRARIES}")
    log_debug("SLEEF_VERSION: ${SLEEF_VERSION}")
endif ()
