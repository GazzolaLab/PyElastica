# Distributed under the MIT License.
# See LICENSE.txt for details.

log_info("Finding HDF5")

find_package(HDF5 COMPONENTS C)

log_debug("HDF5 library: ${HDF5_C_LIBRARIES}")
log_debug("HDF5 include: ${HDF5_C_INCLUDE_DIRS}")
log_debug("HDF5 version: ${HDF5_VERSION}")

file(APPEND
        "${CMAKE_BINARY_DIR}/BuildInformation.txt"
        "HDF5 version: ${HDF5_VERSION}\n"
        )

if (NOT TARGET hdf5::hdf5)
    add_library(hdf5::hdf5 INTERFACE IMPORTED)
    set_target_properties(
            hdf5::hdf5
            PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
    )
    if (DEFINED HDF5_C_DEFINITIONS)
        set_target_properties(
                hdf5::hdf5
                PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_DEFINITIONS}"
        )
    endif ()
endif ()

if (HDF5_IS_PARALLEL)
    message(WARNING "HDF5 is built with MPI support, but MPI was not found. "
            "You may encounter build issues with HDF5, such as missing headers.")
endif ()

set_property(
        GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
        hdf5::hdf5
)
