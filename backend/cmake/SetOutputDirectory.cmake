# Set the directory where the library and executables are placed The default can
# be overridden by specifying `-D CMAKE_RUNTIME_OUTPUT_DIRECTORY=/path/`

# Distributed under the MIT License. See LICENSE for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/bin/"
            CACHE STRING "Choose the directory where executables are placed" FORCE)
endif (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(
            CMAKE_LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib/"
            CACHE STRING "Choose the directory where shared libraries are placed" FORCE
    )
endif (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)

if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY
            "${CMAKE_BINARY_DIR}/lib/"
            CACHE STRING "Choose the directory where static libraries are placed" FORCE
    )
endif (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
