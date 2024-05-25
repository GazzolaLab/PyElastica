# Distributed under the MIT License.
# See LICENSE.txt for details.

# Set up position independent code by default since this is required
# for our python libraries.
#
# Obtained with thanks from https://github.com/sxs-collaboration/spectre
#
# Using CMake's set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# enables -fPIC in shared libraries and -fPIE on executables. Here instead
# of a global property, we add it to the ElasticaFlags property which
# is used by our examples, and python builds internally.
set_property(TARGET ElasticaFlags
        APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>)
