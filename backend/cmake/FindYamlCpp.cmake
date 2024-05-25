# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find yaml-cpp: https://github.com/jbeder/yaml-cpp
# If not in one of the default paths specify -D YAMLCPP_ROOT=/path/to/yaml-cpp
# to search there as well.
# Static libraries can be use by setting -D YAMLCPP_STATIC_LIBRARY=ON

include(CheckCXXSourceRuns)
include(GetEnvPath)

# Get path, convert backslashes as ${ENV_${var}}
getenv_path(YamlCpp_ROOT)

if (NOT YamlCpp_ROOT)
    # Need to set to empty to avoid warnings with --warn-uninitialized
    set(YamlCpp_ROOT "")
endif ()

# find the yaml-cpp include directory
find_path(YamlCpp_INCLUDE_DIRS
        PATH_SUFFIXES include
        NAMES yaml-cpp/yaml.h
        HINTS ${YamlCpp_ROOT} ${ENV_YamlCpp_ROOT}
        DOC "YamlCpp include directory. Use YamlCpp_ROOT to set a search dir.")

if (YamlCpp_STATIC_LIBRARY)
    set(YamlCpp_STATIC libyaml-cpp.a)
else (YamlCpp_STATIC_LIBRARY)
    # Silence CMake uninitialized variable warning
    set(YamlCpp_STATIC "")
endif ()

find_library(YamlCpp_LIBRARIES
        PATH_SUFFIXES lib64 lib build
        NAMES ${YamlCpp_STATIC} yaml-cpp
        HINTS ${YamlCpp_ROOT} ${ENV_YamlCpp_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(YamlCpp
        DEFAULT_MSG YamlCpp_INCLUDE_DIRS YamlCpp_LIBRARIES)
mark_as_advanced(YamlCpp_INCLUDE_DIRS YamlCpp_LIBRARIES)
