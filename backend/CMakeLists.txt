# Automagically build elastica along with its dependencies

# ##############################################################################
# Distributed under the MIT License. See LICENSE for details.
# ##############################################################################
cmake_minimum_required(VERSION 3.6...3.18)

# ##############################################################################
# Set policies
# ##############################################################################
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()
if (POLICY CMP0110)
    # disable support for arbitrary characters (such as whitespace) in test names
    # but rather use underscores
    cmake_policy(SET CMP0110 OLD)
endif ()

# ##############################################################################
# Project details
# Versioning
# ##############################################################################
set(ELASTICA_VERSION "0.0.3")
set(ELASTICA_DESCRIPTION "Package to efficiently simulate soft-filaments on devices from your laptop to supercomputing clusters.")

# C is added for the HDF5 dependency. HighFive also relies on C headers.
project(elastica
        LANGUAGES CXX C
        VERSION ${ELASTICA_VERSION}
        DESCRIPTION ${ELASTICA_DESCRIPTION})


# ##############################################################################
# Setting the directory which contains CMake modules
# ##############################################################################
# Appends the cmake path inside the MAKE_MODULE_PATH variable which stores the
# list of directories where to lookk for additional CMake modules.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Create a new (or overwrite existing) info file at start of configuration
file(WRITE "${CMAKE_BINARY_DIR}/BuildInformation.txt" "")

# ##############################################################################
# Including cmake only modules
# ##############################################################################
include(Logging)
set_logs_on()
set_log_level("DEBUG") # INFO  DEBUG  HEAVYDEBUG

option(ELASTICA_BUILD_PYTHON_BINDINGS "Build the python bindings for elastica" ON)

# Define standard installation directories before since python targets utilize
include(GNUInstallDirs)

# Disable `make install` depending on `make all` since we want to control what
# we install more closely. With this setting, and targets marked as `OPTIONAL`,
# only targets that were built will be installed.
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)


include(CheckCompilerVersion)
#
include(ElasticaAddInterfaceLibraryHeaders)
#include(ElasticaTargetHeaders)
#include(ElasticaTargetSources)
#include(ElasticaAddLibraries)
include(ElasticaSetupFlagsTarget)
include(EnableWarnings)
# need flags to be defined

include(SetBuildType)
#include(SetupCraySupport)
include(SetCxxStandard)
include(SetupCxxFlags)
include(SetupElasticaInlining)
include(SetupMacOsx)
include(SetOutputDirectory)
include(SetupPic)
include(SetupSanitizers)

# In order to use certain code analysis tools like clang-tidy and cppcheck the
# compile commands need to be accessible. CMake can write these to a
# "compile_commands.json" file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Need to load python libs before boost.python. include(SetupPythonLibs)
# include(SetupBlas)  include(SetupBoost) include(SetupBrigand)
# include(SetupCatch) include(SetupGoogleBenchmark) include(SetupGsl)
# include(SetupHdf5) include(SetupAllocator) include(SetupLIBXSMM)
# include(SetupLapack) include(SetupNumPy) include(SetupOpenMP)
# include(SetupPapi) include(SetupSciPy) include(SetupLibsharp)

include(SetupBlaze)
include(SetupBlazeTensor)
#include(SetupBrigand)
#include(SetupProfiling)
#include(SetupHighFive)
#include(SetupSMP)
include(SetupStl)
#include(SetupYamlCpp)
include(SetupFilesystem)

if (ELASTICA_BUILD_PYTHON_BINDINGS)
    include(SetupPybind11)
endif ()
#include(ElasticaSetupPythonPackage)

#add_subdirectory(elastica)


include(PrintUsefulCMakeInfo)
#include(ElasticaInstall)
