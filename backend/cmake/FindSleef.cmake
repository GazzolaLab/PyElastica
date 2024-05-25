# ===============================================
# Do the final processing for the package find.
# ===============================================
macro(findpkg_finish PREFIX)
    # skip if already processed during this run
    if (NOT ${PREFIX}_FOUND)
        if (${PREFIX}_INCLUDE_DIR AND ${PREFIX}_LIBRARY)
            set(${PREFIX}_FOUND TRUE)
            set(${PREFIX}_INCLUDE_DIRS ${${PREFIX}_INCLUDE_DIR})
            set(${PREFIX}_LIBRARIES ${${PREFIX}_LIBRARY})
        else ()
            if (${PREFIX}_FIND_REQUIRED AND NOT ${PREFIX}_FIND_QUIETLY)
                message(FATAL_ERROR "Required library ${PREFIX} not found.")
            endif ()
        endif ()

        # mark the following variables as internal variables
        mark_as_advanced(${PREFIX}_INCLUDE_DIR
                ${PREFIX}_LIBRARY
                ${PREFIX}_LIBRARY_DEBUG
                ${PREFIX}_LIBRARY_RELEASE)
    endif ()
endmacro()

# ===============================================
# See if we have env vars to help us find Sleef
# ===============================================
macro(getenv_path VAR)
    set(ENV_${VAR} $ENV{${VAR}})
    # replace won't work if var is blank
    if (ENV_${VAR})
        string(REGEX
                REPLACE "\\\\"
                "/"
                ENV_${VAR}
                ${ENV_${VAR}})
    endif ()
endmacro()

# =============================================================================
# Now to actually find Sleef
#

# Get path, convert backslashes as ${ENV_${var}}
getenv_path(SLEEF_ROOT)

# initialize search paths
set(SLEEF_PREFIX_PATH ${SLEEF_ROOT} ${ENV_SLEEF_ROOT})
set(SLEEF_INC_SEARCH_PATH "")
set(SLEEF_LIB_SEARCH_PATH "")

# If user built from sources
set(SLEEF_BUILD_PREFIX $ENV{SLEEF_BUILD_PREFIX})
if (SLEEF_BUILD_PREFIX AND ENV_SLEEF_ROOT)
    getenv_path(SLEEF_BUILD_DIR)
    if (NOT ENV_SLEEF_BUILD_DIR)
        set(ENV_SLEEF_BUILD_DIR ${ENV_SLEEF_ROOT}/build)
    endif ()

    # include directory under ${ENV_SLEEF_ROOT}/include
    list(APPEND SLEEF_LIB_SEARCH_PATH
            ${ENV_SLEEF_BUILD_DIR}/${SLEEF_BUILD_PREFIX}_release
            ${ENV_SLEEF_BUILD_DIR}/${SLEEF_BUILD_PREFIX}_debug)
endif ()

# add general search paths
foreach (dir
        IN
        LISTS
        SLEEF_PREFIX_PATH)
    list(APPEND SLEEF_LIB_SEARCH_PATH
            ${dir}/lib
            ${dir}/Lib
            ${dir}/lib/sleef
            ${dir}/Libs)
    list(APPEND SLEEF_INC_SEARCH_PATH
            ${dir}/include
            ${dir}/Include
            ${dir}/include/sleef)
endforeach ()

log_heavy_debug("SLEEF PREFIX : ${SLEEF_PREFIX_PATH}")
log_heavy_debug("SLEEF_INC : ${SLEEF_INC_SEARCH_PATH}")
log_heavy_debug("SLEEF_LIB : ${SLEEF_LIB_SEARCH_PATH}")

set(SLEEF_LIBRARY_NAMES sleef)

find_path(SLEEF_INCLUDE_DIR NAMES sleef.h PATHS ${SLEEF_INC_SEARCH_PATH})

log_heavy_debug("SLEEF_INCLUDE_DIR : ${SLEEF_INCLUDE_DIR}")

find_library(SLEEF_LIBRARY
        NAMES ${SLEEF_LIBRARY_NAMES}
        PATHS ${SLEEF_LIB_SEARCH_PATH})

log_heavy_debug("SLEEF_FIND_LIBRARY : ${SLEEF_LIBRARY}")

findpkg_finish(SLEEF)

# if we haven't found Sleef no point on going any further
if (NOT SLEEF_FOUND)
    return()
endif ()

# =============================================================================
# parse all the version numbers from tbb
if (NOT SLEEF_VERSION)

    set(SLEEF_HEADER "${SLEEF_INCLUDE_DIR}/sleef.h")

    # only read the start of the file
    file(READ ${SLEEF_HEADER} SLEEF_VERSION_CONTENTS
            LIMIT 2048)

    string(REGEX
            REPLACE ".*#define SLEEF_VERSION_MAJOR ([0-9]+).*"
            "\\1"
            SLEEF_VERSION_MAJOR
            "${SLEEF_VERSION_CONTENTS}")

    string(REGEX
            REPLACE ".*#define SLEEF_VERSION_MINOR ([0-9]+).*"
            "\\1"
            SLEEF_VERSION_MINOR
            "${SLEEF_VERSION_CONTENTS}")

    string(REGEX
            REPLACE ".*#define SLEEF_VERSION_PATCHLEVEL ([0-9]+).*"
            "\\1"
            SLEEF_VERSION_PATCH
            "${SLEEF_VERSION_CONTENTS}")

    set(SLEEF_VERSION
            "${SLEEF_VERSION_MAJOR}.${SLEEF_VERSION_MINOR}.${SLEEF_VERSION_PATCH}")
endif ()

set(Sleef_VERSION ${SLEEF_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
        Sleef
        FOUND_VAR SLEEF_FOUND
        REQUIRED_VARS SLEEF_INCLUDE_DIRS SLEEF_LIBRARIES
        VERSION_VAR SLEEF_VERSION)
mark_as_advanced(SLEEF_INCLUDE_DIRS SLEEF_LIBRARIES
        SLEEF_VERSION_MAJOR SLEEF_VERSION_MINOR SLEEF_VERSION_PATCH
        SLEEF_VERSION Sleef_VERSION)
