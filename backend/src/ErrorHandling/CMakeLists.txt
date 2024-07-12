# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY ErrorHandling)

add_elastica_library(${LIBRARY})

elastica_target_sources(
        ${LIBRARY}
        PRIVATE
        AbortWithErrorMessage.cpp
        Breakpoint.cpp
        FloatingPointExceptions.cpp
        NestedExceptions.cpp
)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        AbortWithErrorMessage.hpp
        Abort.hpp
        Assert.hpp
        Breakpoint.hpp
        Error.hpp
        ExpectsAndEnsures.hpp
        Exit.hpp
        FloatingPointExceptions.hpp
        NestedExceptions.hpp
)
