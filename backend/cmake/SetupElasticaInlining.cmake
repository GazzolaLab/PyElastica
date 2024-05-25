# Distributed under the MIT License.
# See LICENSE.txt for details.
# Obtained with thanks from https://github.com/sxs-collaboration/spectre

option(ELASTICA_USE_ALWAYS_INLINE "Force elastica inlining." ON)

set(_ELASTICA_USE_ALWAYS_INLINE 0)

if (ELASTICA_USE_ALWAYS_INLINE)
    set(_ELASTICA_USE_ALWAYS_INLINE 1)
endif ()

set_property(
        TARGET ElasticaFlags
        APPEND PROPERTY
        INTERFACE_COMPILE_DEFINITIONS
        $<$<COMPILE_LANGUAGE:CXX>:ELASTICA_USE_ALWAYS_INLINE=${_ELASTICA_USE_ALWAYS_INLINE}>
)
