# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PySystemsCommon")

add_elastica_library(${LIBRARY} INTERFACE)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}
        BindVariableLocator.hpp
        BindMemberAccess.hpp
        BoundChecks.hpp
        VariableFilter.hpp
        VariableLocator.hpp
)

elastica_target_sources(
        ${LIBRARY}
        INTERFACE
        BindVariableLocator.hpp
        BindMemberAccess.hpp
        VariableFilter.hpp
        VariableLocator.hpp
)

# this library is needed only if bindings are built
elastica_python_link_libraries(
        ${LIBRARY}
        INTERFACE
        Utilities
        ConvertCase #
        pybind11::module
)
