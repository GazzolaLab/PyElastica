set(LIBRARY Utilities)

elastica_target_sources(
        ${LIBRARY}
        PRIVATE
)

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        Rot3.hpp
        Vec3.hpp
        Types.hpp
)

add_subdirectory(Python)

target_link_libraries(
        ${LIBRARY}
        PRIVATE
        Blaze
)
