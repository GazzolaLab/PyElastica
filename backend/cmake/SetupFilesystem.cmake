log_info("Finding Filesystem")

# Older versions of GCC only supports experimental in some laptops
# So including it here

#if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
#    find_package(Filesystem REQUIRED COMPONENTS Experimental)
##    set(CXX_FILESYSTEM_LIBRARIES "stdc++fs")
##    target_link_libraries(elastica PUBLIC ${CXX_FILESYSTEM_LIBRARIES})
#elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
#    find_package(Filesystem REQUIRED COMPONENTS Final)
##    set(CXX_FILESYSTEM_LIBRARIES "")
##    set(CXX_FILESYSTEM_LIBRARIES "c++experimental")
#endif ()

# Hacky, ok for now
add_library(std::filesystem INTERFACE IMPORTED)
target_compile_features(std::filesystem INTERFACE cxx_std_14)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    target_link_libraries(std::filesystem INTERFACE -lstdc++fs)
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    #    target_link_libraries(std::filesystem INTERFACE -lc++fs)
endif ()

#find_package(Filesystem REQUIRED COMPONENTS Final Experimental)

log_debug("FILESYSTEM_TYPE: ${CXX_FILESYSTEM_HEADER}")
