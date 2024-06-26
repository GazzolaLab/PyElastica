# Distributed under the MIT License. See LICENSE.txt for details.

set(LIBRARY Utilities)

add_elastica_library(${LIBRARY})

#add_subdirectory(Geometry)
#add_subdirectory(Logging)
add_subdirectory(Math)
#add_subdirectory(MemoryMeter)
#add_subdirectory(NamedType)
#add_subdirectory(Singleton)
#add_subdirectory(Timing)
#add_subdirectory(TypeTraits)
#add_subdirectory(ValueTraits)

# elastica_target_sources(
#         ${LIBRARY}
#         PRIVATE
#         Demangle.cpp
#         SystemClock.cpp
#         PrettyType.cpp
#         WrapText.cpp
#         WidthStream.cpp
# )

elastica_target_headers(
        ${LIBRARY}
        INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/elastica
        HEADERS
        # Approx.hpp
        # AsConst.hpp
        # Byte.hpp
        # Clamp.hpp
        # CountingIterator.hpp
        # CRTP.hpp
        DefineTypes.h
        # Demangle.hpp
        # Defer.hpp
        # End.hpp
        # FakeVirtual.hpp
        # ForceInline.hpp
        # Geometry.hpp
        # Get.hpp
        # Generators.hpp
        # GSL.hpp
        # IgnoreUnused.hpp
        # Invoke.hpp
        # KarmarkarKarpPartition.hpp
        # ListRequires.hpp
        # Logging.hpp
        # MakeArray.hpp
        # MakeCopyable.hpp
        # MakeFromTuple.hpp
        # MakeNamedFunctor.hpp
        # MakeSignalingNan.hpp
        # MakeString.hpp
        Math.hpp
        # MemoryMeter.hpp
        # MemoryPool.hpp
        # NamedTemplate.hpp
        # NamedType.hpp
        # NumericAt.hpp
        # NumericRange.hpp
        # NonCopyable.hpp
        # NonCreatable.hpp
        # NoSuchType.hpp
        # Overloader.hpp
        # PrettyType.hpp
        # PrintHelpers.hpp
        # ProtocolHelpers.hpp
        # Rampup.hpp
        # Registration.hpp
        # Requires.hpp
        # Singleton.hpp
        # StaticConst.hpp
        # StaticWarning.hpp
        # StdHelpers.hpp
        # StdVectorHelpers.hpp
        # SystemClock.hpp
        # TaggedTuple.hpp
        # Thresholds.hpp
        # TimeStamps.h
        # Timing.h
        # TMPL.hpp
        # TMPLDebugging.hpp
        # TypeTraits.hpp
        # Unroll.hpp
        # WidthStream.hpp
        # WrapText.hpp
)

target_link_libraries(
        ${LIBRARY}
        PUBLIC
        Blaze
        Brigand
        ErrorHandling
)

#set(LIBRARY_SOURCES Utilities.cpp )
#
#add_elastica_library(${LIBRARY} ${LIBRARY_SOURCES})
#
#target_link_libraries(${LIBRARY} INTERFACE ErrorHandling)
#

# add_subdirectory(AppSupport)
# add_subdirectory(ConvertCase)
