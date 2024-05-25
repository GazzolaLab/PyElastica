# Distributed under the MIT License.
# See LICENSE.txt for details.

# Create an C++ standard library target for tracking dependencies
# through includes throughout Elastica.
if (NOT TARGET Stl)
    add_library(Stl INTERFACE IMPORTED)

    add_interface_lib_headers(
            TARGET Stl
            HEADERS
            algorithm
            any
            array
            atomic
            barrier
            bit
            bitset
            cassert
            cctype
            cerrno
            cfenv
            cfloat
            charconv
            chrono
            cinttypes
            climits
            clocale
            cmath
            codecvt
            compare
            complex
            concepts
            condition_variable
            coroutine
            csetjmp
            csignal
            cstdarg
            cstddef
            cstdint
            cstdio
            cstdlib
            cstring
            ctime
            cuchar
            cwchar
            cwctype
            deque
            exception
            execution
            format
            forward_list
            fstream
            functional
            future
            initializer_list
            iomanip
            ios
            iosfwd
            iostream
            istream
            iterator
            latch
            limits
            list
            locale
            map
            memory
            memory_resource
            mutex
            new
            numbers
            numeric
            optional
            ostream
            queue
            random
            ranges
            ratio
            regex
            scoped_allocator
            semaphore
            set
            shared_mutex
            source_location
            span
            sstream
            stack
            stdexcept
            stop_token
            streambuf
            string
            string_view
            strstream
            syncstream
            system_error
            thread
            tuple
            type_traits
            typeindex
            typeinfo
            unordered_map
            unordered_set
            utility
            valarray
            variant
            vector
            version

            # UNIX/Linux specific headers
            dirent.h
            libgen.h
            sys/stat.h
            sys/types.h
            unistd.h
            xmmintrin.h
    )

    set_property(
            GLOBAL APPEND PROPERTY ELASTICA_THIRD_PARTY_LIBS
            Stl
    )
endif (NOT TARGET Stl)
