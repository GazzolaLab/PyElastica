//==============================================================================
/*!
//  \file
//  \brief Macros for unrolling without templates
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
*/
//==============================================================================

// https://stackoverflow.com/a/63769749

/// Helper macros for stringification
#define TO_STRING_HELPER(X) #X
#define TO_STRING(X) TO_STRING_HELPER(X)

// Define loop unrolling depending on the compiler
#if defined(__ICC) || defined(__ICL)
#define UNROLL_LOOP(n) _Pragma(TO_STRING(unroll(n)))
#elif defined(__clang__)
#define UNROLL_LOOP(n) _Pragma(TO_STRING(unroll(n)))
#elif defined(__GNUC__) && !defined(__clang__)
#define UNROLL_LOOP(n) _Pragma(TO_STRING(GCC unroll(16)))
#elif defined(_MSC_BUILD)
#pragma message( \
    "Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
#define UNROLL_LOOP(n)
#else
#warning "Unknown compiler: Loop unrolling not supported!"
#define UNROLL_LOOP(n)
#endif
