// Original Copyright (c) 2014 Adam Wulkiewicz, Lodz, Poland.
//
// Use, modification and distribution is subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//==============================================================================
/*!
//  \file blaze/util/MaybeUnused.h
//  \brief Header file for the ignore_unused function template
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or
modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source
and binary
//  forms, with or without modification, are permitted provided that the
following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice,
this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
this list
//     of conditions and the following disclaimer in the documentation and/or
other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its
contributors
//     may be used to endorse or promote products derived from this software
without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH
//  DAMAGE.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/ForceInline.hpp"

namespace elastica {

  //============================================================================
  //
  //  ignore_unused FUNCTION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Suppression of unused parameter warnings.
  // \ingroup utils
  //
  // \details
  // The ignore_unused function provides the functionality to suppress warnings
  // about any number of unused parameters. Usually this problem occurs in case
  // a parameter is given a name but is not used within the function:
     \code
       void f( int x )
       {}  // x is not used within f!!
     \endcode
  // This may result in a compiler warning. A possible solution is to keep the
  // parameter unnamed:
     \code
       void f( int )
       {}  // No warning about unused parameter is issued
     \endcode
  // However, there are situations where is approach is not possible, as for
  // instance in case the variable must be documented via Doxygen. For these
  // cases, the ignore_unused class can be used to suppress the warnings:
     \code
       void f( int x ){
          ignore_unused( x );  // Suppresses the unused parameter warnings
       }
     \endcode
  //
  // \return void
  */
  template <typename... Args>
  ELASTICA_ALWAYS_INLINE constexpr void ignore_unused(const Args&...) noexcept {
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Suppression of unused parameter warnings.
  // \ingroup utils
  //
  // This version takes in no parameters an is intended for ignored aliases. To
  // suppress unused typedef/aliases warnings, one can simply
     \code
     void f(){
        using type = int;
        ignore_unused< type >();  // Suppresses the unused alias warnings
     }
     \endcode
  //
  // \return void
  */
  template <typename... T>
  ELASTICA_ALWAYS_INLINE constexpr void ignore_unused() noexcept {}
  //****************************************************************************

  //****************************************************************************
  /*!\brief Mark a unused type to be ignored
  // \ingroup utils
  //
  // \details
  // Helper macro for marking types and aliases as unused, used like:
  //
  // \code
      using range_type IGNORE_UNUSED = decltype(range(2,4));
      // range_type is not used below
     \endcode
  //
  // This is quite useful in compile-time metaprogramming when you want to check
  // whether a list of types meet a particular requirement. As opposed to
  // ignore_unused(), you can use this macro even in a non-main() context.
  //
  // \note
  // Newer gcc, clang need special treatment to suppress unused typedef
  // warnings.
  */
  //****************************************************************************
#if defined(__clang__)
#if defined(__apple_build_version__)
#if (__clang_major__ >= 7)
#define IGNORE_UNUSED __attribute__((__unused__))
#endif  // (__clang_major__ >= 7)
#elif ((__clang_major__ == 3) && (__clang_minor__ >= 6)) || \
    (__clang_major__ > 3)
#define IGNORE_UNUSED __attribute__((__unused__))
#endif  // ((__clang_major__ == 3) && (__clang_minor__ >= 6))
        //   || (__clang_major__ > 3)
#elif defined(__GNUC__)
#if ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 8)) || (__GNUC__ > 4)
#define IGNORE_UNUSED __attribute__((__unused__))
#endif  // ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 8)) || (__GNUC__ > 4)
#endif  // defined(__GNUC__)
#if !defined(IGNORE_UNUSED)
#define IGNORE_UNUSED
#endif  // !defined(IGNORE_UNUSED_TYPEDEF)
  //****************************************************************************

}  // namespace elastica
