//==============================================================================
/*!
//  \file
//  \brief Create a static constant that avoids ODR violations
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  This was taken from the Range v3 library with the following copyrights
**  Copyright Eric Niebler 2013-present
**
**  Use, modification and distribution is subject to the
**  Boost Software License, Version 1.0. (See accompanying
**  file LICENSE_1_0.txt or copy at
**  http://www.boost.org/LICENSE_1_0.txt)
**
**  Project home: https://github.com/ericniebler/range-v3
**
*/
//==============================================================================

#pragma once

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Holds a static compile-time constant
  // \ingroup utils
  //
  // static_const is a helper to create and hold a static compile-time constant
  //
  // \usage
     \code
       // create a default initialized integer
       constexpr auto zero = elastica::static_const<int>::value;
     \endcode
  //
  // \example
  // \snippet Test_StaticConst.cpp static_const_example
  //
  // \tparam T Type of static constant to create
  */
  template <typename T>
  struct static_const {
    static constexpr T value{};
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Variable template storage for a static compile-time constant
  // \ingroup utils
  //
  // \tparam T Type of static constant to create
  //
  // \see static_const
  */
  template <typename T>
  constexpr T static_const<T>::value;
  //****************************************************************************

}  // namespace elastica
