//==============================================================================
/*!
//  \file
//  \brief A global functor for size of an iterable
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  The key idea was taken from the Range v3 library with the following
//  copyrights
**
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

//******************************************************************************
// Includes
//******************************************************************************

#include <cassert>
#include <utility>

#include "Utilities/StaticConst.hpp"

namespace cpp17 {

  //****************************************************************************
  /*!\brief Returns the size of a type
   * \ingroup utils
   *
   * \details
   * Drop-in replacement for std::size in C++17. Returns the size of the given
   * range.
   * See https://en.cppreference.com/w/cpp/iterator/size for more details.
   *
   * \example
   * \snippet Test_Size.cpp size_eg
   *
   * \param container a container or view with a size() member function
   * \return The size of container
   */
  template <class Container>
  constexpr auto size(Container const& container) noexcept(
      noexcept(container.size())) -> decltype(container.size()) {
    return container.size();
  }
  //****************************************************************************

}  // namespace cpp17

// size proxy
namespace elastica {

  namespace detail {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================
    using cpp17::size;

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Size type
     * \ingroup utils
     *
     * \details
     * Type of the global size functor for any iterable
     */
    struct impl_size_fn {
      //************************************************************************
      /*!\brief Get the size of any container
       *
       * \param container Any container
       */
      template <class C>
      constexpr auto operator()(C const& container) const
          noexcept(noexcept(size(container))) -> decltype(size(container)) {
        return size(container);
      }
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  // elastica::size is a global function object!
  namespace {
    //**************************************************************************
    /*!\brief The size free function
     * \ingroup utils
     *
     * Returns the size of any range, including STL types
     *
     * \example
     * Works for STL types
     * \snippet Test_Size.cpp size_example
     *
     * And also for user defined types, accessed via the size() free function
     * in its own namespace.
     *
     * \snippet Test_Size.cpp custom_size_example
     */
    constexpr auto const& size = static_const<detail::impl_size_fn>::value;
    //**************************************************************************
  }  // namespace

}  // namespace elastica
