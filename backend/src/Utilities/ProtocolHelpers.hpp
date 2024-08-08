//==============================================================================
/*!
//  \file
//  \brief Helpers for enforcing compile-time protocols in \elastica
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  Reused with thanks from SpECTRE : https://spectre-code.org/
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace tt {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Indicate a class conforms to the `Protocol`.
   * \ingroup protocols
   *
   * (Publicly) inherit classes from this class to indicate they conform to the
   * `Protocol`.
   *
   * \see Documentation on \ref protocols
   */
  template <typename Protocol>
  struct ConformsTo {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Checks if the `ConformingType` conforms to the `Protocol`.
   * \ingroup protocols
   *
   * \details
   * This metafunction is SFINAE-friendly. See `tt::assert_conforms_to` for a
   * metafunction that is not SFINAE-friendly but that triggers static asserts
   * with diagnostic messages to understand why the `ConformingType` does not
   * conform to the `Protocol`.
   *
   * This metafunction only checks if the class derives off the protocol to
   * reduce compile time. Protocol conformance is tested rigorously in the unit
   * tests instead.
   *
   * \note std::is_convertible is used in the following type aliases as it
   * will not match private base classes (unlike std::is_base_of)
   *
   * \see Documentation on \ref protocols
   * \see tt::assert_conforms_to
   */
  template <typename ConformingType, typename Protocol>
  using conforms_to =
      typename std::is_convertible<ConformingType*, ConformsTo<Protocol>*>;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for tt::conforms_to
   * \ingroup protocols
   *
   * \details
   * The conforms_to_v variable template provides a convenient shortcut to
   * access the nested `value` of the conforms_to class template. For
   * instance, given the types `T`, `Protocol` the following two statements are
   * identical:
   * \example
   * \code
   * constexpr bool value1 = tt::conforms_to<T, Protocol>::value;
   * constexpr bool value2 = tt::conforms_to_v<T, Protocol>;
   * \endcode
   *
   * \see tt::conforms_to
   */
  template <typename ConformingType, typename Protocol>
  static constexpr bool conforms_to_v =
      std::is_convertible<ConformingType*, ConformsTo<Protocol>*>::value;
  //****************************************************************************

  namespace detail {

    template <typename ConformingType, typename Protocol>
    struct AssertConformsToImpl : std::true_type {
      static_assert(
          tt::conforms_to_v<ConformingType, Protocol>,
          "The type does not indicate it conforms to the protocol. The type is "
          "listed as the first template parameter to `assert_conforms_to` "
          "and the protocol is listed as the second template parameter. "
          "Have you forgotten to (publicly) inherit the type from "
          "tt::ConformsTo<Protocol>?");
      static_assert(
          not cpp17::is_same_v<
              decltype(typename Protocol::template test<ConformingType>{}),
              void>,
          "Protocol failure!");
    };

  }  // namespace detail

  //****************************************************************************
  /*!\brief Assert that the `ConformingType` conforms to the `Protocol`.
   * \ingroup protocols
   *
   * Similar to tt::conforms_to, but not SFINAE-friendly. Instead, triggers
   * static asserts with diagnostic messages to understand why the
   * `ConformingType` fails to conform to the `Protocol`.
   *
   * \see Documentation on \ref protocols
   * \see tt::conforms_to
   */
  template <typename ConformingType, typename Protocol>
  static constexpr bool assert_conforms_to =
      detail::AssertConformsToImpl<ConformingType, Protocol>::value;
  //****************************************************************************

}  // namespace tt
