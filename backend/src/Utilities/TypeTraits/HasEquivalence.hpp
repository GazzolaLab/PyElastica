// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace tt {

  namespace detail {
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Check for existence of an equality operator
    // \ingroup type_traits_group
    */
    template <class T>
    using equivalence_t = decltype(std::declval<T>() = std::declval<T>());
    /*! \endcond */
    //**************************************************************************
  }  // namespace detail

  //****************************************************************************
  /* \brief Check if type `T` has operator== defined.
  // \ingroup type_traits_group
  //
  // \details
  // Inherits from std::true_type if the type `T` has operator== defined,
  // otherwise inherits from std::false_type
  //
  // \usage
  // For any type `T`,
  // \code
  // using result = tt::has_equivalence<T>;
  // \endcode
  //
  // \metareturns
  // std::bool_constant
  //
  // \semantics
  // If the type `T` has operator== defined, then
  // \code
  // typename result::type = std::true_type;
  // \endcode
  // otherwise
  // \code
  // typename result::type = std::false_type;
  // \endcode
  //
  // \example
  // \snippet Test_HasEquivalence.cpp has_equivalence_example
  //
  // \tparam T the type we want to know if it has operator==
  //
  // \see has_inequivalence
  */
  template <typename T>
  struct has_equivalence : public is_detected<detail::equivalence_t, T> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for eager check of `has_equivalence`
  // \ingroup type_traits_group
  //
  // \details
  // The has_equivalence_v variable template provides a convenient shortcut to
  // access the nested value `value` of `has_equivalence`, used as follows.
  //
  // \usage
  // Given a type `T`, following two statements are identical:
  // \code
  // constexpr bool value1 = tt::has_equivalence<T>::value;
  // constexpr bool value2 = tt::has_equivalence_v<T>;
  // \endcode
  // as demonstrated through this example
  //
  // \example
  // \snippet Test_HasEquivalence.cpp has_equivalence_v_example
  //
  // \tparam T the type we want to know if it has operator==
  //
  // \see has_equivalence
  */
  template <typename T>
  constexpr bool has_equivalence_v = has_equivalence<T>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for `has_equivalence`.
  // \ingroup type_traits_group
  //
  // \details
  // The has_equivalence_t variable template provides a convenient shortcut to
  // access the nested type `type` of `has_equivalence`, used as follows.
  //
  // \usage
  // Given a type `T`, following two statements are identical:
  // \code
  // using type1 = typename tt::has_equivalence<T>::type;
  // using type2 = tt::has_equivalence_t<T>;
  // \endcode
  // as demonstrated through this example
  //
  // \example
  // \snippet Test_HasEquivalence.cpp has_equivalence_t_example
  /
  // \tparam T the type we want to know if it has operator==
  //
  // \see has_equivalence
  */
  template <typename T>
  using has_equivalence_t = typename has_equivalence<T>::type;
  //****************************************************************************

}  // namespace tt
