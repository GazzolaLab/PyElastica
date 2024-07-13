#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>
// void_t
#include "Utilities/TypeTraits/Void.hpp"

namespace cpp17 {

  //****************************************************************************
  /*!\brief Mark a return type as being "void"
  // \ingroup utils
  //
  // \details
  // In C++17 void is a regular type under certain circumstances, so this can
  // be replaced by `void` then. The proposal is available
  // [he](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0146r1.html)
  */
  struct void_type {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief A compile-time boolean
  // \ingroup type_traits_group
  //
  // \usage
  // For any bool `B`
     \code
     using result = cpp17::bool_constant<B>;
     \endcode
  //
  // \see std::bool_constant std::integral_constant std::true_type
  // std::false_type
  */
  template <bool B>
  using bool_constant = std::integral_constant<bool, B>;
  //****************************************************************************

  //****************************************************************************
  /*!\brief A logical AND on the template parameters
  // \ingroup type_traits_group
  //
  // \details
  // Given a list of cpp17::bool_constant template parameters computes their
  // logical AND. If the result is `true` then derive off of std::true_type,
  // otherwise derive from std::false_type. See the documentation for
  // std::conjunction for more details.
  //
  // \usage
  // For any set of types `Ti` that are cpp17::bool_constant like
     \code
     using result = cpp17::conjunction<T0, T1, T2>;
     \endcode
  // \pre For all types `Ti`, `Ti::value` is a `bool`
  //
  // \metareturns
  // cpp17::bool_constant
  //
  // \semantics
  // If `T::value != false` for all `Ti`, then
     \code
     using result = cpp17::bool_constant<true>;
     \endcode
  // otherwise
     \code
     using result = cpp17::bool_constant<false>;
     \endcode
  //
  // \example
  // \snippet TypeTraits/Test_Cpp17.cpp conjunction_example
  //
  // \tparam B... A set of cpp17::bool_constant
  //
  // \see std::conjunction, disjunction, std::disjunction
  */
  template <class...>
  struct conjunction : std::true_type {};
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <class B1>
  struct conjunction<B1> : B1 {};
  template <class B1, class... Bn>
  struct conjunction<B1, Bn...>
      : std::conditional_t<static_cast<bool>(B1::value), conjunction<Bn...>,
                           B1> {};
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for eager check of `cpp17::conjunction`
  // \ingroup type_traits_group
  //
  // \details
  // The conjunction_v variable template provides a convenient shortcut to
  // access the nested value `value` of `conjunction`, used as follows.
  //
  // \usage
  // Given two `cpp17::bool_constant` `B1` and `B2` the following two statements
  // are identical:
     \code
     constexpr bool value1 = cpp17::conjunction<B1, B2>::value;
     constexpr bool value2 = cpp17::conjunction_v<B1, B2>;
     \endcode
  //
  // \tparam B... A set of cpp17::bool_constant
  //
  // \see conjunction
  */
  template <class... B>
  constexpr bool conjunction_v = conjunction<B...>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief A logical OR on the template parameters
  // \ingroup type_traits_group
  //
  // \details
  // Given a list of cpp17::bool_constant template parameters computes their
  // logical OR. If the result is `true` then derive off of std::true_type,
  // otherwise derive from std::false_type. See the documentation for
  // std::conjunction for more details.
  //
  // \usage
  // For any set of types `Ti` that are cpp17::bool_constant like
     \code
     using result = cpp17::disjunction<T0, T1, T2>;
     \endcode
  // \pre For all types `Ti`, `Ti::value` is a `bool`
  //
  // \metareturns
  // cpp17::bool_constant
  //
  // \semantics
  // If `T::value == true` for any `Ti`, then
     \code
     using result = cpp17::bool_constant<true>;
     \endcode
  // otherwise
     \code
     using result = cpp17::bool_constant<false>;
     \endcode
  //
  // \example
  // \snippet TypeTraits/Test_Cpp17.cpp disjunction_example
  //
  // \tparam B... A set of cpp17::bool_constant
  //
  // \see std::disjunction, conjunction, std::conjunction
  */
  template <class...>
  struct disjunction : std::false_type {};
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <class B1>
  struct disjunction<B1> : B1 {};
  template <class B1, class... Bn>
  struct disjunction<B1, Bn...>
      : std::conditional_t<static_cast<bool>(B1::value), B1,
                           disjunction<Bn...>> {};
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for eager check of `cpp17::disjunction`
  // \ingroup type_traits_group
  //
  // \details
  // The disjunction_v variable template provides a convenient shortcut to
  // access the nested value `value` of `disjunction`, used as follows.
  //
  // \usage
  // Given two cpp17::bool_constant `B1` and `B2` the following two statements
  // are identical:
     \code
     constexpr bool value1 = cpp17::disjunction<B1, B2>::value;
     constexpr bool value2 = cpp17::disjunction_v<B1, B2>;
     \endcode
  //
  // \tparam B... A set of cpp17::bool_constant
  //
  // \see disjunction
  */
  template <class... B>
  constexpr bool disjunction_v = disjunction<B...>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Negate a cpp17::bool_constant
  // \ingroup type_traits_group
  //
  // \details
  // Given a ::bool_constant returns the logical NOT of it.
  //
  // \usage
  // For a ::bool_constant `B`
     \code
     using result = cpp17::negate<B>;
     \endcode
  //
  // \metareturns
  // ::bool_constant
  //
  // \semantics
  // If `B::value == true` then
     \code
     using result = cpp17::bool_constant<false>;
     \endcode
  //
  // \example
  // \snippet TypeTraits/Test_Cpp17.cpp negation_example
  //
  // \tparam B the ::bool_constant to negate
  //
  // \see std::negation
  */
  template <class B>
  struct negation : bool_constant<!B::value> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for eager check of `std::is_same`
  // \ingroup type_traits_group
  //
  // \details
  // The cpp17::is_same_v variable template provides a convenient shortcut to
  // access the nested value `value` of `std::is_same`, used as follows.
  //
  // \usage
  // Given two types `T1` and `T2` the following two statements are identical:
     \code
     constexpr bool value1 = std::is_same<T1, T2>::value;
     constexpr bool value2 = cpp17::is_same_v<T1, T2>;
     \endcode
  //
  // \example
  // \snippet TypeTraits/Test_Cpp17.cpp is_same_v_example
  //
  // \tparam T The first type to check
  // \tparam U The second type to check
  //
  // \see std::is_same
  */
  template <typename T, typename U>
  constexpr bool is_same_v = std::is_same<T, U>::value;
  //****************************************************************************

}  // namespace cpp17
