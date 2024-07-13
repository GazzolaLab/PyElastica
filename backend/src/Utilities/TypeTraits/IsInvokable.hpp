#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>
#include <utility>

#include "Cpp17.hpp"

namespace cpp17 {

  namespace is_invocable_detail {
    //**************************************************************************
    /* taken from Boost::Beast
     *
     * Copyright (c) 2016-2019 Vinnie Falco (vinnie dot falco at gmail dot
     * com)
     *
     * Distributed under the Boost Software License, Version 1.0. (See
     * accompanying file LICENSE_1_0.txt or copy at
     * http: *www.boost.org/LICENSE_1_0.txt)
     *
     * Official repository: https: *github.com/boostorg/beast
     *
     */
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <class R, class C, class... A>
    auto is_invocable_test(C&& c, int, A&&... a)
        -> decltype(std::is_convertible<decltype(c(std::forward<A>(a)...)),
                                        R>::value ||
                        std::is_same<R, void>::value,
                    std::true_type());

    template <class R, class C, class... A>
    std::false_type is_invocable_test(C&& c, long, A&&... a);
    /*! \endcond */
    //**************************************************************************

  }  // namespace is_invocable_detail

  //****************************************************************************
  /*!\brief Determines whether F can be invoked with the arguments A... to
   * yield a result that is convertible to R.
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `F` has the call operator, operator()
   * defined with arguments `A...` to result in a type `R` that is convertible
   * to `R`, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `F`, `R` and types `A...`
   * \code
   * using result = cpp17::tt::is_invocable_r<R, F, A...>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * For any type `F` defines an operator() with arguments of types `A...` to
   * return a type convertible to `R`
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_invocable_r_example
   *
   * \tparam R Return type to check
   * \tparam F Callable to check
   * \tparam A... Type of arguments to be passed onto callable
   *
   * \see is_invocable
   */
  template <class R, class F, class... A>
  struct is_invocable_r : decltype(is_invocable_detail::is_invocable_test<R>(
                              std::declval<F>(), 1, std::declval<A>()...)) {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Variable template for eager check of `is_invocable_r`
   * \ingroup type_traits_group
   *
   * \details
   * The is_invocable_r_v is a helper variable template to obtain the nested
   * value `value` of `is_invocable_r`, used as follows.
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_invocable_r_v_example
   *
   * \see is_invocable_r
   */
  template <class R, class Fn, class... ArgTypes>
  constexpr bool is_invocable_r_v = is_invocable_r<R, Fn, ArgTypes...>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Determines whether F can be invoked with the arguments A...
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `F` has the call operator, operator()
   * defined with arguments `A...`, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `F` and types `A...`
   * \code
   * using result = cpp17::tt::is_invocable<F, A...>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * For any type `F` defines an operator() with arguments of types `A...`
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_invocable_example
   *
   * \tparam F Callable to check
   * \tparam A... Type of arguments to be passed onto callable
   *
   * \see is_invocable_r
   */
  template <typename F, typename... A>
  struct is_invocable : is_invocable_r<void, F, A...> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Variable template for eager check of `is_invocable`
   * \ingroup type_traits_group
   *
   * \details
   * The is_invocable_v is a helper variable template to obtain the nested
   * value `value` of `is_invocable`, used as follows.
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_invocable_v_example
   *
   * \see is_invocable
   */
  template <class Fn, class... ArgTypes>
  constexpr bool is_invocable_v = is_invocable<Fn, ArgTypes...>::value;
  //****************************************************************************

  // TODO should be shifted to is_callable
  //****************************************************************************
  /*!\brief Determines whether F can be called with the arguments A... and
   * that such call is known not to throw any exceptions.
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `F` has the call operator, operator()
   * defined with arguments `A...` with `noexcept` specification, otherwise
   * inherits from std::false_type.
   *
   * \usage
   * For any type `F` and types `A...`
   * \code
   * using result = cpp17::tt::is_nothrow_callable<F, A...>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * For any type `F` defines a `noexcept` operator() with arguments of types
   * `A...`
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_nothrow_callable
   *
   * \tparam F Callable to check
   * \tparam A... Type of arguments to be passed onto callable
   *
   * \see is_callable
   */
  template <typename F, typename... Args>
  struct is_nothrow_callable
      : bool_constant<noexcept(std::declval<F>()(std::declval<Args>()...))> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Determines whether F can be invoked with the arguments A... to
   * yield a result that is convertible to R, and that such call is known not
   * to throw any exceptions.
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `F` has a `noexcept` call operator,
   * operator() defined with arguments `A...` to result in a type `R` that is
   * convertible to `R`, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `F`, `R` and types `A...`
   * \code
   * using result = cpp17::tt::is_nothrow_invocable_r<R, F, A...>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * For any type `F` defines a `noexcept` operator() with arguments of types
   * `A...` to return a type convertible to `R`
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_nothrow_invocable_r_example
   *
   * \tparam R Return type to check
   * \tparam F Callable to check
   * \tparam A... Type of arguments to be passed onto callable
   *
   * \see is_invocable_r
   */
  template <class R, class F, class... A>
  struct is_nothrow_invocable_r
      : conjunction<is_invocable_r<R, F, A...>, is_nothrow_callable<F, A...>> {
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Variable template for eager check of `is_nothrow_invocable_r`
   * \ingroup type_traits_group
   *
   * \details
   * The is_nothrow_invocable_r_v is a helper variable template to obtain the
   * nested value `value` of `is_nothrow_invocable_r`, used as follows.
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_nothrow_invocable_r_v_example
   *
   * \see is_nothrow_invocable_r
   */
  template <class R, class Fn, class... ArgTypes>
  constexpr bool is_nothrow_invocable_r_v =
      is_nothrow_invocable_r<R, Fn, ArgTypes...>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Determines whether F can be invoked with the arguments A... and
   * that such call is known not to throw any exceptions.
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `F` has a `noexcept` call operator,
   * operator() defined with arguments `A...`, otherwise inherits from
   * std::false_type.
   *
   * \usage
   * For any type `F` and types `A...`
   * \code
   * using result = cpp17::tt::is_nothrow_invocable<F, A...>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * For any type `F` defines a `noexcept` operator() with arguments of
   * types `A...`
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_nothrow_invocable_example
   *
   * \tparam F Callable to check
   * \tparam A... Type of arguments to be passed onto callable
   *
   * \see is_nothrow_invocable_r
   */
  template <class F, class... A>
  struct is_nothrow_invocable : is_nothrow_invocable_r<void, F, A...> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Variable template for eager check of `is_nothrow_invocable`
   * \ingroup type_traits_group
   *
   * \details
   * The is_nothrow_invocable_v is a helper variable template to obtain the
   * nested value `value` of `is_nothrow_invocable`, used as follows.
   *
   * \example
   * \snippet Test_IsInvokable.cpp is_nothrow_invocable_v_example
   *
   * \see is_nothrow_invocable
   */
  template <class Fn, class... ArgTypes>
  constexpr bool is_nothrow_invocable_v =
      is_nothrow_invocable<Fn, ArgTypes...>::value;
  //****************************************************************************

}  // namespace cpp17
