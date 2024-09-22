#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstdint>
#include <utility>
#include <tuple>

#include "Utilities/TypeTraits/InvokeResult.hpp"
#include "Utilities/TypeTraits/IsInvokable.hpp"

namespace cpp17 {

  namespace invoke_detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief implementation of invoke
    // \ingroup UtilitiesGroup
    //
    // Developer note :
    // This is not strictly C++17 compliant since we haven't implemented
    // specializations for member functions etc.
    */
    template <class F, class... Args>
    constexpr decltype(auto) invoke(F&& f, Args&&... args) {
      return std::forward<F>(f)(std::forward<Args>(args)...);
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace invoke_detail

  //****************************************************************************
  /*!\brief Invoke a function on parameters
  // \ingroup UtilitiesGroup
  //
  // \details
  // Invoke the Callable object `f` with the parameters `args`, like so
  // `f(t1, t2, ..., tN)`.
  //
  // \param f : Callable object to be invoked
  // \param args : Arguments to pass to f
  //
  // \example
  // \snippet Test_Invoke.cpp invoke_example
  //
  // invoke() can also be used in a `constexpr` context
  // \snippet Test_Invoke.cpp invoke_constexpr
  //
  // Developer note:
  // - This is not strictly C++17 compliant since we haven't implemented
  // specializations for member functions etc.
  //
  // \see apply()
  */
  template <class F, class... Args>
  constexpr tt::invoke_result_t<F, Args...> invoke(
      F&& f, Args&&... args) noexcept(is_nothrow_invocable_v<F, Args...>) {
    return invoke_detail::invoke(std::forward<F>(f),
                                 std::forward<Args>(args)...);
  }
  //****************************************************************************

  namespace apply_detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Implementation of apply
    // \ingroup UtilitiesGroup
    */
    template <typename F, typename Tuple, std::size_t... Is>
    constexpr decltype(auto)
    apply_impl(F&& f, Tuple&& tuple, std::index_sequence<Is...>) noexcept(
        noexcept(::cpp17::invoke(
            std::forward<F>(f), std::get<Is>(std::forward<Tuple>(tuple))...))) {
      return ::cpp17::invoke(std::forward<F>(f),
                             std::get<Is>(std::forward<Tuple>(tuple))...);
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace apply_detail

  //****************************************************************************
  /*!\brief Invokes a function on a tuple
  // \ingroup UtilitiesGroup
  //
  // \details
  // Invoke the Callable object `f` with a tuple of arguments.
  //
  // \param f : Callable object to be invoked
  // \param tuple : tuple type with arguments to be passed to f
  //
  // \note
  // The tuple need not be std::tuple, and instead may be anything that supports
  // std::get and std::tuple_size; in particular, std::array and std::pair may
  // be used.
  //
  // \example
  // \snippet Test_Invoke.cpp apply_example
  //
  // apply() can also be used in a `constexpr` context
  // \snippet Test_Invoke.cpp apply_constexpr
  //
  // Developer note :
  // - This is not strictly C++17 compliant because of our implementation of
  // `invoke`
  //
  // \see invoke()
  */
  template <typename F, typename Tuple>
  constexpr decltype(auto) apply(F&& f, Tuple&& tuple) noexcept(
      noexcept(apply_detail::apply_impl(
          std::forward<F>(f), std::forward<Tuple>(tuple),
          std::make_index_sequence<
              std::tuple_size<std::decay_t<Tuple>>::value>()))) {
    return apply_detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(tuple),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>());
  }
  //****************************************************************************

}  // namespace cpp17

namespace index_apply_detail {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Implementation of index apply
  // \ingroup UtilitiesGroup
  */
  template <class F, std::size_t... Is>
  constexpr decltype(auto)
  index_apply_impl(F&& f, std::index_sequence<Is...> /* meta */) noexcept(
      noexcept(f(std::integral_constant<std::size_t, Is>{}...))) {
    return f(std::integral_constant<std::size_t, Is>{}...);
  }
  /*! \endcond */
  //****************************************************************************

}  // namespace index_apply_detail

//******************************************************************************
/*!\brief Invokes a function on compile-time indices
// \ingroup UtilitiesGroup
//
// \details
// Invoke the Callable object `f` with `N` integer parameters `t`, like so
// `f(0, 1, ..., N - 1)`.
//
// \param f : Callable object to be invoked on the indices
//
// \tparam N : The compile-time number of indices
//
// \example
// \snippet Test_Invoke.cpp index_apply_example
//
// index_apply() can also be used in a `constexpr` context, but is less useful
// in C++14 (but more useful in C++17 with constexpr lambdas)
//
// \see apply()
*/
template <std::size_t N, class F>
constexpr decltype(auto) index_apply(F&& f) noexcept(noexcept(
    index_apply_detail::index_apply_impl(std::forward<F>(f),
                                         std::make_index_sequence<N>{}))) {
  return index_apply_detail::index_apply_impl(std::forward<F>(f),
                                              std::make_index_sequence<N>{});
}
//******************************************************************************
