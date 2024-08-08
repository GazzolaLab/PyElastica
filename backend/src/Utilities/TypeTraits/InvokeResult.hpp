#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>
#include <utility>

//
// Copied from https://en.cppreference.com/w/cpp/types/result_of
//

namespace cpp17 {

  namespace tt {

    namespace detail {

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Mock declaration of invoke for free functions
      // \ingroup UtilitiesGroup
      */
      template <class T>
      struct invoke_impl {
        template <class F, class... Args>
        static auto call(F&& f, Args&&... args)
            -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
      };
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Forward declaration of invoke
      // \ingroup UtilitiesGroup
      //
      // Developer note :
      // This is not strictly C++17 compliant since we haven't implemented
      // specializations for member functions etc.
      */
      template <class F, class... Args, class Fd = typename std::decay<F>::type>
      auto INVOKE(F&& f, Args&&... args)
          -> decltype(invoke_impl<Fd>::call(std::forward<F>(f),
                                            std::forward<Args>(args)...));
      /*! \endcond */
      //************************************************************************

    }  // namespace detail

    namespace detail {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Get result upon invocation for a non-invocable
      // \ingroup UtilitiesGroup
      //
      // \details
      // Doesn't have the nested member `type` when non-invocable
      */
      template <typename AlwaysVoid, typename, typename...>
      struct invoke_result {};
      //************************************************************************

      //************************************************************************
      /*!\brief Get result upon invocation for a invocable
      // \ingroup UtilitiesGroup
      */
      template <typename F, typename... Args>
      struct invoke_result<decltype(void(detail::INVOKE(
                               std::declval<F>(), std::declval<Args>()...))),
                           F, Args...> {
        using type = decltype(detail::INVOKE(std::declval<F>(),
                                             std::declval<Args>()...));
      };
      /*! \endcond */
      //************************************************************************
    }  // namespace detail

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Get result of invocation for type `F` with `Args`
    // \ingroup type_traits_group
    //
    // \details
    // The invoke_result type trait has a nested member `type` when
    //
    // \usage
    // For any type `F` callable with arguments of types `Args...`
       \code
       using result = cpp17::tt::invoke_result<T>;
       \endcode
    // \metareturns
    // the type `T = F(Args...)`
    //
    // \example
    // \snippet Test_InvokeResult.cpp invoke_result_example
    //
    // \tparam F Callable to check
    // \tparam Args... Type of arguments to be passed onto callable
    */
    template <class F, class... Args>
    struct invoke_result : detail::invoke_result<void, F, Args...> {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Eager result of invocation for type `F` with `Args`
    // \ingroup type_traits_group
    //
    // \details
    // The invoke_result_t type trait is a helper template to obtain the nested
    // type `type` of `invoke_result`, used as follows.
    //
    // \example
    // \snippet Test_InvokeResult.cpp invoke_result_t_example
    //
    // \see invoke_result
    */
    template <class F, class... Args>
    using invoke_result_t = typename invoke_result<F, Args...>::type;
    //**************************************************************************

  }  // namespace tt

}  // namespace cpp17
