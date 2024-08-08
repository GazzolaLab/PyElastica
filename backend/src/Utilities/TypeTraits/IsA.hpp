#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace tt {

  namespace detail {
    /// \cond HIDDEN_SYMBOLS
    // reduce instantiations using nested structs
    template <template <typename...> class U>
    struct is_a {
      template <typename...>
      struct check : std::false_type {};

      template <typename... Args>
      struct check<U<Args...>> : std::true_type {};
    };
    /// \endcond
  }  // namespace detail

  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type `T` is a template specialization of `U`
  ///
  /// \requires `U` is a class template
  /// \effects If `T` is a template specialization of `U`, then inherits from
  /// std::true_type, otherwise inherits from std::false_type
  ///
  /// \usage
  /// For any type `T` and class template `U`
  /// \code
  /// using result = tt::is_a<U, T>;
  /// \endcode
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If the type `T` is a template specialization of the type `U`, then
  /// \code
  /// typename result::type = std::true_type;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = std::false_type;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp is_a_example
  /// \see is_std_array
  /// \tparam T type to check
  /// \tparam U the type that T might be a template specialization of
  template <template <typename...> class U, typename... Args>
  struct is_a : detail::is_a<U>::template check<Args...> {};

  /// \see is_a
  template <template <typename...> class U, typename... Args>
  constexpr bool is_a_v = detail::is_a<U>::template check<Args...>::value;

  /// \see is_a
  template <template <typename...> class U, typename... Args>
  using is_a_t = typename detail::is_a<U>::template check<Args...>::type;
  // @}

}  // namespace tt
