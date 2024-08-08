#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "Utilities/TypeTraits/Void.hpp"

namespace tt {

  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type T has a begin() and end() function
  ///
  /// \details
  /// Given a type `T` inherits from std::true_type if `T` has member functions
  /// `begin()` and `end()`, otherwise inherits from std::false_type
  ///
  /// \usage
  /// For any type `T`
  /// \code
  /// using result = tt::is_iterable<T>;
  /// \endcode
  ///
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If `T` has member function `begin()` and `end()` then
  /// \code
  /// typename result::type = std::true_type;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = std::false_type;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp is_iterable_example
  /// \see has_size
  /// \tparam T the type to check
  namespace detail {
    template <typename T, typename = cpp17::void_t<>>
    struct is_iterable : std::false_type {};
    /// \cond HIDDEN_SYMBOLS
    template <typename T>
    struct is_iterable<T, cpp17::void_t<decltype(std::declval<T>().begin(),
                                                 std::declval<T>().end())>>
        : std::true_type {};
    /// \endcond
    /// \see is_iterable
  }  // namespace detail
  template <typename T>
  constexpr bool is_iterable_v = detail::is_iterable<T>::value;

  /// \see is_iterable
  template <typename T>
  using is_iterable_t = typename detail::is_iterable<T>::type;
  // @}

}  // namespace tt
