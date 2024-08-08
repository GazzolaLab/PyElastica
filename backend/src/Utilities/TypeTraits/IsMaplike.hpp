#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"
#include "Utilities/TypeTraits/Void.hpp"

namespace tt {

  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type `T` is like a std::map or std::unordered_map
  ///
  /// \details
  /// Inherits from std::true_type if the type `T` has a type alias `key_type`,
  /// type alias `mapped_type`, and `operator[](const typename T::key_type&)`
  /// defined, otherwise inherits from std::false_type
  ///
  /// \usage
  /// For any type `T`,
  /// \code
  /// using result = tt::is_maplike<T>;
  /// \endcode
  ///
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If the type `T` has a type alias `key_type`,
  /// type alias `mapped_type`, and `operator[](const typename T::key_type&)`
  /// defined, then
  /// \code
  /// typename result::type = std::true_type;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = std::false_type;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp is_maplike_example
  /// \see std::map std::unordered_map is_a
  /// \tparam T the type to check
  namespace detail {
    template <typename T, typename = cpp17::void_t<>>
    struct is_maplike : std::false_type {};
    /// \cond HIDDEN_SYMBOLS
    template <typename T>
    struct is_maplike<
        T, cpp17::void_t<typename T::key_type, typename T::mapped_type,
                         decltype(std::declval<T&>()[std::declval<
                             const typename T::key_type&>()]),
                         Requires<tt::is_iterable_v<T>>>> : std::true_type {};
    /// \endcond
  }  // namespace detail

  /// \see is_maplike
  template <typename T>
  constexpr bool is_maplike_v = detail::is_maplike<T>::value;

  /// \see is_maplike
  template <typename T>
  using is_maplike_t = typename detail::is_maplike<T>::type;
  // @}

}  // namespace tt
