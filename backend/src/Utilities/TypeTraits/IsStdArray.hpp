// Reused from SpECTRE : https://spectre-code.org/
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <type_traits>

namespace tt {
  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type T is a std::array
  ///
  /// \details
  /// Given a type `T` derives from std::true_type if `T` is a std::array and
  /// from std::false_type if `T` is not a std::array.
  ///
  /// \usage
  /// For any type `T`
  /// \code
  /// using result = tt::is_std_array<T>;
  /// \endcode
  ///
  /// \metareturns
  /// std::bool_constant
  ///
  /// \semantics
  /// If `T` is a std::array then
  /// \code
  /// typename result::type = cpp17::bool_constant<true>;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = cpp17::bool_constant<false>;
  /// \endcode
  ///
  /// \example
  /// \snippet Test_IsStdArray.cpp is_std_array_example
  ///
  /// \see is_a is_std_array_of_size
  ///
  /// \tparam T the type to check
  template <typename T>
  struct is_std_array : std::false_type {};

  /// \cond HIDDEN_SYMBOLS
  template <typename T, std::size_t N>
  struct is_std_array<std::array<T, N>> : std::true_type {};
  /// \endcond

  /// \see is_std_array
  template <typename T>
  constexpr bool is_std_array_v = is_std_array<T>::value;

  /// \see is_std_array
  template <typename T>
  using is_std_array_t = typename is_std_array<T>::type;
  // @}
}  // namespace tt
