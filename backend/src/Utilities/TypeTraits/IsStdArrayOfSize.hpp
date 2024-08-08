// Reused from SpECTRE : https://spectre-code.org/
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <array>
#include <cstddef>
#include <type_traits>

namespace tt {

  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type T is a std::array of a given size
  ///
  /// \details
  /// Given a size_t `N` and type `T` derives from std::true_type if `T`
  /// is a std::array of size `N` and from std::false_type otherwise.
  ///
  /// \usage
  /// For any type `T`
  /// \code
  /// using result = tt::is_std_array_of_size<N, T>;
  /// \endcode
  ///
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If `T` is a std::array of size `N` then
  /// \code
  /// typename result::type = cpp17::bool_constant<true>;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = cpp17::bool_constant<false>;
  /// \endcode
  ///
  /// \example
  /// \snippet Test_IsStdArrayOfSize.cpp is_std_array_of_size_example
  /// \see is_std_array
  /// \tparam T the type to check
  template <size_t N, typename T>
  struct is_std_array_of_size : std::false_type {};

  /// \cond HIDDEN_SYMBOLS
  template <size_t N, typename T>
  struct is_std_array_of_size<N, std::array<T, N>> : std::true_type {};
  /// \endcond

  /// \see is_std_array_of_size
  template <size_t N, typename T>
  constexpr bool is_std_array_of_size_v = is_std_array_of_size<N, T>::value;

  /// \see is_std_array_of_size
  template <size_t N, typename T>
  using is_std_array_of_size_t = typename is_std_array_of_size<N, T>::type;
  // @}

}  // namespace tt
