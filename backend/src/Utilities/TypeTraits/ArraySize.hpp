#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <array>
#include <cstddef>
#include <type_traits>

namespace tt {

  namespace detail {
    template <typename T, std::size_t N>
    std::integral_constant<std::size_t, N> array_size_impl(
        const std::array<T, N>& /*array*/);
  }  // namespace detail

  // @{
  /// \ingroup type_traits_group
  /// \brief Get the size of a std::array as a std::integral_constant
  ///
  /// \details
  /// Given a std::array, `Array`, returns a std::integral_constant that has the
  /// size of the array as its value
  ///
  /// \usage
  /// For a std::array `T`
  /// \code
  /// using result = tt::array_size<T>;
  /// \endcode
  ///
  /// \metareturns
  /// std::integral_constant<std::size_t>
  ///
  /// \semantics
  /// For a type `T`,
  /// \code
  /// using tt::array_size<std::array<T, N>> =
  /// std::integral_constant<std::size_t, N>; \endcode
  ///
  /// \example
  /// \snippet Test_ArraySize.cpp array_size_example
  /// \tparam Array the whose size should be stored in value of array_size
  template <typename Array>
  using array_size =
      decltype(detail::array_size_impl(std::declval<const Array&>()));
  // @}

}  // namespace tt
