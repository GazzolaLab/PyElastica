#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TypeTraits/Void.hpp"

namespace tt {

  // @{
  /// \ingroup type_traits_group
  /// \brief Check if type `T` has operator<<(`S`, `T`) defined.
  ///
  /// \details
  /// Inherits from std::true_type if the type `T` has operator<<(`S`, `T`)
  /// defined for a stream `S`, otherwise inherits from std::false_type
  ///
  /// \usage
  /// For any type `T` and stream type `S`,
  /// \code
  /// using result = tt::is_streamable<S, T>;
  /// \endcode
  ///
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If the type `T` has operator<<(`S`, `T`) defined for stream `S`, then
  /// \code
  /// typename result::type = std::true_type;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = std::false_type;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp is_streamable_example
  /// \see std::cout std::ifstream std::sstream std::ostream
  /// \tparam S the stream type, e.g. std::stringstream or std::ostream
  /// \tparam T the type we want to know if it has operator<<
  template <typename S, typename T, typename = cpp17::void_t<>>
  struct is_streamable : std::false_type {};
  /// \cond HIDDEN_SYMBOLS
  template <typename S, typename T>
  struct is_streamable<
      S, T,
      cpp17::void_t<decltype(std::declval<std::add_lvalue_reference_t<S>>()
                             << std::declval<T>()),
                    Requires<not std::is_same<S, T>::value>>> : std::true_type {
  };
  /// \endcond

  /// \see is_streamable
  template <typename S, typename T>
  constexpr bool is_streamable_v = is_streamable<S, T>::value;

  /// \see is_streamable
  template <typename S, typename T>
  using is_streamable_t = typename is_streamable<S, T>::type;

  // @}

}  // namespace tt
