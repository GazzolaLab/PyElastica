#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace cpp20 {

  /// \ingroup type_traits_group
  /// \brief Given a type T, removes reference and cv-qualifiers
  ///
  /// \details
  /// If the type T is a reference type, provides the member typedef type which
  /// is the type referred to by T with its topmost cv-qualifiers removed.
  /// Otherwise type is T with its topmost cv-qualifiers removed. Useful for
  /// metaprogramming within expression templates
  template <class T>
  struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
  };

  // todo document
  template <class T>
  using remove_cvref_t = typename remove_cvref<T>::type;

}  // namespace cpp20
