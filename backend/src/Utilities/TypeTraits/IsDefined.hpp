#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/TypeTraits/Void.hpp"

namespace tt {

  /// \note
  /// Developers use this variable template with caution as it leads to
  /// weird ODR violations :
  /// https://devblogs.microsoft.com/oldnewthing/20190711-00/?p=102682

  //****************************************************************************
  /*!\brief Variable template for checking if type is defined
  // \ingroup type_traits
  //
  // \details
  // The is_defined_v variable template checks for definition (not only
  // declaration!) of the input type \c T. If the type \c T is defined, then the
  // value is set to true, otherwise it is set to false
  //
  // \usage
  // For any type `T`,
     \code
     constexpr bool result = tt::is_defined_v<T>;
     \endcode
  //
  // \returns
  // bool
  //
  // \example
     \code
       constexpr bool value1 = tt::is_defined_v<int>; // evaluates to true
       struct A;
       constexpr bool value2 = tt::is_defined_v<A>; // evaluates to false
     \endcode
  */
  template <typename, typename = void>
  constexpr bool is_defined_v = false;
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of the is_defined_v variable template if type \c T
  // exists
  // \ingroup type_traits
  */
  template <typename T>
  constexpr bool is_defined_v<T, cpp17::void_t<decltype(sizeof(T))>> = true;
  /*! \endcond */
  //****************************************************************************

}  // namespace tt
