#pragma once

namespace cpp17 {

  //****************************************************************************
  /*!\brief Given a set of types, returns `void`
  // \ingroup type_traits_group
  //
  // \details
  // Given a list of types, returns `void`. This is very useful for
  // controlling name lookup resolution via partial template specialization.
  //
  // \usage
  // For any set of types `Ti`,
     \code
       using result = cpp17::void_t<T0, T1, T2, T3>;
     \endcode
  //
  // \metareturns
  // void
  //
  // \semantics
  // For any set of types `Ti`,
     \code
       using result = void;
     \endcode
  //
  // \example
  // \snippet TypeTraits/Test_Void.cpp void_t_example
  //
  // \tparam Ts... A set of types
  //
  // \see std::void_t
  */
  template <typename... Ts>
  using void_t = void;
  //****************************************************************************

}  // namespace cpp17
