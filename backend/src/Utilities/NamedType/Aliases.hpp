#pragma once

namespace named_type {

  //============================================================================
  //
  //  ALIAS DECLARATIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Gets the nested `TagType` from `T`
  // \ingroup named_type
  //
  // The tag_type_t alias declaration provides a convenient shortcut to access
  // the nested `TagType` type definition of a given type \a T. The
  // following code example shows both ways to access the nested type
  // definition:
  //
  // \example
     \code
       using Type1 = typename T::TagType;
       using Type2 = tag_type<T>;
     \endcode
  //
  // \see NamedType
  */
  template <typename T>
  using tag_type = typename T::TagType;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `UnderlyingType` from `T`
  // \ingroup named_type
  //
  // The underlying_type_t alias declaration provides a convenient shortcut to
  // access the nested `UnderlyingType` type definition of a given type \a T.
  // The following code example shows both ways to access the nested type
  // definition:
  //
  // \example
     \code
       using Type1 = typename T::UnderlyingType;
       using Type2 = underlying_type<T>;
     \endcode
  //
  // \see NamedType
  */
  template <typename T>
  using underlying_type = typename T::UnderlyingType;
  //****************************************************************************

}  // namespace named_type
