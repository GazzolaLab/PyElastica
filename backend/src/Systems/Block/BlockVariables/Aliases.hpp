#pragma once

namespace blocks {

  //============================================================================
  //
  //  ALIAS DECLARATIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Gets the nested `Parameter` from `T`
   * \ingroup block_tt
   *
   * The parameter_t alias declaration provides a convenient shortcut to access
   * the nested `Parameter` type definition of the given type \a T. The
   * following code example shows both ways to access the nested type
   * definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::Parameter;
   *   using Type2 = parameter_t<T>;
   * \endcode
   *
   * \see Variable, blocks::protocols::Variable
   */
  /// [parameter_t]
  template <typename T>
  using parameter_t = typename T::Parameter;
  /// [parameter_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `Rank` from `T`
   * \ingroup block_tt
   *
   * The rank_t alias declaration provides a convenient shortcut to access
   * the nested `Rank` type definition of the given type \a T. The
   * following code example shows both ways to access the nested type
   * definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::Rank;
   *   using Type2 = rank_t<T>;
   * \endcode
   *
   * \see Variable, blocks::protocols::Variable
   */
  /// [rank_t]
  template <typename T>
  using rank_t = typename T::Rank;
  /// [rank_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `SliceType` from `T`
   * \ingroup block_tt
   *
   * The slice_type_t alias declaration provides a convenient shortcut to access
   * the nested `SliceType` type definition of the given type \a T. The
   * following code example shows both ways to access the nested type
   * definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::SliceType;
   *   using Type2 = slice_type_t<T>;
   * \endcode
   *
   * \see Variable, blocks::protocols::Variable
   */
  /// [slice_type_t]
  template <typename T>
  using slice_type_t = typename T::SliceType;
  /// [slice_type_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `ConstSliceType` from `T`
   * \ingroup block_tt
   *
   * The const_slice_type_t alias declaration provides a convenient shortcut to
   * access the nested `ConstSliceType` type definition of the given type \a T.
   * The following code example shows both ways to access the nested type
   * definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::ConstSliceType;
   *   using Type2 = const_slice_type_t<T>;
   * \endcode
   *
   * \see Variable, blocks::protocols::Variable
   */
  /// [const_slice_type_t]
  template <typename T>
  using const_slice_type_t = typename T::ConstSliceType;
  /// [const_slice_type_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `Initializer` from `T`
   * \ingroup block_tt
   *
   * The initializer_t alias declaration provides a convenient shortcut to
   * access the nested `Initializer` type definition of the given type \a T. The
   * following code example shows both ways to access the nested type
   * definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::Initializer;
   *   using Type2 = initializer_t<T>;
   * \endcode
   *
   * \see Variable, blocks::protocols::Variable
   */
  /// [initializer_t]
  template <typename T>
  using initializer_t = typename T::Initializer;
  /// [initializer_t]
  //****************************************************************************

}  // namespace blocks
