#pragma once

namespace blocks {

  //============================================================================
  //
  //  ALIAS DECLARATIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Gets the nested `Variables` from `T`
   * \ingroup block_tt
   *
   * The variables_t alias declaration provides a convenient
   * shortcut to access the nested `Variables` type definition of
   * the given type \a T. The following code example shows both ways to access
   * the nested type definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::Variables;
   *   using Type2 = variables_t<T>;
   * \endcode
   *
   * \see Block, BlockSlice
   */
  // [variables_t]
  template <typename T>
  using variables_t = typename T::Variables;
  // [variables_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `InitializedVariables` from `T`
   * \ingroup block_tt
   *
   * The initialized_variables_t alias declaration provides a convenient
   * shortcut to access the nested `InitializedVariables` type definition of
   * the given type \a T. The following code example shows both ways to access
   * the nested type definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::InitializedVariables;
   *   using Type2 = initialized_variables_t<T>;
   * \endcode
   *
   * \see Block, BlockSlice
   */
  // [initialized_variables_t]
  template <typename T>
  using initialized_variables_t = typename T::InitializedVariables;
  // [initialized_variables_t]
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets the nested `ComputedVariables` from `T`
   * \ingroup block_tt
   *
   * The computed_variables_t alias declaration provides a convenient
   * shortcut to access the nested `ComputedVariables` type definition of
   * the given type \a T. The following code example shows both ways to access
   * the nested type definition:
   *
   * \example
   * \code
   *   using Type1 = typename T::ComputedVariables;
   *   using Type2 = computed_variables_t<T>;
   * \endcode
   *
   * \see Block, BlockSlice
   */
  // [computed_variables_t]
  template <typename T>
  using computed_variables_t = typename T::ComputedVariables;
  // [computed_variables_t]
  //****************************************************************************

}  // namespace blocks
