#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Utilities/AsTaggedTuple.hpp"

namespace blocks {

  //****************************************************************************
  /*!\brief Metafunction to define storage a typelist of block variables
   * \ingroup blocks
   *
   * \details
   * The as_block_variables template type helps obtain the heterogeneous
   * variable storage container from a typelist of declared blocks::Variable,
   * used as shown below.
   *
   * \example
   * \code
   *  // ... Declare variables Var1, Var2, Var3
   *
   *  // TypeList of all variables
   *  using VariablesList = tmpl::list<Var1, Var2, Var3>;
   *
   *  // Convert into a storage type which has value semantics
   *  using BlockVariables = as_block_variables<VariablesList>;
   * \endcode
   *
   * \tparam L typelist of block::Variables
   *
   * \see blocks::Variables, Block
   */
  template <typename L>
  using as_block_variables = tmpl::as_tagged_tuple<L>;
  //****************************************************************************

}  // namespace blocks
