#pragma once

namespace blocks {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations of block variable types
  //
  //////////////////////////////////////////////////////////////////////////////

  template <typename Variable>
  struct VariableSlice;

  template <typename Variable>
  struct ConstVariableSlice;

  template <typename ParameterTag, typename RankTag, typename... Tags>
  struct Variable;

  template <typename ParameterTag, typename RankTag, typename... Tags>
  struct InitializedVariable;

  template <typename VariableOrDerivedVariable>
  struct same_as;

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations of Initializer
  //
  //////////////////////////////////////////////////////////////////////////////
  namespace detail{
    template <typename ParameterTag, typename NamedFunction>
    struct BlockInitializerVariable;
  }

  template <typename Plugin, typename ...InitializerVariables>
  struct BlockInitializer;
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks
