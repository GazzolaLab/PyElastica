#pragma once

//******************************************************************************
// Includes
//******************************************************************************

//
#include "Systems/Block/BlockVariables/Types.hpp"
//
#include "Systems/Block/BlockVariables/Aliases.hpp"
#include "Systems/Block/BlockVariables/Protocols.hpp"
//
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace blocks {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Check whether a given type `Var` conforms to
   * blocks::protocols::Variable
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `Var` conforms to
   * blocks::protocols::Variable, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `Var`,
   * \code
   * using result = IsVariable<Var>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `Var` inherits from protocols::Variable, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet BlockVariables/Test_TypeTraits.cpp is_variable_example
   *
   * \tparam Var : the type to check
   *
   * \see blocks::protocols::Variable, blocks::Variable
   */
  template <typename Var>
  struct IsVariable : public ::tt::conforms_to<Var, protocols::Variable> {};
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a BlockInitializer
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::BlockInitializer, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsBlockInitializer<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::BlockInitializer, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet BlockVariables/Test_TypeTraits.cpp is_block_initializer_example
   *
   * \tparam B : the type to check
   *
   * \see Block
   */
  template <typename B>
  struct IsBlockInitializer : ::tt::is_a<BlockInitializer, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Metafunction to obtain slice types of a typelist of block
   * variables
   * \ingroup blocks
   *
   * \details
   * The as_slices template type helps obtain the slice types of a typelist of
   * declared blocks::Variable, used as shown below. The resulting type
   * contains nested types `slices` and `const_slices` which are typelists
   * containing the slice types and constant slice types of the
   * blocks::Variable.
   *
   * \example
   * \code
   *  // ... Declare variables Var1, Var2, Var3
   *
   *  // Typelist of all variable slices
   *  using VariablesList = tmpl::list<Var1, Var2, Var3>;
   *
   *  // Type containing information about slices
   *  using VariableSlicesInformation = as_slices<VariablesList>;
   *
   *  // TypeList of slices of all variables
   *  using VariableSlices = typename VariableSlicesInformation::slices;
   *
   *  // TypeList of constant slices of all variables
   *  using VariableConstSlices
   *    = typename VariableSlicesInformation::const_slices;
   * \endcode
   *
   * \tparam L typelist of block::Variables
   *
   * \see blocks::Variables, Block
   */
  template <typename L>
  struct as_slices {
    //**Type definitions******************************************************
    //! Typelist of slices
    using slices = tmpl::transform<L, tmpl::bind<slice_type_t, tmpl::_1>>;
    //! Typelist of const slices
    using const_slices =
        tmpl::transform<L, tmpl::bind<const_slice_type_t, tmpl::_1>>;
    //************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helper to form associations between a list of types `L` and the
   * result of applying metafunction `Op` to types in `L`
   * \ingroup block_tt
   *
   * \details
   * Forms associations (both-ways) between a types `T` belonging to a list `L`
   * and the result type of applying metafunction `Op` to `T`
   *
   * \usage
   * 1. For any metafunction `Op` and a list of types `L`
   * \code
   * using result = FormMap::from<Op>::to<L>;
   * \endcode
   * forms an associative type map from `Op<T>` to `T` for `T` in `L`.
   *
   * 2. For any metafunction `Op` and a list of types `L`
   * \code
   * using result = FormMap::to<Op>::from<L>;
   * \endcode
   * forms an associative type map from `T` to `Op<T>` for `T` in `L`.
   *
   * \metareturns
   * A type `A` which is a map from `Op<T>` to `T` when invoked with signature
   * (1) above and from `T` to `Op<T>` when invoked with signature (2), for all
   * types `T` in the typelist `L`
   *
   * \example
   * With the following setup
   * \snippet BlockVariables/Test_TypeTraits.cpp map_setup
   * we form `Op<T>` to `T` association using
   * \snippet BlockVariables/Test_TypeTraits.cpp map_from_to_eg
   * and we form `T` to `Op<T>` association using
   * \snippet BlockVariables/Test_TypeTraits.cpp map_to_from_eg
   *
   * \tparam Op : meta-function to act on `T` in `L`
   * \tparam L :  list of types
   */
  struct FormMap {
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Helper for associating from `Op<T>` to `T`
     * \ingroup block_tt
     */
    template <template <typename> class Op>
    struct from {
      //**Type definitions******************************************************
      //! Wrapper type to generate map
      template <typename... T>
      using TemplateMap_t = tmpl::map<tmpl::pair<Op<T>, T>...>;

      //! Wrapped map type
      template <typename L>
      using to = tmpl::wrap<L, TemplateMap_t>;
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Helper for associating from `T` to `Op<T>`
     * \ingroup block_tt
     */
    template <template <typename> class Op>
    struct to {
      //**Type definitions******************************************************
      //! Wrapper type to generate map
      template <typename... T>
      using TemplateMap_t = tmpl::map<tmpl::pair<T, Op<T>>...>;

      //! Wrapped map type
      template <typename L>
      using from = tmpl::wrap<L, TemplateMap_t>;
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helper to form associations between Variables and Parameters
   * \ingroup block_tt
   *
   * \details
   * Forms associations between underlying parameters `T` of a list of block
   * variables `L`.
   *
   * \usage
   * For a list of block variable types `L`
   * \code
   * using VariableMapping = blocks::VariableMapping<L>;
   * \endcode
   * sets up the map from parameter_t<T> to `T` for `T` in `L`.
   *
   * \example
   * With the following setup
   * \snippet BlockVariables/Test_TypeTraits.cpp variable_mapping_eg
   *
   * \tparam TypeListOfVariables: A tmpl::list of Variables
   */
  template <typename TypeListOfVariables>
  struct VariableMapping {
    //**Type definitions********************************************************
    //! Map from the underlying tag parameters to the variables
    using Mapping = FormMap::from<parameter_t>::to<TypeListOfVariables>;
    //! Lookup for the parameter
    template <typename ParameterTag>
    using variable_from = tmpl::lookup<Mapping, ParameterTag>;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace blocks
