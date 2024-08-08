#pragma once

//******************************************************************************
// Includes
//******************************************************************************

///
#include "Systems/Block/Block/Protocols.hpp"
#include "Systems/Block/Block/Types.hpp"
///
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a Block
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::Block, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsBlock<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::Block, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_block_example
   *
   * \tparam B : the type to check
   *
   * \see Block
   */
  template <typename B>
  struct IsBlock : ::tt::is_a<Block, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsBlock type trait.
   * \ingroup block_tt
   *
   * The is_block_v variable template provides a convenient
   * shortcut to access the nested \a value of the IsBlock
   * class template. For instance, given the type \a T the following two
   * statements are identical:
   * \example
   * \code
   *   constexpr bool value1 = IsBlock<T>::value;
   *   constexpr bool value2 = is_block_v<T>;
   * \endcode
   *
   * \tparam B : the type to check
   *
   * \see IsBlock
   */
  template <typename B>
  constexpr bool is_block_v = IsBlock<B>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a BlockSlice
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::BlockSlice, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsBlockSlice<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::BlockSlice, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_blockslice_example
   *
   * \tparam B : the type to check
   *
   * \see BlockSlice
   */
  template <typename B>
  struct IsBlockSlice : ::tt::is_a<BlockSlice, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsBlockSlice type trait.
   * \ingroup block_tt
   *
   * The is_block_slice_v variable template provides a convenient
   * shortcut to access the nested \a value of the IsBlockSlice
   * class template. For instance, given the type \a T the following two
   * statements are identical:
   * \example
   * \code
   *   constexpr bool value1 = IsBlockSlice<T>::value;
   *   constexpr bool value2 = is_block_slice_v<T>;
   * \endcode
   *
   * \tparam B : the type to check
   *
   * \see IsBlockSlice
   */
  template <typename B>
  constexpr bool is_block_slice_v = IsBlockSlice<B>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a ConstBlockSlice
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::ConstBlockSlice, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsConstBlockSlice<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::ConstBlockSlice, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_const_blockslice_example
   *
   * \tparam B : the type to check
   *
   * \see ConstBlockSlice
   */
  template <typename B>
  struct IsConstBlockSlice : ::tt::is_a<ConstBlockSlice, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsConstBlockSlice type trait.
   * \ingroup block_tt
   *
   * The is_const_block_slice_v variable template provides a convenient
   * shortcut to access the nested \a value of the IsConstBlockSlice
   * class template. For instance, given the type \a T the following two
   * statements are identical:
   * \example
   * \code
   *   constexpr bool value1 = IsConstBlockSlice<T>::value;
   *   constexpr bool value2 = is_const_block_slice_v<T>;
   * \endcode
   *
   * \tparam B : the type to check
   *
   * \see IsBlockSlice
   */
  template <typename B>
  constexpr bool is_const_block_slice_v = IsConstBlockSlice<B>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a BlockView
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::BlockView, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsBlockView<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::BlockView, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_blockview_example
   *
   * \tparam B : the type to check
   *
   * \see BlockView
   */
  template <typename B>
  struct IsBlockView : ::tt::is_a<BlockView, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsBlockView type trait.
   * \ingroup block_tt
   *
   * The is_block_view_v variable template provides a convenient
   * shortcut to access the nested \a value of the IsBlockView
   * class template. For instance, given the type \a T the following two
   * statements are identical:
   * \example
   * \code
   *   constexpr bool value1 = IsBlockView<T>::value;
   *   constexpr bool value2 = is_block_view_v<T>;
   * \endcode
   *
   * \tparam B : the type to check
   *
   * \see IsBlockView
   */
  template <typename B>
  constexpr bool is_block_view_v = IsBlockView<B>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` is a ConstBlockView
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a template specialization of a
   * blocks::ConstBlockView, otherwise inherits from std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = IsConstBlockView<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of elastica::ConstBlockView, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_const_blockview_example
   *
   * \tparam B : the type to check
   *
   * \see ConstBlockView
   */
  template <typename B>
  struct IsConstBlockView : ::tt::is_a<ConstBlockView, B> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsConstBlockView type trait.
   * \ingroup block_tt
   *
   * The is_const_block_view_v variable template provides a convenient
   * shortcut to access the nested \a value of the IsConstBlockView
   * class template. For instance, given the type \a T the following two
   * statements are identical:
   * \example
   * \code
   *   constexpr bool value1 = IsConstBlockView<T>::value;
   *   constexpr bool value2 = is_const_block_view_v<T>;
   * \endcode
   *
   * \tparam B : the type to check
   *
   * \see IsBlockView
   */
  template <typename B>
  constexpr bool is_const_block_view_v = IsConstBlockView<B>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `P` is a Plugin type
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` is a Plugin type otherwise inherits
   * from std::false_type.
   *
   * \usage
   * For any type `P`,
   * \code
   * using result = blocks::IsPlugin<P>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `P` is marked as a Plugin by inheriting from
   * blocks::protocols::Plugin, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp is_plugin_example
   *
   * \tparam P : the type to check
   *
   * \see blocks::protocols::Plugin
   */
  template <typename P>
  struct IsPlugin : public tt::conforms_to<P, protocols::Plugin> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for the IsPlugin type trait.
   * \ingroup block_tt
   *
   * The is_plugin_v variable template provides a convenient shortcut to access
   * the nested `value` of the IsPlugin class template. For instance, given the
   * type `T` the following two statements are identical:
   *
   * \example
   * \code
   *   constexpr bool value1 = IsPlugin<T>::value;
   *   constexpr bool value2 = is_plugin_v<T>;
   * \endcode
   *
   * \tparam P : the type to check
   *
   * \see IsPlugin
   */
  template <typename P>
  constexpr bool is_plugin_v = IsPlugin<P>::value;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check whether a given type `B` models the \a Block concept
   * \ingroup block_tt
   *
   * \details
   * Inherits from std::true_type if `B` models the \a Block concept, that is
   * `B` is one of blocks::Block, blocks::BlockSlice, otherwise inherits from
   * std::false_type.
   *
   * \usage
   * For any type `B`,
   * \code
   * using result = ModelsBlockConcept<B>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `B` is an instantiation of either a blocks::Block or a blocks
   * ::BlockSlice, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp models_block_concept_example
   *
   * \tparam B : the type to check
   *
   * \see Block, BlockSlice, ConstBlockSlice, BlockView
   */
  template <typename B>
  struct ModelsBlockConcept
      : ::cpp17::disjunction<IsBlock<B>, IsBlockSlice<B>, IsConstBlockSlice<B>,
                             IsBlockView<B>, IsConstBlockView<B>> {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get the plugin type `P` of a block type `B`
   * \ingroup block_tt
   *
   * \details
   * The PluginTrait type trait has a nested member `type` representing the
   * plugin `P` when the meta-argument `B` models block concept
   *
   * \usage
   * For any type `B` such that ModelsBlockConcept<B>::type = std::true_type,
   * \code
   * using result = PluginTrait<B>;
   * \endcode
   * \metareturns
   * the type `T = P` such that `B = X<P>` where `X` is the type modeling block
   * concept
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp plugin_trait_example
   *
   * \tparam B Type (modeling block concept) whose `PluginTrait` is to be
   * retrieved
   *
   * \see ModelsBlockConcept
   */
  template <typename B>
  struct PluginTrait;
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plain plugin
   * \ingroup block_tt
   */
  template <typename Plugin>
  struct PluginTrait<Block<Plugin>> {
    template <template <typename> class>
    using type = Plugin;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block, without
   * Traits
   * \ingroup block_tt
   */
  template <template <template <typename /*Plugin*/> class /*BlockLike*/,
                      template <typename /*InstantiatedBlock*/>
                      class... /*Components*/>
            class Plugin,
            template <typename /*InstantiatedBlock*/> class... Components>
  struct PluginTrait<Block<Plugin<Block, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block and Traits
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<Block<Plugin<Traits, Block, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Traits, Tag, Block
   * and components
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                typename /* Tag*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits, typename Tag,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<Block<Plugin<Traits, Block, Tag, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Tag, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plain plugin
   * \ingroup block_tt
   */
  template <typename Plugin>
  struct PluginTrait<BlockSlice<Plugin>> {
    template <template <typename> class>
    using type = Plugin;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block, without
   * Traits
   * \ingroup block_tt
   */
  template <template <template <typename /*Plugin*/> class /*BlockLike*/,
                      template <typename /*InstantiatedBlock*/>
                      class... /*Components*/>
            class Plugin,
            template <typename /*InstantiatedBlock*/> class... Components>
  struct PluginTrait<BlockSlice<Plugin<BlockSlice, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block and Traits
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<BlockSlice<Plugin<Traits, BlockSlice, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Traits, Tag, Block
   * and components
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                typename /* Tag*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits, typename Tag,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<
      BlockSlice<Plugin<Traits, BlockSlice, Tag, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Tag, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plain plugin
   * \ingroup block_tt
   */
  template <typename Plugin>
  struct PluginTrait<ConstBlockSlice<Plugin>> {
    template <template <typename> class>
    using type = Plugin;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block, without
   * Traits
   * \ingroup block_tt
   */
  template <template <template <typename /*Plugin*/> class /*BlockLike*/,
                      template <typename /*InstantiatedBlock*/>
                      class... /*Components*/>
            class Plugin,
            template <typename /*InstantiatedBlock*/> class... Components>
  struct PluginTrait<ConstBlockSlice<Plugin<ConstBlockSlice, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block and Traits
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<
      ConstBlockSlice<Plugin<Traits, ConstBlockSlice, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Traits, Tag, Block
   * and components
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                typename /* Tag*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits, typename Tag,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<
      ConstBlockSlice<Plugin<Traits, ConstBlockSlice, Tag, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Tag, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plain plugin
   * \ingroup block_tt
   */
  template <typename Plugin>
  struct PluginTrait<BlockView<Plugin>> {
    template <template <typename> class>
    using type = Plugin;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block, without
   * Traits
   * \ingroup block_tt
   */
  template <template <template <typename /*Plugin*/> class /*BlockLike*/,
                      template <typename /*InstantiatedBlock*/>
                      class... /*Components*/>
            class Plugin,
            template <typename /*InstantiatedBlock*/> class... Components>
  struct PluginTrait<BlockView<Plugin<BlockView, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block and Traits
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<BlockView<Plugin<Traits, BlockView, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Traits, Tag, Block
   * and components
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                typename /* Tag*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits, typename Tag,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<BlockView<Plugin<Traits, BlockView, Tag, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Tag, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plain plugin
   * \ingroup block_tt
   */
  template <typename Plugin>
  struct PluginTrait<ConstBlockView<Plugin>> {
    template <template <typename> class>
    using type = Plugin;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block, without
   * Traits
   * \ingroup block_tt
   */
  template <template <template <typename /*Plugin*/> class /*BlockLike*/,
                      template <typename /*InstantiatedBlock*/>
                      class... /*Components*/>
            class Plugin,
            template <typename /*InstantiatedBlock*/> class... Components>
  struct PluginTrait<ConstBlockView<Plugin<ConstBlockView, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Block and Traits
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<
      ConstBlockView<Plugin<Traits, ConstBlockView, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of PluginTrait for a plugin with Traits, Tag, Block
   * and components
   * \ingroup block_tt
   */
  template <
      template <typename /*Traits*/,
                template <typename /*Plugin*/> class /*BlockLike*/,
                typename /* Tag*/,
                template <typename /*Traits*/, typename /*InstantiatedBlock*/>
                class... /*Components*/>
      class Plugin,
      typename Traits, typename Tag,
      template <typename /*Traits*/, typename /*InstantiatedBlock*/>
      class... Components>
  struct PluginTrait<
      ConstBlockView<Plugin<Traits, ConstBlockView, Tag, Components...>>> {
    template <template <typename> class BlockLike>
    using type = Plugin<Traits, BlockLike, Tag, Components...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Eager result for obtaining plugin type `P` of a block model `B`
   * \ingroup block_tt
   *
   * \details
   * The PluginFrom type trait is a helper template to obtain the nested
   * template type `type` of PluginTrait, used as follows.
   *
   * \example
   * \snippet Block/Test_TypeTraits.cpp plugin_t_example
   *
   * \see PluginTrait
   */
  template <typename BlockLike>
  struct PluginFrom {
    template <template <typename> class NewBlock>
    struct to {
      using type =
          NewBlock<typename PluginTrait<BlockLike>::template type<NewBlock>>;
    };
    //  /// [plugin_t]
    //  template <typename B>
    //  using plugin_t = typename PluginTrait<B>::type;
    //  /// [plugin_t]
  };
  //****************************************************************************

}  // namespace blocks
