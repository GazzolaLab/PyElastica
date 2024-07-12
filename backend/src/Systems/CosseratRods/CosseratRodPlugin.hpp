#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "ErrorHandling/Assert.hpp"

/// Forward declarations
#include "Systems/CosseratRods/Tags.hpp"
#include "Systems/CosseratRods/Types.hpp"
///
#include "Systems/Block/Block.hpp"  // for plugin, get
///
//
#include "Systems/common/Cardinality.hpp"
#include "Systems/common/IndexCheck.hpp"
//
#include "Utilities/TMPL.hpp"
///
#include <numeric>    // for std::accumulate
#include <stdexcept>  // for out of range errors
#include <string>
#include <utility>
#include <vector>

/*
 * Checklist
 *
 * Rod
  [x] masses_;
  [x] density_; (per rod or for entire)
  [x] restJ_;
  [x] rest_inv_J_;

  [ ] shear_forces_;
  [ ] external_forces_;
  [ ] intrinsic_shear_strain_
  [ ] shear_strain_;
  [ ] shear_torques_;
  [ ] external_torques_;
  [ ] bending_torques_;

  * Timestepper adapter (customization point for the Time-stepper)
   - Prerequisites : Geometry Component, Kinematics Component
  [ ] state

  * Elasticity layer
   - Prerequisites : Geometry Component
  [x] rest_B
  [x] rest_S
  [x] internal_model_loads_ (n_l)
  [x] internal_model_couples_ (tau_l)

  * Kinematics Component
   - Prerequisites : Geometry Component
  [x] velocities_;
  [x] omega_;
  [x] dilatation_rate_ (calculation, may be lazy)
  [x] curvature_rate_ (calculation, may be lazy)

    tangents
      |
  positions -> lengths
               /
  Q -> curvature
  sep : rest_lengths
  sep : rest_curvatures


  Shape Component
  [x] shape_factor
  [x] radii_;
  [x] volumes_;
   |  Geometry
   |  [x] rest_lengths_
   |  [x] lengths_
   |  [x] dilatations_;
   |  [x] voronoi_rest_lengths_
   |  [x] voronoi_lengths_
   |  [x] voronoi_dilatations_
   |  [x] intrinsic_curvature_
   |  [x] curvature_;
   |  [x] tangents_;
       | GeometryBase
       | [x] positions_;
       | [x] Q_;
 */

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      //========================================================================
      //
      //  CLASS DEFINITIONS
      //
      //========================================================================

      //************************************************************************
      /*!\brief Additional Variables corresponding to a Cosserat rod plugin
       * \ingroup cosserat_rod
       *
       * \details
       * CosseratRodPluginVariables contains the definitions of variables
       * used within the Blocks framework for a Cosserat rod, that do not belong
       * to any other component (such as geometry, elasticity).
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       *
       * \see elastica::cosserat_rod::CosseratRodPlugin
       */
      template <typename CRT>
      class CosseratRodPluginVariables {
       private:
        //**Type definitions****************************************************
        //! Traits type
        using Traits = CRT;
        //**********************************************************************

       protected:
        //**Variable definitions************************************************
        /*!\name Variable definitions*/
        //@{

        //**********************************************************************
        /*!\brief Variable marking number of elements within the Cosserat rod
         * hierarchy
         */
        struct NElement
            : public Traits::template CosseratRodInitializedVariable<
                  tags::NElement,                    //
                  typename Traits::DataType::Index,  //
                  typename Traits::Place::OnRod> {};
        //**********************************************************************

        //**Type definitions****************************************************
        //! List of computed variables
        using ComputedVariables = tmpl::list<>;
        //! List of initialized variables
        using InitializedVariables = tmpl::list<NElement>;
        //! List of all variables
        using Variables = tmpl::append<InitializedVariables, ComputedVariables>;
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace detail

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Plugin for implementing Cosserat rods within @ref blocks
     * in \elastica
     * \ingroup cosserat_rod
     *
     * \details
     * CosseratRodPlugin is the computational plugin modeling a Cosserat rod
     * for use with template types modeling a \c ComputationalBlock concept
     * (blocks::Block, blocks::BlockSlice) in \elastica.
     *
     * As with other Plugins, CosseratRodPlugin is templated for customization.
     * The three template parameters serve different purposes.
     *
     * The first template parameter is a Traits class `CRT` for customizing the
     * data -structures, placement and algorithms used. For example, see
     * elastica::cosserat_rod::CosseratRodTraits.
     *
     * The second template parameter is the Block-like entity which derives
     * from the CosseratRodPlugin using the CRTP pattern. For examples of this
     * pattern, refer to examples in @ref blocks.
     *
     * The third template parameter is reserved for multiple `Components`. These
     * are templated types controlling customizing orthogonal aspects of a
     * Cosserat rod. For examples, see @ref cosserat_rod_component
     *
     * \tparam CRT A valid Cosserat Rod Traits class
     * \tparam ComputationalBlock A template type modeling the
     * `ComputationalBlock` concept
     * \tparam Components Variadic components for customizing behavior
     *
     * \see blocks::Block, blocks::BlockSlice
     */
    template <typename CRT, template <typename> class ComputationalBlock,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components>
    class CosseratRodPlugin
        : public detail::CosseratRodPluginVariables<CRT>,
          public HasMultipleCardinality<
              CosseratRodPlugin<CRT, ComputationalBlock, Components...>>,
          public Components<
              CRT, ComputationalBlock<CosseratRodPlugin<CRT, ComputationalBlock,
                                                        Components...>>>...,
          public ::tt::ConformsTo<blocks::protocols::Plugin> {
     private:
      //**Type definitions******************************************************
      //! Traits type
      using Traits = CRT;
      //! This type
      using This = CosseratRodPlugin<Traits, ComputationalBlock, Components...>;
      //! Variable definitions
      using VariableDefinitions = detail::CosseratRodPluginVariables<Traits>;
      //! Size type
      using size_type = typename Traits::size_type;
      //! Index type
      using index_type = typename Traits::index_type;
      //************************************************************************

     private:
      //**CRTP section**********************************************************
      /*!\name CRTP section */
      //@{

      //**Type definitions******************************************************
      //! Type of the bottom level derived class
      using Self = ComputationalBlock<This>;
      //! Reference type of the bottom level derived class
      using Reference = Self&;
      //! const reference type of the bottom level derived class
      using ConstReference = Self const&;
      //************************************************************************

     public:
      //**Self method***********************************************************
      /*!\brief Access to the underlying derived
      //
      // \return Mutable reference to the underlying derived
      //
      // Safely down-casts this module to the underlying derived type, using
      // the Curiously Recurring Template Pattern (CRTP).
      */
      inline constexpr auto self() & noexcept -> Reference {
        return static_cast<Reference>(*this);
      }
      //************************************************************************

      //**Self method***********************************************************
      /*!\brief Access to the underlying derived
      //
      // \return Const reference to the underlying derived
      //
      // Safely down-casts this module to the underlying derived type, using
      // the Curiously Recurring Template Pattern (CRTP).
      */
      inline constexpr auto self() const& noexcept -> ConstReference {
        return static_cast<ConstReference>(*this);
      }
      //************************************************************************

      //@}
      //************************************************************************

     private:
      //**Useful aliases********************************************************
      //! Variable for marking number of elements
      using NElement = typename VariableDefinitions::NElement;

     public:
      //************************************************************************
      //! Tag to place on nodes
      using OnNode = typename Traits::Place::OnNode;
      //! Tag to place on elements
      using OnElement = typename Traits::Place::OnElement;
      //! Tag to place on voronois
      using OnVoronoi = typename Traits::Place::OnVoronoi;
      //! Tag to place on the whole rod rather than at discrete points
      using OnRod = typename Traits::Place::OnRod;
      //************************************************************************

     public:
      // These are promoted to public scope for protocols to confirm they work
      // as well as any type traits based on the Plugin (for example, to check
      // whether to have a default initializer). Additionally this simplifies
      // and honors the contract with BlockFacade.

      //**Block type definitions************************************************
      //! List of computed variables
      using ComputedVariables =
          tmpl::append<typename VariableDefinitions::ComputedVariables,
                       typename Components<CRT, Self>::ComputedVariables...>;
      //! List of initialized variables
      using InitializedVariables =
          tmpl::append<typename VariableDefinitions::InitializedVariables,
                       typename Components<CRT, Self>::InitializedVariables...>;
      //! List of all variables
      using Variables =
          tmpl::append<typename VariableDefinitions::Variables,
                       typename Components<CRT, Self>::Variables...>;
      //************************************************************************

     protected:
      //************************************************************************
      /*!\brief Initialize method for the entire cosserat rod
       *
       * \details
       * initialize() is responsible for initializing the variables present
       * in this plugin and all its components in the final block of data. For
       * InitializedVariables, it fetches the corresponding initializer from
       * the set of `initializers` to fill in the data. For ComputedVariables
       * the default value is usually set. The plugin then calls the
       * corresponding `initialize` method for each component and hence fills in
       * all component Variables as well.
       *
       * \tparam DownstreamBlock The final block-like object which is
       * derived from the current component
       * \tparam CosseratInitializer An Initializer corresponding to the
       * Cosserat rod hierarchy
       */
      template <template <typename> class BlockLike,
                typename CosseratInitializer>
      static void initialize(
          CosseratRodPlugin<Traits, BlockLike, Components...>& this_component,
          CosseratInitializer&& initializer) {
        // initialize the elements boi
        {
          using Variable = typename VariableDefinitions::NElement;
          using Tag = blocks::initializer_t<Variable>;

          auto&& variable(blocks::get<Tag>(this_component.self()));
          // this automatically is deduced to 1
          constexpr index_type index_of_rod = 0UL;
          auto const n_elem = blocks::get<Tag>(cpp17::as_const(initializer))();
          ELASTICA_ASSERT(
              Variable::slice(variable, index_of_rod) == n_elem,
              "Invariant violation detected, contact the developers!");
        }

        // Pass self here to make minimal changes to the tests.
        EXPAND_PACK_LEFT_TO_RIGHT(Components<CRT, Self>::initialize(
            this_component.self(),
            std::forward<CosseratInitializer>(initializer)));
      }
      //************************************************************************

      //**Utility functions*****************************************************
      /*!\name Utility functions */
      //@{

     public:
      //************************************************************************
      /*!\name Element methods*/
      //@{
      /*!\brief Gets the n_elements buffer of the current rod
       */
      inline constexpr decltype(auto) n_elements() & noexcept {
        // this version is only for testing
        return blocks::get<tags::NElement>(self());
      }

      inline constexpr decltype(auto) n_elements() const& noexcept {
        // this version is only for testing
        return blocks::get<tags::NElement>(self());
      }

      inline auto n_elements(index_type const unit) & noexcept -> size_type& {
        // Zero-based indexing
        return NElement::slice(n_elements(), unit);
        // return n_elements()[unit];
      }

      inline auto n_elements(index_type const unit) const& noexcept
          -> size_type const& {
        // Zero-based indexing
        return NElement::slice(n_elements(), unit);
        // return n_elements()[unit];
      }

      inline auto n_elements(index_type const start_unit,
                             size_type const slice_size) const& noexcept
          -> size_type {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"

        // may need to change based on elements() structure
        auto&& elements_buffer = n_elements();
        auto const start =
            elements_buffer.begin() + static_cast<std::size_t>(start_unit);
        return OnElement::get_dofs(
            std::accumulate(start, start + static_cast<std::size_t>(slice_size),
                            static_cast<size_type>(n_ghosts(OnElement{}) *
                                                   (slice_size - 1UL))));
#pragma GCC diagnostic pop
      }
      //@}
      //************************************************************************

     public:
      //************************************************************************
      /*!\name Ghost methods*/
      //@{
      /*!\brief Gets the intended number of ghosts for the Cosserat rod, for
       * each placement type. This requires all placement types to be different
       * types and not aliases.
       *
       * \note
       * We can have a ghost policy here, but its unnecessary as only
       * one optimal implementation exists, for a second-order implementation.
       */
      static inline constexpr auto n_ghosts(OnElement /*meta*/) noexcept
          -> size_type {
        return OnElement::n_ghosts();
      };
      static inline constexpr auto n_ghosts(OnNode /* meta */) noexcept
          -> size_type {
        return OnNode::n_ghosts();
      };
      static inline constexpr auto n_ghosts(OnVoronoi /* meta */) noexcept
          -> size_type {
        return OnVoronoi::n_ghosts();
      };
      //@}
      //************************************************************************

     public:
      //************************************************************************
      /*!\brief Returns number of units (unique, individual cosserat rods)
       * stored in the block
       */
      inline constexpr auto n_units() const noexcept -> size_type {
        return n_elements().size();
      }
      //************************************************************************

      //************************************************************************
      /*!\name Total methods*/
      //@{
      /*!\brief Gets the total number of elements/nodes/voronois for the
       * Cosserat rod, for each placement type.
       */
      inline auto total_n_elements() const noexcept -> size_type {
        return n_elements(0UL, n_units());
      }
      inline auto total_n_nodes() const noexcept -> size_type {
        return OnNode::get_dofs(total_n_elements());
      }
      inline auto total_n_voronois() const noexcept -> size_type {
        return OnVoronoi::get_dofs(total_n_elements());
      }
      //@}
      //************************************************************************

      //************************************************************************
      /*!\brief Checks if the current Block/BlockSlice is empty
       */
      inline auto empty() const noexcept -> bool {
        // n_elements().empty() doesn't work with blaze
        return not n_elements().size();
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Returns the size of the current Block/BlockSlice
       */
      inline auto size() const noexcept { return total_n_elements(); }
      //************************************************************************

      //@}
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Documentation stub with tags of CosseratRodPlugin
     * \ingroup cosserat_rod_component
     *
     * | Cosserat Rod Variables   ||
     * |--------------------------|-----------------------------|
     * | On Rod                   | elastica::tags::NElement    |
     */
    template <typename CRT, template <typename> class ComputationalBlock,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components>
    using CosseratRodPluginTagsDocsStub =
        CosseratRodPlugin<CRT, ComputationalBlock, Components...>;
    //**************************************************************************

  }  // namespace cosserat_rod

  //****************************************************************************
  /*!\brief Checks whether indices are within range of accessible values
   * \ingroup cosserat_rod
   * */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename IndexType>
  auto index_check(
      cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                      Components...> const& block_like,
      IndexType index_to_be_sliced) -> std::size_t {
    return index_check_helper(size(block_like.self()),
                              std::move(index_to_be_sliced));
  }
  //****************************************************************************

}  // namespace elastica

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of Metadata for a
   * elastica::cosserat_rod::CosseratRodPlugin
   * \ingroup cosserat_rod
   *
   * \tparam CRT A valid Cosserat Rod Traits class
   * \tparam ComputationalBlock A template type modeling the
   * `ComputationalBlock` concept
   * \tparam Components Variadic components for customizing behavior
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  struct Metadata<elastica::cosserat_rod::CosseratRodPlugin<
      CRT, ComputationalBlock, Components...>> {
    template <typename Var>
    static inline auto apply() noexcept
        -> std::vector<std::pair<std::string, std::string>> {
      // return {{"Staggering", pretty_type::get_name<typename Var::Stagger>()}};
      return {{"Staggering", Var::Stagger::name()}};
    }
  };
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks

namespace elastica {

  namespace cosserat_rod {

    //**************************************************************************
    /*!\brief The default tag parameter for a TaggedCosseratRodPlugin
     * \ingroup cosserat_rod
     *
     * \details
     * Default tag parameter for the third template parameter of the
     * TaggedCosseratRodPlugin
     * \see TaggedCosseratRodPlugin
     */
    struct DefaultCosseratRodPluginTag;
    //**************************************************************************

    //**************************************************************************
    /*!\brief A cosserat rod plugin with a tag parameter
     * \ingroup cosserat_rod
     *
     * \tparam CRT        The traits class for a Cosserat rod
     * \tparam ComputationalBlock A template type modeling the
     * `ComputationalBlock` concept
     * \tparam Tag        The tag for the Tagged Cosserat plugin
     * \tparam Components Components customizing a Cosserat rod (such as
     * geometry, elasticity)
     *
     * \see CosseratRodPlugin
     */
    template <typename CRT, template <typename> class ComputationalBlock,
              typename Tag = DefaultCosseratRodPluginTag,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components>
    class TaggedCosseratRodPlugin
        : public CosseratRodPlugin<CRT, ComputationalBlock, Components...> {};
    //**************************************************************************

  }  // namespace cosserat_rod

  //****************************************************************************
  /*!\brief Checks whether indices are within range of accessible values
   * \ingroup cosserat_rod
   * */
  template <typename CRT, template <typename> class ComputationalBlock,
            typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename IndexType>
  auto index_check(
      cosserat_rod::TaggedCosseratRodPlugin<CRT, ComputationalBlock, Tag,
                                            Components...> const& block_like,
      IndexType index_to_be_sliced) -> std::size_t {
    return index_check_helper(size(block_like.self()),
                              std::move(index_to_be_sliced));
  }
  //****************************************************************************

}  // namespace elastica

// namespace blocks {
//
//  namespace detail {
//
//    // Implementation for a block variable inside the hierarchy
//    template <typename BlockVariable, template <typename> class B,
//              typename CRT,                  // Cosserat Rod Traits
//              template <typename, typename>  // Skills
//              class... Components>
//    inline constexpr decltype(auto) get_impl(
//        B<elastica::CosseratRodPlugin<CRT, B, Components...>>& blk) noexcept {
//      return blk.template get<BlockVar>();
//    }
//
//    template <typename BlockVariable, template <typename> class B,
//              typename CRT,                  // Cosserat Rod Traits
//              template <typename, typename>  // Skills
//              class... Components>
//    inline constexpr decltype(auto) get_impl(
//        B<elastica::CosseratRodPlugin<CRT, B, Components...>> const&
//            blk) noexcept {
//      return blk.template get<BlockVar>();
//    }
//
//    template <typename BlockVariable, template <typename> class B,
//              typename CRT,                  // Cosserat Rod Traits
//              template <typename, typename>  // Skills
//              class... Components>
//    inline constexpr decltype(auto) get_impl(
//        B<elastica::CosseratRodPlugin<CRT, B, Components...>>&& blk) noexcept
//        {
//      return static_cast<Block_t&&>(blk).template get<BlockVar>();
//      return blk.template get<BlockVar>();
//    }
//
//  }  // namespace detail
//
//  // specialization of blocks::get for a CosseratRodPlugin
//  template <typename Tag, template <typename> class B,
//            typename CRT,                  // Cosserat Rod Traits
//            template <typename, typename>  // Skills
//            class... Components>
//  inline constexpr decltype(auto) get(
//      B<elastica::CosseratRodPlugin<CRT, B, Components...>>& blk) noexcept {
//    using Block_t = cpp20::remove_cvref_t<decltype(blk)>;
//    using BlockVar = typename Block_t::template var_from<Tag>;
//    return blk.template get<BlockVar>();
//  }
//
//  template <typename Tag, template <typename> class B,
//            typename CRT,                  // Cosserat Rod Traits
//            template <typename, typename>  // Skills
//            class... Components>
//  inline constexpr decltype(auto) get(
//      B<elastica::CosseratRodPlugin<CRT, B, Components...>> const&
//          blk) noexcept {
//    using Block_t = cpp20::remove_cvref_t<decltype(blk)>;
//    using BlockVar = typename Block_t::template var_from<Tag>;
//    return blk.template get<BlockVar>();
//  }
//
//  template <typename Tag, template <typename> class B,
//            typename CRT,                  // Cosserat Rod Traits
//            template <typename, typename>  // Skills
//            class... Components>
//  inline constexpr decltype(auto) get(
//      B<elastica::CosseratRodPlugin<CRT, B, Components...>>&& blk) noexcept {
//    using Block_t = cpp20::remove_cvref_t<decltype(blk)>;
//    using BlockVar = typename Block_t::template var_from<Tag>;
//    return static_cast<Block_t&&>(blk).template get<BlockVar>();
//  }
//
//  template <typename Tag, template <typename> class B,
//            typename CRT,                  // Cosserat Rod Traits
//            template <typename, typename>  // Skills
//            class... Components>
//  inline constexpr decltype(auto) get(
//      B<elastica::CosseratRodPlugin<CRT, B, Components...>> const&&
//          blk) noexcept {
//    using Block_t = cpp20::remove_cvref_t<decltype(blk)>;
//    using BlockVar = typename Block_t::template var_from<Tag>;
//    return static_cast<Block_t const&&>(blk).template get<BlockVar>();
//  }
//
//}  // namespace blocks

//  template <typename RodTraits, template <typename> class ShapePolicy,
//            template <typename> class GrowthPolicy>
//  class CosseratRod : public ShapePolicy<RodTraits>,
//  GrowthPolicy<RodTraits>
//  {
//    using MassStorage = typename RodTraits::NodeVector;
//
//   private:
//    MassStorage masses_;
//  };

//  template <typename ComputationalBlock, typename RodTraits,
//            template <typename> class ShapePolicy>

/*
 * Design:
 *
 * Ideally, we want that each and every mixin has access to the
 computational
 * block that has all the data members like so:
 *
 * <Traits, Block>
 * struct Geometyr, Elasticity etc.
 *
 * Then we mix them into RodPlugin (which acts like a wrapper with all the
 * mixed in classes)
 *
    template <typename CRT,                 // Cosserat Rod Traits
              typename ComputationalBlock,  // Block with implementation
              template <typename , typename >
    class... Components>  // Skills
    class CosseratRodPlugin : public
    : public Components<
          CRT, ComputationalBlock>...{};

    A nice feature of this desing is that each and every mixin has access
    to the block via CRTP and can access everything.

    However there is a problem with the block construction:

    First the default block template should be changed from

     // Models the block concept
      template <template <typename...> class Plugin>
      class Block;

    to

    template <template <typename Traits, typename Block ,
                  template <typename , typename >
                  class... Skills>
        class Plugin>
    class Block;

    Then during partial template specialization we have
    template <typename CRT,                  // Cosserat Rod Traits
              template <typename, typename>  // Skills
              class... Components>
    class Block<CosseratRodPlugin<CRT, Block<CosseratRodPlugin>,
                                  Components<CRT, Block<CosseratRodPlugin<
                                                  CRT,
 Block<CosseratRodPlugin>, Components>
                                                    >>...>>
        : public CosseratRodPlugin<CRT, Block<CosseratRodPlugin>,
                                   Components<CRT,
 Block<CosseratRodPlugin>>...>, private NonCopyable { It becomes unreadable
 and hard to parse whats exactly going on.


    An alternate approoch is to not template the mixins on the block, but
    to template them on the RodPlugin itself like so

 * <Traits, Plugin>
 * struct Geometyr, Elasticity etc.
 *
 * Then we can use them like so:
 *
    template <typename CRT,                 // Cosserat Rod Traits
              typename ComputationalBlock,  // Block with implementation
              template <typename , typename >
    class... Components>  // Skills
    class CosseratRodPlugin : public
    : public Components<
          CRT, CosseratRodPlugin<CRT, ComputaionalBlock,
 Components...>>...{};

    An obvious disadvantage is that for the mixins to have access to data
    members we now need to use CRTP twice at the call site, something like

    // First self makes the mixin a CosseratRodPlugin
    // Second self makes the mixin a ComputationalBlock
    // which then has access to rod properties
    self().self().rod_properties();

    As a benefit however the instantatino is easier and muhc more readable
    The code is much more readable
*/

/*
 * Design requirement:
 *
 * The layers should compulsarily have a:
 * - public "MemberTags" typedef
 * - public initialize function with specific signature
 */

// private:
//// friendships
////! Friend the plugin for testing
//// friend blocks::protocols::Plugin;
//// frienships
//
////   protected:
////    // Import staggering tags
////    using NodeTag = typename Traits::NodeTag;
////    using ElementTag = typename Traits::ElementTag;
////    using VoronoiTag = typename Traits::VoronoiTag;
////
////    // Import all tag lists
////    // Just transform and flatten here, although it seems weird to use
////    // a "transform"
////    template <typename ParentType>
////    struct AddMemberTags {
////      friend ParentType;
////      using type = typename ParentType::MemberTags;
////    };
////    //    using InternalMemberTags =
////    //        tmpl::flatten<tmpl::transform<Parents,
////    AddMemberTags<tmpl::_1>>>; using InternalMemberTags = tmpl::flatten<
////        tmpl::list<typename Components<CRT, This>::MemberTags...>>;
////    //    TypeDisplayer<InternalMemberTags> td{};
////    /* using SliceTags = typename Parent::SliceTags; */
////
////    template <typename T>
////    using slice_type_t = typename Traits::template slice_type_t<T>;
////
////    using NodeMemberTags =
////        tmpl::filter<InternalMemberTags,
////                     typename Traits::template
////                     IsNodeStaggered<tmpl::_1>>;
////    using ElementMemberTags =
////        tmpl::filter<InternalMemberTags,
////                     typename Traits::template
////                     IsElementStaggered<tmpl::_1>>;
////    using VoronoiMemberTags =
////        tmpl::filter<InternalMemberTags,
////                     typename Traits::template
////                     IsVoronoiStaggered<tmpl::_1>>;
////
////    using MemberTags =
////        tmpl::append<NodeMemberTags, ElementMemberTags,
////        VoronoiMemberTags>;
////    /* using SliceTags = typename Parent::SliceTags; */
////    using SliceTags =
////        tmpl::transform<MemberTags, tmpl::bind<slice_type_t, tmpl::_1>>;
