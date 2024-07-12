#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <vector>

/// Forward declarations
#include "Systems/CosseratRods/Types.hpp"
///

#include "ErrorHandling/Assert.hpp"
#include "Systems/Block/Block.hpp"
#include "Systems/CosseratRods/BlockInitializer.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
#include "Systems/common/SymplecticStepperAdapter.hpp"
///
#include "Utilities/NonCopyable.hpp"
#include "Utilities/Requires.hpp"
// #include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// serialization support, feels strange coming from simulator
// #include "Simulator/IO/Serialization/Serialize.hpp"

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of blocks::Block for CosseratRodPlugin
   * \ingroup blocks
   *
   * \tparam CRT        The traits class for a Cosserat rod
   * \tparam Components Components customizing a Cosserat rod (such as geometry,
   *                    elasticity)
   *
   * \see blocks::Block
   */
  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class Block<
      elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>
      // final (not final because testing requires deriving)
      : private elastica::NonCopyable,
        // public ::elastica::SymplecticPolicy<
        //     BlockFacade<elastica::cosserat_rod::CosseratRodPlugin<
        //        CRT, Block, Components...>>>
        public BlockFacade<elastica::cosserat_rod::CosseratRodPlugin<
            CRT, Block, Components...>> {
   private:
    //**Type definitions********************************************************
    //! Traits type
    using Traits = CRT;
    //! Facade type
    using Facade = BlockFacade<
        elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>;
    //! Parent type
    // using Parent = ::elastica::SymplecticPolicy<Facade>;
    using Parent = Facade;
    //! Plugin type
    using Plugin = typename Parent::PluginType;
    //! This type
    using This = Block<Plugin>;
    //! This slice type
    using ThisSlice =
        BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<CRT, BlockSlice,
                                                             Components...>>;
    //! Plugin type for the slice
    using SlicePlugin = typename ThisSlice::PluginType;

   public:
    //! List of variables
    using typename Parent::Variables;

   protected:
    //! List of variables placed on the whole rod
    using VariablesOnRod =
        tmpl::filter<Variables, typename Traits::template IsOnRod<tmpl::_1>>;
    //! List of variables placed on the nodes in the grid
    using VariablesOnNode =
        tmpl::filter<Variables, typename Traits::template IsOnNode<tmpl::_1>>;
    //! List of variables placed on the elements in the grid
    using VariablesOnElement =
        tmpl::filter<Variables,
                     typename Traits::template IsOnElement<tmpl::_1>>;
    //! List of variables placed on the voronois in the grid
    using VariablesOnVoronoi =
        tmpl::filter<Variables,
                     typename Traits::template IsOnVoronoi<tmpl::_1>>;
    //! Template type of expected initializer
    template <typename... Initializers>
    using CosseratInitializer =
        // note : here it is not templated on the slice of the plugin but rather
        // the plugin itself
        ::elastica::cosserat_rod::CosseratInitializer<Plugin, Initializers...>;
    //! Type of index
    using index_type = typename Traits::index_type;
    // Same as typename Traits::DataType::Index::type::ElementType;
    // but this one is tailored to blaze
    //! Type of size
    using size_type = typename Traits::size_type;
    //! Type of total element count
    using TotalElementCount = size_type;
    //! Type of ghost node indices
    using GhostNodeIndices = std::vector<index_type>;
    //**************************************************************************

   public:
    //**************************************************************************
    //! Tag to place on nodes
    using typename Plugin::OnNode;
    //! Tag to place on elements
    using typename Plugin::OnElement;
    //! Tag to place on voronois
    using typename Plugin::OnVoronoi;
    //! Tag to place on the whole rod rather than at discrete points
    using typename Plugin::OnRod;
    //**************************************************************************

   private:
    //**Friendships*************************************************************
    // Friend the Facade (for what?)
    friend Parent;
    // Friend the serialization struct
    // friend elastica::io::Serialize<This>;
    //**************************************************************************

    //**Parent methods**********************************************************
    //! Methods inherited from parent class
   public:
    using Plugin::n_elements;
    using Plugin::total_n_elements;
    using Plugin::total_n_nodes;
    using Plugin::total_n_voronois;
    // for n_units() free function
    using Parent::n_ghosts;
    using Parent::n_units;
    //**************************************************************************

    /* Not really needed?
    static constexpr std::size_t get_number_of_node_members() {
      return tmpl::size<NodeMemberTags>::value;
    }

    static constexpr std::size_t get_number_of_element_members() {
      return tmpl::size<ElementMemberTags>::value;
    }

    static constexpr std::size_t get_number_of_voronoi_members() {
      return tmpl::size<VoronoiMemberTags>::value;
    }
    */

    //**************************************************************************
    /*!\name Get variable helpers */
    //@{
    /*!\brief Helper to get the list of variables based on the stagger type
     */
   public:  // FIXME : Does not work since get_variables instantiates it
    static constexpr inline auto GetVariablesHelper(OnRod) noexcept ->
        typename This::VariablesOnRod;
    static constexpr inline auto GetVariablesHelper(OnNode) noexcept ->
        typename This::VariablesOnNode;
    static constexpr inline auto GetVariablesHelper(OnElement) noexcept ->
        typename This::VariablesOnElement;
    static constexpr inline auto GetVariablesHelper(OnVoronoi) noexcept ->
        typename This::VariablesOnVoronoi;
    //@}
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! Template type for getting the list of variables based on
    //! the stagger type
    template <typename OnStaggerType>
    using get_variables_placed =
        decltype(GetVariablesHelper(std::declval<OnStaggerType>()));
    //**************************************************************************

    //**************************************************************************
    /*!\name Start of slice */
    //@{
   private:
    //**************************************************************************
    template <typename T>
    using identity = tmpl::type_<T>;
    /* Dev note:
     * Need tag dispatch here as function specialization is not valid
     * (even with a single template parameter) because it is an
     * Explicit specialization in non-namespace scope
     * This was corrected in a later draft, so Clang++ works fine. However
     * GCC bugs out, even with the retroactive fix applied to C++14 compilers.
     * https://stackoverflow.com/questions/3052579/explicit-specialization-in-non-namespace-scope
     */
    template <typename OnStaggerType>
    inline auto get_start_of_slice_helper(
        index_type const unit, identity<OnStaggerType>) const noexcept
        -> size_type {
      // Zero-based indexing
      // this is implemented in check() now
      //      ELASTICA_ASSERT(ghost_node_indices_.size() >= unit,
      //                      "Wrong number of units requested");

      // unit - 1 cannot be negative
      return unit ? (ghost_node_indices_[unit - 1UL] + 1UL) : 0UL;
    }
    inline auto get_start_of_slice_helper(index_type const unit,
                                          identity<OnRod>) const noexcept
        -> size_type {
      return unit;
    }
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Gets starting location of new slice, based on the stagger type
     */
    template <typename OnStaggerType>
    inline auto get_start_of_slice(index_type const unit) const noexcept
        -> size_type {
      return get_start_of_slice_helper(unit, identity<OnStaggerType>{});
    }
    //@}
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets size of new slice, based on the stagger type
     */
    template <typename OnStaggerType>
    inline auto get_size_of_slice(
        index_type const index_to_be_sliced) const noexcept -> size_type {
      return OnStaggerType::get_dofs(n_elements(index_to_be_sliced));
    }
    //**************************************************************************
   private:
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets size of new slice, based on the stagger type
     * Non-rod specialization
     */
    template <typename OnStaggerType>
    inline auto get_size_of_slice_helper(
        index_type const start_index, size_type const slice_size,
        identity<OnStaggerType> /*meta*/) const noexcept -> size_type {
      return OnStaggerType::get_dofs(n_elements(start_index, slice_size));
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets size of new slice, based on the stagger type
     *
     * For types on rod the size of the slice is the size requested itself.
     * The get_dofs() is designed to always return 1, so this is a workaround.
     */
    inline auto get_size_of_slice_helper(
        index_type const, size_type const slice_size,
        identity<OnRod> /*meta*/) const noexcept -> size_type {
      return slice_size;
    }
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Gets size of new slice, based on the stagger type
     */
    template <typename OnStaggerType>
    inline auto get_size_of_slice(index_type const start_index,
                                  size_type const slice_size) const noexcept
        -> size_type {
      return get_size_of_slice_helper(start_index, slice_size,
                                      identity<OnStaggerType>{});
    }
    //**************************************************************************

   public:
    //**************************************************************************
    /*
     * \note
     * Most of the non-default public interface should more or less come from
     * the Cosserat Rod plugin class per design
     */
    //**************************************************************************

    //////////////////////////// start conformance /////////////////////////////
    // To work with block facade, implement customization point for generating
    // a slice.
    //**Slice functions*********************************************************
    /*!\name Slice functions */
    //@{

    //**************************************************************************
    /*!\brief Takes slice of each individual variable
     */
    template <typename Var>
    inline constexpr typename Var::slice_type slice(
        index_type index_to_be_sliced) & {
      return Var::slice(
          blocks::get<Var>(*this),
          get_start_of_slice<typename Var::Stagger>(index_to_be_sliced),
          get_size_of_slice<typename Var::Stagger>(index_to_be_sliced));
    }
    template <typename Var>
    inline constexpr typename Var::const_slice_type slice(
        index_type index_to_be_sliced) const& {
      return Var::slice(
          blocks::get<Var>(*this),
          get_start_of_slice<typename Var::Stagger>(index_to_be_sliced),
          get_size_of_slice<typename Var::Stagger>(index_to_be_sliced));
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Takes slice of each individual variable in a range
     */
    template <typename Var>
    inline constexpr typename Var::slice_type slice(index_type start_index,
                                                    size_type slice_size) & {
      return Var::slice(
          blocks::get<Var>(*this),
          get_start_of_slice<typename Var::Stagger>(start_index),
          get_size_of_slice<typename Var::Stagger>(start_index, slice_size));
    }
    template <typename Var>
    inline constexpr typename Var::const_slice_type slice(
        index_type start_index, size_type slice_size) const& {
      return Var::slice(
          blocks::get<Var>(*this),
          get_start_of_slice<typename Var::Stagger>(start_index),
          get_size_of_slice<typename Var::Stagger>(start_index, slice_size));
    }
    //**************************************************************************

    //@}
    //**************************************************************************
    ///////////////////////////// end conformance //////////////////////////////

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     */
    Block() : Parent(), ghost_node_indices_(){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other Other block to move from
     */
    Block(Block&& other) noexcept
        : Parent(std::move(other)),
          ghost_node_indices_(std::move(other.ghost_node_indices_)){};
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~Block() = default;
    //@}
    //**************************************************************************

   protected:
    //**Resizing functions******************************************************
    /*!\name Resizing functions */
    //@{

    //**************************************************************************
    /*!\brief Prepares for resizing the entire rod, with some bounds checking
     *
     * \param n_elems The number of elements in the newly introduced rod
     */
    auto prepare_for_resize(std::size_t n_elems) -> size_type {
      ELASTICA_ASSERT(n_elems > 1UL, "Cannot have rods with < 2 elements!");
      const size_type current_n_units = n_units();
      size_type total_n_elements_after_resizing(0UL);

      // A. increase units count taken care by the plugin
      //      element_counts_.push_back(n_elems);

      // A. resize dofs
      if (current_n_units) {
        // pre-existing block, so

        // 1. push back a new ghost node (based on zero-indexing)
        // i > 0
        // gn_indices[i]
        //   = gn_indices[i - 1] + (n_elements[i] + 1) + 1
        //   = gn_indices[i - 1] + n_nodes[i] + 1
        //   = gn_indices[i - 1] + 1 + n_nodes[i]
        //   = gn_indices[i - 2] + (1 + n_nodes[i-1]) + (1 + n_nodes[i])
        // i = 0
        // ghost_node_indices[0UL] = n_elements[0UL] + 1
        //                         = n_nodes[0]
        // meanwhile
        // total_n_nodes = n_nodes[0] + 1 + n_nodes[1] + 1 + .... + nodes[i - 1]
        // and hence they are equivalent
        ghost_node_indices_.push_back(total_n_nodes());

        // 2. calculate the new number of elements including ghosts
        total_n_elements_after_resizing =
            total_n_elements() + n_elems + n_ghosts(OnElement{});

      } else {
        // Fresh block, so
        ELASTICA_ASSERT(ghost_node_indices_.empty(),
                        "Block assert failed, contact developers");
        // cannot call total_n_elements() as we have an assert inside it
        // ghost_node_indices_.push_back(void) => No ghosts in this case

        // 1. Increment total number of elements => No ghosts in this case
        total_n_elements_after_resizing = n_elems;
      }
      return total_n_elements_after_resizing;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Resizes variables placed on entire rods
     *
     * \param total_number_of_units The total number of units/rods
     */
    void resize_members_on_rod(size_type const total_number_of_units) {
      tmpl::for_each<VariablesOnRod>(
          [this, new_size = total_number_of_units](auto v) {
            using Variable = tmpl::type_from<decltype(v)>;
            auto& var = blocks::get<Variable>(*this);
            Variable::resize(var, new_size);
          });
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Resizes variables placed `OnStaggerType`
     *
     * \param total_elements The total elements across rods
     *
     * \tparam OnStaggerType The staggering/placement type
     */
    template <typename OnStaggerType>
    void resize_members_placed(TotalElementCount const total_elements) {
      using VariablesPlacedOnStaggerType = get_variables_placed<OnStaggerType>;
      // new size does not depend on the variable, so we can factor it outside
      // the lambda
      tmpl::for_each<VariablesPlacedOnStaggerType>(
          [this, new_size = OnStaggerType::get_dofs(total_elements)](auto v) {
            using Variable = tmpl::type_from<decltype(v)>;
            auto& var = blocks::get<Variable>(*this);
            Variable::resize(var, new_size);
          });
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Resizes variables other than those placed on the rod
     *
     * \param total_elements The total elements across rods
     */
    void resize_members_not_on_rod(TotalElementCount const total_elements) {
      resize_members_placed<OnNode>(total_elements);
      resize_members_placed<OnElement>(total_elements);
      resize_members_placed<OnVoronoi>(total_elements);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Resizes all variables in the current Block
     *
     * \param total_elements        The total elements across all rods
     * \param total_number_of_units The total number of units/rods
     */
    void resize_internal(TotalElementCount const total_elements,
                         size_type const total_number_of_units) {
      // 2.1 resize dofs on rods by new_n_units
      resize_members_on_rod(total_number_of_units);

      // 2.2 resize dofs on nodes, elements and voronoi by total_n_elements_
      resize_members_not_on_rod(total_elements);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Gets the ghost nodes buffer
     *
     * \details
     * This is mostly used in contexts such as serialization
     */
    inline constexpr auto get_ghost_node_buffer() & noexcept
        -> GhostNodeIndices& {
      return ghost_node_indices_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets the ghost nodes buffer for a const Block
     *
     * \details
     * This is mostly used in contexts such as serialization
     */
    inline constexpr auto get_ghost_node_buffer() const& noexcept
        -> GhostNodeIndices const& {
      return ghost_node_indices_;
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   protected:
    //**Element buffer function*************************************************
    /*!\brief Initialize the element buffer at a given location
     *
     * \param n_elems Number of elements in the new rod
     * \param loc_n_units Fill location
     *
     * \note
     * This is mainly useful in testing
     */
    inline void initialize_elements_buffer(
        size_type const n_elems, size_type const loc_n_units) noexcept {
      n_elements(loc_n_units) = n_elems;
    }
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Emplaces a new rod to the block
     *
     * \details
     * This is the main entry-point for the Simulator to add new rods to the
     * current Block.
     *
     * \param block_initializer The initializer to fill in all elements of the
     * new block
     */
    template <typename... FirstBlockInitializers>
    auto emplace(
        CosseratInitializer<FirstBlockInitializers...>&& block_initializer)
        -> ThisSlice {
      using Init = CosseratInitializer<FirstBlockInitializers...>;
      // 0. number of preexisting rods
      const size_type curr_n_units = n_units();
      const size_type new_n_units = curr_n_units + 1UL;

      // 1. increase total number of elements and ghosts (block specific)
      const size_type n_elements_added(
          static_cast<Init&&>(block_initializer).n_elems());
      const size_type total_n_elements_after_resizing =
          prepare_for_resize(n_elements_added);

      // 2. Resize the block to fit the new rod
      // 2.1 resize dofs on nodes, elements and voronoi by total_n_elements_
      // 2.2 resize dofs on rods by new_n_units
      resize_internal(total_n_elements_after_resizing, new_n_units);

      // 3. The slicing logic (which calls check() to see if a buffer overflows)
      // requires that we know the number of elements that
      // needs to be sliced before making the slice itself, which is taken from
      // the elements buffer. So here we fill in the total_n_elements
      initialize_elements_buffer(n_elements_added, curr_n_units);

      // after resizing the units should match
      ELASTICA_ASSERT(n_units() == new_n_units,
                      "Invariant violation, contact the developers");

      // 4. fill ghosts
      fill_ghosts(*this);

      // after this step, we should have enough capacity for the n'th rod

      // 5. make a slice for the n'th unit (0 based indexing with checks)
      ThisSlice latest_slice(blocks::slice(*this, curr_n_units));

      // ... and initialize it ...
      Parent::initialize(latest_slice, std::move(block_initializer));

      // copied to the output buffer, NRVO should kick in
      return latest_slice;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Emplaces (back) a new rod to the block
     *
     * \param block_initializer The initializer to fill in all elements of the
     * new block
     *
     * \see emplace
     */
    template <typename... FirstBlockInitializers>
    inline decltype(auto) emplace_back(
        CosseratInitializer<FirstBlockInitializers...>&& block_initializer) {
      return emplace(std::move(block_initializer));
    }
    //**************************************************************************

   protected:
    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    /// Extra members for book-keeping
    //! Indices of nodal ghosts across
    GhostNodeIndices ghost_node_indices_;
    //@}
    //**************************************************************************
  };
  /*! \endcond */
  //****************************************************************************

  //**Ghosting functions********************************************************
  /*!\name Ghosting functions
   * \ingroup cosserat_rod */
  //@{

  //****************************************************************************
  /*!\brief Fills ghosts for a `Variable` in the current Block over a
   * certain region
   *
   * \tparam Variable
   * \param block_like Block to fill ghosts
   * \param region_start Start of region
   * \param region_size Size of region
   * \param deficit The number of ghosts to be filled, from the ghost node
   * index location.
   */
  template <typename Variable,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(Block<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, Block, Components...>>& block_like,
                       std::size_t region_start, std::size_t region_size,
                       std::size_t deficit) noexcept {
    // kernel
    auto& var = blocks::get<Variable>(block_like);
    const auto ghost_value = Variable::ghost_value();

    using signedt = std::make_signed_t<decltype(region_start)>;
    auto const it =
        std::cbegin(block_like.get_ghost_node_buffer()) + signedt(region_start);
    std::for_each(it, it + signedt(region_size),
                  [&](std::size_t const ghost_node_idx) {
                    const auto ghost_stop_idx = ghost_node_idx;
                    ELASTICA_ASSERT(ghost_stop_idx > deficit,
                                    "Ghost invariant violated!");
                    const auto ghost_start_idx = ghost_stop_idx - deficit;

                    for (auto ghost_idx = ghost_stop_idx;
                         ghost_idx > ghost_start_idx; --ghost_idx) {
                      Variable::slice(var, ghost_idx) = ghost_value;
                    }
                    // rewrite the loop above using positive indices
                    // for (auto ghost_idx = ghost_start_idx + 1;
                    //      ghost_idx <= ghost_stop_idx; ++ghost_idx) {
                    //   Variable::slice(var, ghost_idx) = ghost_value;
                    // }
                  });
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for a variable of tag `Tag` in the current Block
   *
   * \tparam Tag Tag of the variable
   * \param block_like Block to fill ghosts
   * \param region_start Start of region
   * \param region_size Size of region
   */
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(Block<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, Block, Components...>>& block_like,
                       std::size_t region_start,
                       std::size_t region_size) noexcept {
    using Variable = typename std::remove_reference_t<
        decltype(block_like)>::VariableMap::template variable_from<Tag>;
    const auto deficit = block_like.n_ghosts(typename Variable::Stagger{});
    fill_ghosts_for<Variable>(block_like, region_start, region_size, deficit);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for a variable of tag `Tag` in the current Block
   *
   * \tparam Tag Tag of the variable
   * \param block_like Block to fill ghosts
   */
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_for(
      Block<
          elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>&
          block_like) noexcept {
    fill_ghosts_for<Tag>(block_like, 0UL,
                         block_like.get_ghost_node_buffer().size());
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts on variables placed `OnStaggerType`
   *
   * \param block_like Block to fill ghosts
   * \param region_start Start of region
   * \param region_size Size of region
   * \param deficit The number of ghosts to be filled, from the ghost node
   * index location.
   *
   * \tparam OnStaggerType The staggering/placement type
   */
  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_placed(Block<elastica::cosserat_rod::CosseratRodPlugin<
                              CRT, Block, Components...>>& block_like,
                          std::size_t region_start, std::size_t region_size,
                          const std::size_t deficit) noexcept {
    using B = std::remove_reference_t<decltype(block_like)>;
    using VariablesPlacedOnStaggeredType =
        typename B::template get_variables_placed<OnStaggerType>;

    tmpl::for_each<VariablesPlacedOnStaggeredType>([=, &block_like](auto v) {
      using Variable = tmpl::type_from<decltype(v)>;
      fill_ghosts_for<Variable>(block_like, region_start, region_size, deficit);
    });
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts on variables placed `OnStaggerType`
   *
   * \param block_like Block to fill ghosts
   * \param region_start Start of region
   * \param region_size Size of region
   *
   * \tparam OnStaggerType The staggering/placement type
   */
  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_placed(
      Block<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, Block, Components...>>& block_like,
      std::size_t region_start, std::size_t region_size) noexcept {
    auto const deficit = block_like.n_ghosts(OnStaggerType{});
    fill_ghosts_placed<OnStaggerType>(block_like, region_start, region_size,
                                      deficit);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts on variables placed `OnStaggerType`
   *
   * \param block_like Block to fill ghosts
   *
   * \tparam OnStaggerType The staggering/placement type
   */
  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_placed(
      Block<
          elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>&
          block_like) noexcept {
    auto const gs = block_like.get_ghost_node_buffer().size();
    fill_ghosts_placed<OnStaggerType>(block_like, 0UL, gs);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for all variables in the current Block
   *
   * \param block_like Block to fill ghosts
   * \param region_start Start of region
   * \param region_size Size of region
   */
  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(Block<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, Block, Components...>>& block_like,
                   std::size_t region_start, std::size_t region_size) noexcept {
    using B = std::remove_reference_t<decltype(block_like)>;

    // fill ghosts for members not on rod, others don't need it
    fill_ghosts_placed<typename B::OnNode>(block_like, region_start,
                                           region_size);
    fill_ghosts_placed<typename B::OnElement>(block_like, region_start,
                                              region_size);
    fill_ghosts_placed<typename B::OnVoronoi>(block_like, region_start,
                                              region_size);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for all variables in the current Block
   *
   * \param block_like Block to fill ghosts
   */
  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(Block<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, Block, Components...>>& block_like) noexcept {
    using B = std::remove_reference_t<decltype(block_like)>;

    // fill ghosts for members not on rod, others don't need it
    fill_ghosts_placed<typename B::OnNode>(block_like);
    fill_ghosts_placed<typename B::OnElement>(block_like);
    fill_ghosts_placed<typename B::OnVoronoi>(block_like);
  }
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of blocks::Block for a TaggedCosseratRodPlugin
   * \ingroup blocks
   *
   * \tparam CRT        The traits class for a Tagged Cosserat plugin
   * \tparam Tag        The tag for the Tagged Cosserat plugin
   * \tparam Components Components customizing a Cosserat rod (such as geometry,
   *                    elasticity)
   *
   * \see blocks::Block
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class Block<elastica::cosserat_rod::TaggedCosseratRodPlugin<CRT, Block, Tag,
                                                              Components...>>
      : public Sliceable<Block<elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>,
        public Viewable<Block<elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>,
        public Block<elastica::cosserat_rod::CosseratRodPlugin<CRT, Block,
                                                               Components...>> {
   private:
    //**Type definitions********************************************************
    //! This type
    using This =
        Block<elastica::cosserat_rod::TaggedCosseratRodPlugin<CRT, Block, Tag,
                                                              Components...>>;
    //! This slice type
    using ThisSlice =
        BlockSlice<elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, BlockSlice, Tag, Components...>>;
    //! Parent type
    using Parent = Block<
        elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>;
    //! Sliceable type
    using SliceAffordance = Sliceable<This>;
    //! Viewable type
    using ViewAffordance = Viewable<This>;

   protected:
    //! Template type of expected initializer
    template <typename... Initializers>
    using CosseratInitializer =
        typename Parent::template CosseratInitializer<Initializers...>;
    //**************************************************************************

   public:
    //**Access operators********************************************************
    /*!\name Access operators */
    //@{
    //! Operator for slicing
    using SliceAffordance::operator[];
    //! Operator for viewing
    using ViewAffordance::operator[];
    //@}
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Emplaces a new rod to the block
     *
     * \details
     * This is the main entry-point for the Simulator to add new rods to the
     * current Block.
     *
     * \param block_initializer The initializer to fill in all elements of the
     * new block
     */
    template <typename... FirstBlockInitializers>
    auto emplace(
        CosseratInitializer<FirstBlockInitializers...>&& block_initializer)
        -> ThisSlice {
      // converts to a tagged slice
      return Parent::emplace(std::move(block_initializer));
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Emplaces (back) a new rod to the block
     *
     * \param block_initializer The initializer to fill in all elements of the
     * new block
     *
     * \see emplace
     */
    template <typename... FirstBlockInitializers>
    inline decltype(auto) emplace_back(
        CosseratInitializer<FirstBlockInitializers...>&& block_initializer) {
      return This::emplace(std::move(block_initializer));
    }
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace blocks

namespace blocks {

  //============================================================================
  //
  //  FREE FUNCTIONS
  //
  //============================================================================

  // These slice overloads are added to prevent template deduction failure from
  // Sliceable<Tagged> and Sliceable<Block> (which we derive from).

  //****************************************************************************
  /*!\brief Slices a tagged Cosserat Rod.
   * \ingroup cosserat_rod blocks
   *
   * \param cosserat_rod The current cosserat rod.
   * \param index The index of the slice (int or from_end)
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename Index>
  inline decltype(auto) slice(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>>& cosserat_rod,
      Index index) /*noexcept*/ {
    using Affordance =
        Sliceable<Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>;
    return slice(static_cast<Affordance&>(cosserat_rod), index);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Slices a tagged Cosserat Rod.
   * \ingroup cosserat_rod blocks
   *
   * \param cosserat_rod The current cosserat rod.
   * \param index The index of the slice (int or from_end)
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename Index>
  inline decltype(auto) slice(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>> const& cosserat_rod,
      Index index) /*noexcept*/ {
    using Affordance =
        Sliceable<Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>;
    return slice(static_cast<Affordance const&>(cosserat_rod), index);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Slices a tagged Cosserat Rod.
   * \ingroup cosserat_rod blocks
   *
   * \param cosserat_rod The current cosserat rod.
   * \param index The index of the slice (int or from_end)
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename StartIndex, typename StopIndex>
  inline decltype(auto) slice(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>>& cosserat_rod,
      StartIndex start_index, StopIndex stop_index) /*noexcept*/ {
    using Affordance =
        Viewable<Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>;
    return slice(static_cast<Affordance&>(cosserat_rod), std::move(start_index),
                 std::move(stop_index));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Slices a tagged Cosserat Rod.
   * \ingroup cosserat_rod blocks
   *
   * \param cosserat_rod The current cosserat rod.
   * \param index The index of the slice (int or from_end)
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components,
            typename StartIndex, typename StopIndex>
  inline decltype(auto) slice(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>> const& cosserat_rod,
      StartIndex start_index, StopIndex stop_index) /*noexcept*/ {
    using Affordance =
        Viewable<Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
            CRT, Block, Tag, Components...>>>;
    return slice(static_cast<Affordance const&>(cosserat_rod),
                 std::move(start_index), std::move(stop_index));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the size of the Cosserat Rod.
   * \ingroup cosserat_rod blocks
   *
   * \param cosserat_rod The current cosserat rod.
   * \return The  number of elements in the cosserat rod.
   */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline constexpr auto size(
      Block<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, Block, Components...>> const& cosserat_rod) noexcept {
    // it might be not the best idea to have a size() member in the first place
    return cosserat_rod.size();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get number of units within a CosseratRod
   * \ingroup cosserat_rod blocks
   *
   * \details
   * Get number of units from a Block. By 'units' we mean the number of
   * individual BlockSlice(s) composing the Block.
   *
   * \usage
   * \code
     Block<...> b;
     std::size_t n_units = blocks::n_units(b);
   * \endcode
   *
   * \param cosserat_rod The cosserat rod whose n_units is to be extracted
   */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline auto n_units(
      Block<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, Block, Components...>> const& cosserat_rod) noexcept
      -> std::size_t {
    return cosserat_rod.n_units();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the size of the Cosserat Rod.
   * \ingroup cosserat_rod
   *
   * \param cosserat_rod The current cosserat rod.
   * \return The number of elements in the cosserat rod.
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline constexpr auto size(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>> const& cosserat_rod) noexcept {
    // it might be not the best idea to have a size() member in the first place
    return cosserat_rod.size();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get number of units within a CosseratRod
   * \ingroup cosserat_rod blocks
   *
   * \details
   * Get number of units from a Block. By 'units' we mean the number of
   * individual BlockSlice(s) composing the Block.
   *
   * \usage
   * \code
     Block<...> b;
     std::size_t n_units = blocks::n_units(b);
   * \endcode
   *
   * \param cosserat_rod The cosserat rod whose n_units is to be extracted
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline auto n_units(
      Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
          CRT, Block, Tag, Components...>> const& cosserat_rod) noexcept
      -> std::size_t {
    return cosserat_rod.n_units();
  }
  //****************************************************************************

}  // namespace blocks

//    template <typename CRT,                 // Cosserat Rod Traits
//              template <typename> ComputationalBlock,  // Block with
//              implementation
//            template <typename /*CRT*/, typename /* CosseratRodPlugin */>
//            class... Components>  // Skills
//  class CosseratRodPlugin;

// template <typename> Block

//  template <template <typename /*Traits*/, typename /* Block */,
//                      template <typename /*Traits*/, typename /* Block */>
//                      class... /* Skills*/>
//            class Plugin>
//  class Block;

// Specialization of blocks for the cosserat rods class
// TODO : Blocking strategy :
//  1. Default : free blocks
//  2. fixed size blocks (compile time-parameter)

//  template <template <typename, typename...> class TT, typename SENDER,
//            typename... ARGS>
//  class CRTPBase<TT<SENDER, ARGS...>> {
//    // ...
//  };

// Specialization for a Cosserat Rod plugin
//  template <template <typename /*Traits*/, typename /* Block */,
//                      template <typename /*Traits*/, typename /* Block */>
//                      class... /* Skills*/>
//            class CRPlugin,
//            typename CRT, template <typename, typename> class... Skills>
//  class Block<CRPlugin<CRT, Block>>;

/*
template <typename StaggeredTag>
constexpr auto make_slices_of_members_marked_with(const size_type unit) {
  // Always one behind is where the ghost starts according to the indexing
  const size_type start_slice = get_start_of_slice<StaggeredTag>(unit);
  const size_type size_slice = get_size_of_slice<StaggeredTag>(unit);
  // StaggeredTag::get_dofs(elements_.back());

  using StaggeredMemberTags = get_members_marked_with<StaggeredTag>;

  auto applier = [this, start_slice, size_slice](auto Is) {
    using StaggeredMemberTag = tmpl::at_c<StaggeredMemberTags, Is>;
    using StaggeredMemberSliceTag = slice_type_t<StaggeredMemberTag>;

    // Unsliced
    auto& full = std::get<StaggeredMemberTag>(rod_properties_);
    // Sliced for initialization : from ghost_idx + 1 to last
    // element
    return StaggeredMemberSliceTag{
        StaggeredMemberTag::slice(full, start_slice, size_slice)};
  };

  return index_apply<tmpl::size<StaggeredMemberTags>::value>(
      [&applier](auto... Is) { return std::make_tuple(applier(Is)...); });
}
*/

//    void resize(NodeTag /* meta */) {
//      tmpl::for_each<NodeMemberTags>(
//          [this, resized = n_nodes()](auto node_member_tag) {
//            using NodeMemberTag =
//            tmpl::type_from<decltype(node_member_tag)>; auto& node_member
//            = std::get<NodeMemberTag>(rod_properties_);
//            // Resize dofs irrespective of the type
//            NodeMemberTag::resize(node_member, resized);
//          });
//    }
//
//    void resize(ElementTag /* meta */) {
//      tmpl::for_each<ElementMemberTags>([this, resized = n_elements()](
//                                            auto element_member_tag) {
//        using ElementMemberTag =
//        tmpl::type_from<decltype(element_member_tag)>; auto&
//        element_member = std::get<ElementMemberTag>(rod_properties_);
//        // Resize dofs irrespective of the type
//        ElementMemberTag::resize(element_member, resized);
//      });
//    }
//
//    void resize(VoronoiTag /* meta */) {
//      tmpl::for_each<VoronoiMemberTags>([this, resized = n_voronois()](
//                                            auto voronoi_member_tag) {
//        using VoronoiMemberTag =
//        tmpl::type_from<decltype(voronoi_member_tag)>; auto&
//        voronoi_member = std::get<VoronoiMemberTag>(rod_properties_);
//        // Resize dofs irrespective of the type
//        VoronoiMemberTag::resize(voronoi_member, resized);
//      });
//    }

/*
Slices make_slices_deprecated() {
  // IMPOSSIBLE : Submatrices and subvectors disallow default
  // construction, hence do in a round about way
  Slices slices{};
  make_slices_of_members_marked_with_deprecated<NodeTag>(slices);
  make_slices_of_members_marked_with_deprecated<ElementTag>(slices);
  make_slices_of_members_marked_with_deprecated<VoronoiTag>(slices);
  return slices;
}
 */

/* this now comes from the blocks library
auto make_slices(size_type const unit) {
  ELASTICA_ASSERT(not element_counts_.empty(),
                  "No elements to make slices for in the Block!");
  //
TypeDisplayer<decltype(make_slices_of_members_marked_with<ElementTag>(unit))>
  //      td{};
  return std::tuple_cat(
      make_slices_of_members_marked_with<NodeTag>(unit),
      make_slices_of_members_marked_with<ElementTag>(unit),
      make_slices_of_members_marked_with<VoronoiTag>(unit));
}
*/

/*
// TODO : Check if all are valid blocks
// The implementation is variadic because resizing every time is inefficient
// rather we can add all dofs in one go. Of course, by putting in one block
// initializer, we get the regular implementation
template <typename... FirstBlockInitializers, typename... BlockInitializers,
          Requires<cpp17::conjunction_v<
              detail_cosserat_rod::tt::IsBlockInitializer<
                  CRT, BlockInitializers>...>> = nullptr>
auto emplace_back(CosseratInitializer<FirstBlockInitializers...>&&
                      first_block_initializer,
                  BlockInitializers&&... block_initializers) {
  // 0. number of preexisting rods
  const auto n_units = size();
  constexpr auto n_units_added = 1 + sizeof...(BlockInitializers);

  // 1. Increase units count, linear
  prepare_for_resize(first_block_initializer.n_elems);
  EXPAND_PACK_LEFT_TO_RIGHT(prepare_for_resize(block_initializers.n_elems));

  // 2. Bulk resize
  resize();

  // 3. Bulk fill all ghosts
  initialize_ghosts();

  auto initializer = [n_units, this,
                      cap = std::forward_as_tuple(first_block_initializer,
                                                  block_initializers...)](
                         auto Is) {
    auto&& block_initializer(std::get<Is>(cap));

    const size_type unit = n_units + Is;

    // 4. Make slices for each rod
    CosseratSlice slices(make_slices(unit));

    ELASTICA_ASSERT(element_counts_[unit] == (block_initializer.n_elems),
                    "Invariant violation detected, contact the developers");

    // Parent::print(std::forward<decltype(block_initializer.initializers)>(
    //                block_initializer.initializers));

    // ... and initialize them ...
    Parent::initialize(slices, std::move(block_initializer));

    // TODO : Returning a copy, is it okay?
    return slices;
  };

  // by going through the rods that are just added
  return index_apply<n_units_added>([&initializer](auto... Is) {
    return std::make_tuple(initializer(Is)...);
  });
}

// TODO : Check for initializer requirements here
template <typename... UnitInitializers>
auto emplace_back(std::size_t n_elems,
                  UnitInitializers&&... unit_initializers) {
  // A.
  prepare_for_resize(n_elems);

  // B. Resize memory for all the member types based on new
  total_n_elements_ resize();

  // C. Initialize ghosts based on new total_n_elements_
  // no-op if ghost_node_indices has no elements
  initialize_ghosts();

  // D. Initialize new dofs for the new cosserat rod (the one just added)
  auto slices(make_slices_v2(size()));

  Parent::initialize(
      n_elems, slices,
      std::forward<std::tuple<InitializerPack...>>(initializer_pack));

  // At this point the total number of elements and the accumulate
  // of individual elements should match
  // For this case, directly return the view/slice rather than a tuple
  // of them
  return std::get<0UL>(emplace_back(make_cosserat_rod(
      n_elems, std::forward<UnitInitializers>(unit_initializers)...)));
}
*/
