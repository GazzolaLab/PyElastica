#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "ErrorHandling/ExpectsAndEnsures.hpp"
/// Forward declarations
#include "Systems/CosseratRods/Types.hpp"
///
#include "Systems/Block/Block.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
#include "Utilities/End.hpp"  // from_end

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of a slice/view on a Block for Tagged Cosserat rods
   * \ingroup blocks
   *
   * \tparam CRT        The traits class for a Cosserat rod
   * \tparam Tag        The tag for the Tagged Cosserat plugin
   * \tparam Components Components customizing a Cosserat rod (such as geometry,
   *                    elasticity)
   *
   * \see Block
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class BlockView<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, BlockView, Tag, Components...>>
      : public BlockView<elastica::cosserat_rod::CosseratRodPlugin<
            CRT, BlockView, Components...>> {
    // only conceptually "new" constructor that we need to allow is conversion
    // of a normal block slice to a tagged block slice.
    // but even this conversion is facilitated by the inherited constructors.
    //
    // All other constructors should follow through, by virtue of slicing. The
    // "reference" stored is actually a reference to a non-tagged block, but we
    // can always convert it back to a tagged block if needed.
   private:
    //**Type definitions********************************************************
    //! Traits type
    using Traits = CRT;
    //! Type of the parent plugin
    using Parent =
        BlockView<elastica::cosserat_rod::CosseratRodPlugin<Traits, BlockView,
                                                            Components...>>;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{
    // constructor used in slice() methods
    using Parent::Parent;
    //@}
    //**************************************************************************
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of a const slice/view on a Block for Tagged Cosserat
   * rods
   * \ingroup blocks
   *
   * \tparam CRT        The traits class for a Cosserat rod
   * \tparam Tag        The tag for the Tagged Cosserat plugin
   * \tparam Components Components customizing a Cosserat rod (such as geometry,
   *                    elasticity)
   *
   * \see Block
   */
  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class ConstBlockView<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, ConstBlockView, Tag, Components...>>
      : public ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
            CRT, ConstBlockView, Components...>> {
   private:
    //**Type definitions********************************************************
    //! Traits type
    using Traits = CRT;
    //! Type of the parent plugin
    using Parent = ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
        Traits, ConstBlockView, Components...>>;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{
    using Parent::Parent;
    //@}
    //**************************************************************************
  };
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  FREE FUNCTIONS
  //
  //============================================================================

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
      BlockView<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, BlockView, Components...>> const& cosserat_rod) noexcept {
    // it might be not the best idea to have a size() member
    return cosserat_rod.size();
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
      ConstBlockView<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockView, Components...>> const& cosserat_rod) noexcept {
    // it might be not the best idea to have a size() member
    return cosserat_rod.size();
  }
  //****************************************************************************

  //**Ghosting functions********************************************************
  /*!\name Ghosting functions
   * \ingroup cosserat_rod */
  //@{
  namespace detail {

    // The last rod does not need it rightmost ghost filled, so only do till
    // region.size() - 1
    inline auto get_ghosts_size(Region const& region) noexcept -> std::size_t {
      return region.size - 1;
    }

  }  // namespace detail

  //****************************************************************************
  /*!\brief Fills ghosts for a variable of tag `Tag` in the current Block
   * Needed because some of the kernels use it
   */
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_for(
      BlockView<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, BlockView, Components...>>& block_like) noexcept {
    auto const& region = block_like.region();
    auto const gs = detail::get_ghosts_size(region);

    Expects((region.start + gs) <
            block_like.parent().get_ghost_node_buffer().size());
    fill_ghosts_for<Tag>(block_like.parent(), region.start, gs);
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
  inline void fill_ghosts(
      BlockView<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, BlockView, Components...>>& block_like) noexcept {
    auto const& region = block_like.region();
    auto const gs = detail::get_ghosts_size(region);

    Expects((region.start + gs) <
            block_like.parent().get_ghost_node_buffer().size());
    fill_ghosts(block_like.parent(), region.start, gs);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for a variable of tag `Tag` in the current Block
   * Needed because some of the kernels use it
   */
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_for(
      ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockView, Components...>>&) noexcept {
    // This should be a compile-time error, but we don't enforce it.
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
  inline void fill_ghosts(
      ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockView, Components...>>&) noexcept {
    // This should be a compile-time error, but we don't enforce it.
  }
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks
