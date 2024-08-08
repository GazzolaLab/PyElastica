#pragma once

//******************************************************************************
// Includes
//******************************************************************************

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
  class BlockSlice<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, BlockSlice, Tag, Components...>>
      : public BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
            CRT, BlockSlice, Components...>> {
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
        BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<Traits, BlockSlice,
                                                             Components...>>;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{
    BlockSlice() = delete;
    // constructor usde in slice() methods
    using Parent::Parent;
    // these are constructors used in emplace() of block
    BlockSlice(Parent const& other) : Parent(other){};
    BlockSlice(Parent&& other) noexcept : Parent(std::move(other)){};
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
  class ConstBlockSlice<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, ConstBlockSlice, Tag, Components...>>
      : public ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
            CRT, ConstBlockSlice, Components...>> {
   private:
    //**Type definitions********************************************************
    //! Traits type
    using Traits = CRT;
    //! Type of the parent plugin
    using Parent = ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
        Traits, ConstBlockSlice, Components...>>;
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
      BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, BlockSlice, Components...>> const& cosserat_rod) noexcept {
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
      ConstBlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockSlice, Components...>> const& cosserat_rod) noexcept {
    // it might be not the best idea to have a size() member
    return cosserat_rod.size();
  }
  //****************************************************************************

  //**Ghosting functions********************************************************
  /*!\name Ghosting functions
   * \ingroup cosserat_rod */
  //@{

  //****************************************************************************
  /*!\brief Fills ghosts for a variable of tag `Tag` in the current Block
   * Needed because some of the kernels use it
   */
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts_for(
      BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, BlockSlice, Components...>>&) noexcept {}
  //****************************************************************************

  //****************************************************************************
  /*!\brief Fills ghosts for all variables in the current Block
   *
   * \param block_like Block to fill ghosts
   */
  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  inline void fill_ghosts(BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
                              CRT, BlockSlice, Components...>>&) noexcept {}
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
      ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockSlice, Components...>>&) noexcept {}
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
      ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockSlice, Components...>>&) noexcept {}
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks
