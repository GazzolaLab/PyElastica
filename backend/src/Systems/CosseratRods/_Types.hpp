#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>

#include "Systems/Block/Types.hpp"

namespace elastica {

  namespace cosserat_rod {

    ////////////////////////////////////////////////////////////////////////////
    //
    // Forward declarations of cosserat rod customization types
    //
    ////////////////////////////////////////////////////////////////////////////

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename /*DataType*/, typename /*PlacementTraits*/,
              typename /*AllocatorTraits*/>
    struct CosseratRodTraits;

    struct DefaultCosseratRodTraits;

    template <typename /*CRT*/,
              template <typename /*Plugin*/> class /*ComputationalBlock*/,
              template <typename /*CRT*/, typename /*Instantiated Block*/>
              class... /*Components*/>
    class CosseratRodPlugin;

    template <typename /*CRT*/,
              template <typename /*Plugin*/> class /*ComputationalBlock*/,
              typename /*Tag*/,
              template <typename /*CRT*/, typename /*Instantiated Block*/>
              class... /*Components*/>
    class TaggedCosseratRodPlugin;

    template <class /*Plugin*/, class... /*Initializers*/>
    struct CosseratInitializer;
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

  namespace cosserat_rod {

    ////////////////////////////////////////////////////////////////////////////
    //
    // Forward declarations of rod types
    //
    ////////////////////////////////////////////////////////////////////////////

  }  // namespace cosserat_rod

}  // namespace elastica

namespace blocks {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class Block<
      elastica::cosserat_rod::CosseratRodPlugin<CRT, Block, Components...>>;

  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class BlockSlice<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, BlockSlice, Tag, Components...>>;

  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedConstBlock*/>
            class... Components>
  class ConstBlockSlice<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, ConstBlockSlice, Tag, Components...>>;

  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  class BlockView<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, BlockView, Tag, Components...>>;

  template <typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedConstBlock*/>
            class... Components>
  class ConstBlockView<elastica::cosserat_rod::TaggedCosseratRodPlugin<
      CRT, ConstBlockView, Tag, Components...>>;

  /*! \endcond */
  //****************************************************************************

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations of functions
  //
  //////////////////////////////////////////////////////////////////////////////

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename Variable,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(Block<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, Block, Components...>>& block_like,
                       std::size_t region_start, std::size_t region_size,
                       std::size_t deficit) noexcept;
  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(Block<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, Block, Components...>>& block_like,
                       std::size_t region_start,
                       std::size_t region_size) noexcept;

  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(Block<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, Block, Components...>>& block_like) noexcept;

  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_placed(Block<elastica::cosserat_rod::CosseratRodPlugin<
                              CRT, Block, Components...>>& block_like,
                          std::size_t region_start, std::size_t region_size,
                          const std::size_t deficit) noexcept;

  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_placed(Block<elastica::cosserat_rod::CosseratRodPlugin<
                              CRT, Block, Components...>>& block_like,
                          std::size_t region_start,
                          std::size_t region_size) noexcept;

  template <typename OnStaggerType,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_placed(Block<elastica::cosserat_rod::CosseratRodPlugin<
                              CRT, Block, Components...>>& block_like) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(Block<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, Block, Components...>>& block_like,
                   std::size_t region_start, std::size_t region_size) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(Block<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, Block, Components...>>& block_like) noexcept;

  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, BlockSlice, Components...>>&) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(BlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, BlockSlice, Components...>>&) noexcept;

  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(
      ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ConstBlockSlice, Components...>>&) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(ConstBlockSlice<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, ConstBlockSlice, Components...>>&) noexcept;

  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(BlockView<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, BlockView, Components...>>&) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(BlockView<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, BlockView, Components...>>&) noexcept;

  template <typename Tag,
            typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts_for(ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
                           CRT, ConstBlockView, Components...>>&) noexcept;

  template <typename CRT,  // Cosserat Rod Traits
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void fill_ghosts(ConstBlockView<elastica::cosserat_rod::CosseratRodPlugin<
                       CRT, ConstBlockView, Components...>>&) noexcept;
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks
