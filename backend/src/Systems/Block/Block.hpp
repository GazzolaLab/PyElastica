#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Types.hpp"
//
#include "Systems/Block/Protocols.hpp"
#include "Systems/Block/TypeTraits.hpp"
// include implementations now
#include "Systems/Block/Block/Block.hpp"
#include "Systems/Block/Block/BlockFacade.hpp"
#include "Systems/Block/Block/BlockIterator.hpp"
#include "Systems/Block/Block/BlockSlice.hpp"
#include "Systems/Block/Block/BlockView.hpp"
#include "Systems/Block/Block/Metadata.hpp"
#include "Systems/Block/Block/VariableCache.hpp"
// Variable implementations
#include "BlockVariables/BlockInitializer.hpp"
#include "BlockVariables/BlockVariables.hpp"
//
#include "Utilities/End.hpp"  // from_end

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup blocks Blocks
 * \ingroup systems
 * \brief Efficient blockwise-programming of \elastica entities
 *
 * The blocks module contains the interface for efficiently programming
 * different \elastica entities (such as rods, rigid bodies etc.) and more
 * generally, Lagrangian data-structures.
 */
//******************************************************************************

//******************************************************************************
/*!\brief Block data-structures and routines
// \ingroup blocks
*/
namespace blocks {}
//******************************************************************************

namespace blocks {

  //****************************************************************************
  /*!\brief Get number of units from a Block
   * \ingroup blocks
   *
   * \details
   * Get number of units from a Block. By 'units' we mean the number of
   * individual BlockSlice(s) composing the Block.
   *
   * \usage
   * \code
   * Block<...> b;
   * std::size_t n_units = blocks::n_units(b);
   * \endcode
   *
   * \param block_like The block whose number of units is to be extracted.
   *
   * \note
   * These need to be customized for your block type.
   */
  template <typename Plugin>
  inline auto n_units(Block<Plugin> const& block) noexcept -> std::size_t;
  //****************************************************************************

  //**Iterator support**********************************************************
  /*!\name Block iterators */
  //@{

  //****************************************************************************
  /*!\brief Get a random access iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto begin(Block<Plugin>& block) noexcept -> BlockIterator<Plugin> {
    return {&block, 0UL};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto end(Block<Plugin>& block) noexcept -> BlockIterator<Plugin> {
    return {&block, n_units(block)};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cbegin(Block<Plugin> const& block) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&block, 0UL};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cend(Block<Plugin> const& block) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&block, n_units(block)};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto begin(Block<Plugin> const& block) noexcept {
    return cbegin(block);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto end(Block<Plugin> const& block) noexcept {
    return cend(block);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto begin(BlockView<Plugin>& block_view) noexcept
      -> BlockIterator<Plugin> {
    return {&block_view.parent(), block_view.region().start};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto end(BlockView<Plugin>& block_view) noexcept
      -> BlockIterator<Plugin> {
    return {&block_view.parent(),
            block_view.region().start + block_view.region().size};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cbegin(BlockView<Plugin> const& block_view) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&cpp17::as_const(block_view.parent()), block_view.region().start};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cend(BlockView<Plugin> const& block_view) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&cpp17::as_const(block_view.parent()),
            block_view.region().start + block_view.region().size};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto begin(BlockView<Plugin> const& block) noexcept {
    return cbegin(block);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access const iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto end(BlockView<Plugin> const& block) noexcept {
    return cend(block);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cbegin(ConstBlockView<Plugin> const& block_view) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&block_view.parent(), block_view.region().start};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto cend(ConstBlockView<Plugin> const& block_view) noexcept
      -> ConstBlockIterator<Plugin> {
    return {&block_view.parent(),
            block_view.region().start + block_view.region().size};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the start of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto begin(ConstBlockView<Plugin> const& block_view) noexcept {
    return cbegin(block_view);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get a random access iterator to the end of the block
   * \ingroup blocks
   */
  template <typename Plugin>
  inline auto end(ConstBlockView<Plugin> const& block_view) noexcept {
    return cend(block_view);
  }
  //****************************************************************************

  // @}
  //****************************************************************************

  //**Slice support*************************************************************
  /*!\name Whole block slice */
  //@{

  //****************************************************************************
  /*!\brief Slice operator for an entire block
   * \ingroup blocks
   *
   * \details
   * Gets a view into the entire block
   *
   * \param block_like Data-structure modeling the block concept
   *
   * \example
   * Given a block `b`, we can slice it using
   * \code
   * auto b_slice = block::slice(b); // Generates a view into the block
   * \endcode
   *
   * \note
   * not marked noexcept because we can have out of range indices
   */
  template <typename Plugin>
  inline constexpr decltype(auto) slice(Block<Plugin>& block_like) {
    return slice(block_like, 0UL, ::elastica::end);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline constexpr decltype(auto) slice(Block<Plugin> const& block_like) {
    return slice(block_like, 0UL, ::elastica::end);
  }
  //****************************************************************************

  // @}
  //****************************************************************************

  //****************************************************************************
  /*!\brief Adapt a function to be applied over slices instead of blocks
   * \ingroup blocks
   *
   * \details
   * Given a callable `func`, adapt it so that it can be applied on a
   * slice-wise basis rather than a blockwise basis. This proves useful for
   * iteration in the ::elastica::modules::Systems module of Simulator.
   *
   * \usage
   * \code
   * auto func = [](auto & system){ // User code to iterate on systems; }
   * // use to apply over slices
   * auto func_to_be_applied_over_slices =
   * blocks::for_each_slice_adapter(func);
   * \endcode
   *
   * \param func Callable to be applied over slices
   */
  template <typename F>
  constexpr auto for_each_slice_adapter(F func) noexcept {
    return [func = std::move(func)](auto& block) /*mutable*/ -> void {
      for (auto slice : block)
        func(slice);
    };
  }
  //****************************************************************************

}  // namespace blocks
