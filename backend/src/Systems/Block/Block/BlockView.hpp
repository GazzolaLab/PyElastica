#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block/Block/Types.hpp"
//
#include "Systems/Block/Block/Concepts.hpp"
//
#include "Systems/Block/Block/AsVariables.hpp"
#include "Systems/Block/Block/BlockViewFacade.hpp"
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/TypeTraits.hpp"
//
#include "Utilities/AsConst.hpp"
#include "Utilities/End.hpp"
#include "Utilities/TMPL.hpp"
//
#include <cassert>
#include <utility>

namespace blocks {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <class Plugin>
  inline constexpr auto operator==(BlockView<Plugin> const& lhs,
                                   BlockView<Plugin> const& rhs) noexcept
      -> bool;
  template <class Plugin>
  inline constexpr auto operator==(ConstBlockView<Plugin> const& lhs,
                                   ConstBlockView<Plugin> const& rhs) noexcept
      -> bool;
  // template <class Plugin>
  // inline auto size(BlockView<Plugin> const& view) -> bool;
  // template <class Plugin>
  // inline auto size(ConstBlockView<Plugin> const& view) -> bool;
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Implementation of a view/view on an \elastica Block
   * \ingroup blocks
   *
   * \details
   * The BlockView template class provides a view (or) view into a Block of
   * `data` and `operations`. It also models the `ComputationalBlock` concept
   * and hence can be used across places where a \elastica Block is used as
   * a template parameter.
   *
   * Similar to Block, BlockView can also be customized (specialized) for any
   * Lagrangian entity, but experience suggests that extensive customization is
   * not needed. For a concrete example, please see specializations of
   * BlockView in @ref cosserat_rod.
   *
   * BlockView is different from a BlockSlice : a BlockSlice represents a single
   * physical entity, meanwhile BlockView may represent a group of entities.
   *
   * \usage
   * The intended usage of a BlockView is when a view is required into the data
   * held by a block---this is frequently the case when a user either
   * 1. adds an entity into the Simulator, or
   * 2. requires access (for e.g. reading/writing to disk) to only a portion of
   * the block.
   * The pattern that is most commonly seen in the use case (1) is for a
   * BlockView templated on some `Plugin` type to be only used with a Block
   * of the same `Plugin` type, when adding new entities to the Block.
   * For use case (2) we suggest the user to the blocks::slice() function, which
   * has an intuitive, explicit view syntax, or even the subscript operator[].
   * We note that we might explicitly disable the subscript operator [] for
   * slicing in the future, as the semantics are potentially unclear.
   *
   * Finally, with no additional coding effort, the BlockView has exactly the
   * same operations as the mother Block (aka Block of the same `Plugin` type),
   * but now it operates only on that view of the data. This means that a
   * BlockView can be used as a small Block in itself which greatly simplifies
   * interfacing different components of \elastica---the user need not care or
   * even know about whether the data that she has is a Block or a BlockView!
   *
   * \tparam Plugin The computational plugin modeling a Lagrangian entity
   *
   * \see Block, blocks::slice
   */
  template <class Plugin>
  class BlockView : public detail::BlockViewFacade<Plugin>,
                    public Gettable<BlockView<Plugin>>,
                    public Spannable<BlockView<Plugin>>,
                    public Plugin {
    // NOTE : Plugin is inherited after Facade so that indices and references to
    // the parent block is filled first. This is because some elements of Plugin
    // may require that the slice (and index) be valid, for example to generate
    // internal refs/pointers for time-stepping.
   protected:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using Parent = detail::BlockViewFacade<Plugin>;
    //! Type of the view
    using This = BlockView<Plugin>;
    //! Type of gettable
    using GetAffordance = Gettable<This>;
    //! Type of sliceable
    using SpanAffordance = Spannable<This>;
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using typename Parent::PluginType;
    //! Type of the parent block
    using typename Parent::ParentBlock;
    //! Type of Variables
    using typename Parent::Variables;
    //**************************************************************************

    //**Friendships*************************************************************
    //! Friend the main block
    friend ParentBlock;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     * \param parent_block The parent block that
     * \param start_index The start index
     * \param region_size The stop index
     */
    explicit BlockView(ParentBlock& parent_block, std::size_t start_index,
                       std::size_t region_size) noexcept
        : Parent(parent_block, start_index, region_size),
          GetAffordance(),
          SpanAffordance(),
          PluginType() {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief The copy constructor.
     *
     * \param other view to copy
     */
    BlockView(BlockView const& other)
        : Parent(static_cast<Parent const&>(other)),
          GetAffordance(),
          SpanAffordance(),
          PluginType(static_cast<PluginType const&>(other)){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other view to move from
     */
    BlockView(BlockView&& other) noexcept
        : Parent(static_cast<Parent&&>(other)),
          GetAffordance(),
          SpanAffordance(),
          PluginType(static_cast<PluginType&&>(other)){};
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~BlockView() = default;
    //@}
    //**************************************************************************

   public:
    //**Access operators********************************************************
    /*!\name Access operators */
    //@{
    //! Operator for slicing and viewing
    using SpanAffordance::operator[];
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Implementation of a view/view on a const \elastica Block
   * \ingroup blocks
   *
   * \details
   * The ConstBlockView template class provides a view (or) view into a
   * constant Block of \a data and \a operations. It also models the
   * `ComputationalBlock` concept and hence can be used across places where a
   * \elastica Block is used as a template parameter.
   *
   * Notably, it differs from BlockView in one key aspect : the underlying
   * data/view is always constant and cannot be modified (it is only read only).
   * Hence ConstBlock is useful for propagating const-correctness throughout the
   * code. It can be used in places where one needs to pass (const)-data to the
   * user which she can then copy and use it for her own purposes.
   *
   * \tparam Plugin The computational plugin modeling a Lagrangian entity
   *
   * \see BlockView
   */
  template <class Plugin>
  class ConstBlockView : public detail::ConstBlockViewFacade<Plugin>,
                         public Gettable<ConstBlockView<Plugin>>,
                         public Spannable<ConstBlockView<Plugin>>,
                         public Plugin {
    // NOTE : Plugin is inherited after Facade so that indices and references to
    // the parent block is filled first. This is because some elements of Plugin
    // may require that the slice (and index) be valid, for example to generate
    // internal refs/pointers for time-stepping.
   protected:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using Parent = detail::ConstBlockViewFacade<Plugin>;
    //! Type of the view
    using This = ConstBlockView<Plugin>;
    //! Type of gettable
    using GetAffordance = Gettable<This>;
    //! Type of sliceable
    using SpanAffordance = Spannable<This>;
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using typename Parent::PluginType;
    //! Type of the parent block
    using typename Parent::ParentBlock;
    //! Type of Variables
    using typename Parent::Variables;
    //**************************************************************************

    //**Friendships*************************************************************
    //! Friend the main block
    friend ParentBlock;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     * \param parent_block The parent block that
     * \param start_index The start index
     * \param region_size The stop index
     */
    explicit ConstBlockView(ParentBlock const& parent_block,
                            std::size_t start_index,
                            std::size_t region_size) noexcept
        : Parent(parent_block, start_index, region_size),
          GetAffordance(),
          SpanAffordance(),
          PluginType() {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief The copy constructor.
     *
     * \param other view to copy
     */
    ConstBlockView(ConstBlockView const& other)
        : Parent(static_cast<Parent const&>(other)),
          GetAffordance(),
          SpanAffordance(),
          PluginType(static_cast<PluginType const&>(other)){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other view to move from
     */
    ConstBlockView(ConstBlockView&& other) noexcept
        : Parent(static_cast<Parent&&>(other)),
          GetAffordance(),
          SpanAffordance(),
          PluginType(static_cast<PluginType&&>(other)){};
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~ConstBlockView() = default;
    //@}
    //**************************************************************************

   public:
    //**Access operators********************************************************
    /*!\name Access operators */
    //@{
    //! Operator for slicing and viewing
    using SpanAffordance::operator[];
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //============================================================================
  //
  //  GLOBAL OPERATORS
  //
  //============================================================================

  //**Equality operator*********************************************************
  /*!\brief Equality comparison between two BlockView objects.
   *
   * \param lhs The left-hand side view.
   * \param rhs The right-hand side view.
   * \return \a true if the views are same, else \a false
   */
  template <class Plugin>
  inline constexpr auto operator==(BlockView<Plugin> const& lhs,
                                   BlockView<Plugin> const& rhs) noexcept
      -> bool {
    return static_cast<detail::BlockViewFacade<Plugin> const&>(lhs) ==
           static_cast<detail::BlockViewFacade<Plugin> const&>(rhs);
  }
  //****************************************************************************

  //**Equality operator*********************************************************
  /*!\brief Equality comparison between two ConstBlockView objects.
   *
   * \param lhs The left-hand side const view.
   * \param rhs The right-hand side const view.
   * \return \a true if the const views are same, else \a false
   */
  template <class Plugin>
  inline constexpr auto operator==(ConstBlockView<Plugin> const& lhs,
                                   ConstBlockView<Plugin> const& rhs) noexcept
      -> bool {
    return static_cast<detail::ConstBlockViewFacade<Plugin> const&>(lhs) ==
           static_cast<detail::ConstBlockViewFacade<Plugin> const&>(rhs);
  }
  //****************************************************************************

  //============================================================================
  //
  //  FREE FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Get number of units from a BlockView
   * \ingroup blocks
   *
   * \details
   * Get number of units from a BlockView. By 'units' we mean the number of
   * individual BlockSlice(s) composing the View.
   *
   * \usage
   * \code
   * BlockView<...> b;
   * std::size_t n_units = blocks::n_units(b);
   * \endcode
   *
   * \param block_like The block whose number of units is to be extracted.
   *
   * \note
   * These can be customized for your block type.
   */
  template <typename Plugin>
  inline auto n_units(BlockView<Plugin> const& block_view) noexcept
      -> std::size_t {
    return block_view.region().size;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Get number of units from a ConstBlockView
   * \ingroup blocks
   *
   * \details
   * Get number of units from a ConstBlockView. By 'units' we mean the number of
   * individual BlockSlice(s) composing the View.
   *
   * \usage
   * \code
   * ConstBlockView<...> b;
   * std::size_t n_units = blocks::n_units(b);
   * \endcode
   *
   * \param block_like The block whose number of units is to be extracted.
   *
   * \note
   * These can be customized for your block type.
   */
  template <typename Plugin>
  inline auto n_units(ConstBlockView<Plugin> const& block_view) noexcept
      -> std::size_t {
    return block_view.region().size;
  }
  //****************************************************************************

  //============================================================================
  //
  //  GET FUNCTIONS
  //
  //============================================================================

  namespace detail {

    template <typename BlockVariableTag, typename ViewLike>
    inline constexpr decltype(auto) get_view(ViewLike&& view_like) noexcept {
      return std::forward<ViewLike>(view_like)
          .parent()
          .template slice<BlockVariableTag>(
              std::forward<ViewLike>(view_like).region().start,
              std::forward<ViewLike>(view_like).region().size);
    }

  }  // namespace detail

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name Get functions for Slice types */
  //@{

  //****************************************************************************
  /*!\brief Extract element from a BlockView
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the BlockView `block_view` whose tag type is `Tag`.
   * Fails to compile unless the block has the `Tag` being extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     Block<...> b;
     auto my_tag_data = blocks::get<tags::MyTag>(b);
   * \endcode
   *
   * \tparam Tag Tag to extract
   *
   * \param block_like The block to extract the tag from
   */
  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockView<Plugin>& block_view) noexcept {
    return detail::get_view<BlockVariableTag>(block_view);
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockView<Plugin> const& block_view) noexcept {
    return detail::get_view<BlockVariableTag>(block_view);
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockView<Plugin>&& block_view) noexcept {
    return detail::get_view<BlockVariableTag>(
        static_cast<BlockView<Plugin>&&>(block_view));
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockView<Plugin> const&& block_view) noexcept {
    return detail::get_view<BlockVariableTag>(
        static_cast<BlockView<Plugin> const&&>(block_view));
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      ConstBlockView<Plugin> const& block_view) noexcept {
    return detail::get_view<BlockVariableTag>(block_view);
  }
  //@}
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  VIEW FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name Block view functions */
  //@{

  //****************************************************************************
  /*!\brief View into blocks
   * \ingroup blocks
   *
   * \details
   * Implements contiguous views on Blocks
   *
   * \param block_like Data-structure modeling the block concept
   * \param start_index The start index of the slice (int or from_end)
   * \param stop_index The stop index of the slice (int or from_end)
   *
   * \example
   * Given a block or a view `b`, we can obtain a view into it by using
   * \code
   * auto b_view = block::slice(b, 5UL, 10UL); // gets b[5, 6, 7, 8, 9]
   * // Gets b[end - 5, end - 4, end - 3]
   * auto b_view_from_end = block::slice(b, elastica::end - 5UL, elastica::end -
   * 2UL);
   * \endcode
   */
  template <typename Plugin>
  inline decltype(auto) slice_backend(Block<Plugin>& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    using ReturnType =
        typename PluginFrom<Block<Plugin>>::template to<BlockView>::type;
    return ReturnType{block_like, start_index, region_size};
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(Block<Plugin> const& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    using ReturnType = const typename PluginFrom<Block<Plugin>>::template to<
        ConstBlockView>::type;
    return ReturnType{block_like, start_index, region_size};
  }
  //****************************************************************************

  //@}
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name BlockView view functions */
  //@{

  //****************************************************************************
  /*!\brief View into views
   * \ingroup blocks
   *
   * \details
   * Implements contiguous views on Views
   *
   * \param block_like Data-structure modeling the block concept
   * \param start_index The start index of the slice
   * \param region_size The size of the slice
   *
   * \example
   * Given a block or a view `b`, we can obtain a view into it by using
   * \code
   * auto b_view = block::slice(b, 5UL, 10UL); // gets b[5, 6, 7, 8, 9]
   * // Gets b[end - 5, end - 4, end - 3]
   * auto b_view_from_end = block::slice(b, elastica::end - 5UL, elastica::end -
   * 2UL);
   * \endcode
   *
   */

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin>& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + start_index, region_size);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin> const& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + start_index, region_size);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin>&& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(
        static_cast<BlockView<Plugin>&&>(block_like).parent(),
        static_cast<BlockView<Plugin>&&>(block_like).region().start +
            start_index,
        region_size);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(ConstBlockView<Plugin> const& block_like,
                                      std::size_t start_index,
                                      std::size_t region_size) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + start_index, region_size);
  }
  //****************************************************************************

  //@}
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  SLICE FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name BlockView slice functions */
  //@{

  //****************************************************************************
  /*!\brief Slice into views
   * \ingroup block_customization
   *
   * \details
   * Implements slices on entities modeling block concept.
   *
   * \param block_like Data-structure modeling the block concept
   * \param index The index of the slice
   *
   * \example
   * Given a view `b`, we can obtain a slice
   * \code
   * auto b_slice = block::slice(b, 5UL); // at index 5 from start
   * // 2 from end
   * auto b_slice_from_end = block::slice(b, elastica::end - 2UL);
   * \endcode
   *
   * \note
   * not marked noexcept because we can have out of range indices
   */
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin>& block_like,
                                      std::size_t index) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin> const& block_like,
                                      std::size_t index) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(BlockView<Plugin>&& block_like,
                                      std::size_t index) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(
        static_cast<BlockView<Plugin>&&>(block_like).parent(),
        static_cast<BlockView<Plugin>&&>(block_like).region().start + index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(ConstBlockView<Plugin> const& block_like,
                                      std::size_t index) noexcept {
    // index is already confirmed within the acceptable limits, so use
    // slice_backend instead of slice(parent())
    return slice_backend(block_like.parent(),
                         block_like.region().start + index);
  }
  //****************************************************************************

  //@}
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks
