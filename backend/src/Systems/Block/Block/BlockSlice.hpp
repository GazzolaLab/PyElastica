#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block/Block/Types.hpp"
//
#include "Systems/Block/Block/AsVariables.hpp"
#include "Systems/Block/Block/BlockSliceFacade.hpp"
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/TypeTraits.hpp"

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Implementation of a slice/view on an \elastica Block
   * \ingroup blocks
   *
   * \details
   * The BlockSlice template class provides a slice (or) view into a Block of
   * `data` and `operations`. It also models the `ComputationalBlock` concept
   * and hence can be used across places where a \elastica Block is used as
   * a template parameter.
   *
   * Similar to Block, BlockSlice can also be customized (specialized) for any
   * Lagrangian entity, but experience suggests that extensive customization is
   * not needed. For a concrete example, please see specializations of
   * BlockSlice in @ref cosserat_rod.
   *
   * \usage
   * The intended usage of a BlockSlice is when a view is required into the data
   * held by a block---this is frequently the case when a user either
   * 1. adds an entity into the Simulator, or
   * 2. requires access (for e.g. reading/writing to disk) to only a portion of
   * the block.
   * The pattern that is most commonly seen in the use case (1) is for a
   * BlockSlice templated on some `Plugin` type to be only used with a Block
   * of the same `Plugin` type, when adding new entities to the Block.
   * For use case (2) we suggest the user to the blocks::slice() function, which
   * has an intuitive, explicit slice syntax, or even the subscript operator[].
   * We note that we might explicitly disable the subscript operator [] for
   * slicing in the future, as the semantics are potentially unclear.
   *
   * Finally, with no additional coding effort, the BlockSlice has exactly the
   * same operations as the mother Block (aka Block of the same `Plugin` type),
   * but now it operates only on that slice of the data. This means that a
   * BlockSlice can be used as a small Block in itself which greatly
   * sbackendifies interfacing different components of \elastica---the user need
   * not care or even know about whether the data that she has is a Block or a
   * BlockSlice! For example, \code
   *     // ... make simulator etc ...
   *     auto my_rod = simulator.emplace_back<CosseratRod>( * ...args... *);
   *     // the args go and form a Block
   *     // which in turn returns a BlockSlice
   *     // which the user gets as my_rod
   *
   *     // use my_rod like a regular cosserat rod
   *     simulator->constrain(my_rod)->using<SomeConstraint>( *...args... *);
   * \endcode
   * This abstraction helped us constrain \c my_rod, embedded in a Block of
   * data, using \c SomeConstraint just like any non-Blocked item of the
   * \elastica library.
   *
   * \tparam Plugin The computational plugin modeling a Lagrangian entity
   *
   * \see Block, blocks::slice
   */
  template <class Plugin>
  class BlockSlice : public detail::BlockSliceFacade<Plugin>,
                     public Gettable<BlockSlice<Plugin>>,
                     public Plugin {
    // NOTE : Plugin is inherited after Facade so that indices and references to
    // the parent block is filled first. This is because some elements of Plugin
    // may require that the slice (and index) be valid, for example to generate
    // internal refs/pointers for time-stepping.
   protected:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using Parent = detail::BlockSliceFacade<Plugin>;
    //! Type of the slice
    using This = BlockSlice<Plugin>;
    //! Type of gettable
    using GetAffordance = Gettable<This>;
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
     * \param parent_block Parent block for the slice
     * \param index Index of the slice
     */
    explicit BlockSlice(ParentBlock& parent_block, std::size_t index) noexcept
        : Parent(parent_block, index), GetAffordance(), PluginType() {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief The copy constructor.
     *
     * \param other slice to copy
     */
    BlockSlice(BlockSlice const& other)
        : Parent(static_cast<Parent const&>(other)),
          GetAffordance(),
          PluginType(static_cast<PluginType const&>(other)){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other slice to move from
     */
    BlockSlice(BlockSlice&& other) noexcept
        : Parent(static_cast<Parent&&>(other)),
          GetAffordance(),
          PluginType(static_cast<PluginType&&>(other)){};
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~BlockSlice() = default;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Implementation of a slice/view on a const \elastica Block
   * \ingroup blocks
   *
   * \details
   * The ConstBlockSlice template class provides a slice (or) view into a
   * constant Block of \a data and \a operations. It also models the
   * `ComputationalBlock` concept and hence can be used across places where a
   * \elastica Block is used as a template parameter.
   *
   * Notably, it differs from BlockSlice in one key aspect : the underlying
   * data/view is always constant and cannot be modified (it is only read only).
   * Hence ConstBlock is useful for propagating const-correctness throughout the
   * code. It can be used in places where one needs to pass (const)-data to the
   * user which she can then copy and use it for her own purposes.
   *
   * \tparam Plugin The computational plugin modeling a Lagrangian entity
   *
   * \see BlockSlice
   */
  template <class Plugin>
  class ConstBlockSlice : public detail::ConstBlockSliceFacade<Plugin>,
                          public Gettable<ConstBlockSlice<Plugin>>,
                          public Plugin {
    // NOTE : Plugin is inherited after Facade so that indices and references to
    // the parent block is filled first. This is because some elements of Plugin
    // may require that the slice (and index) be valid, for example to generate
    // internal refs/pointers for time-stepping.
   protected:
    //**Type definitions********************************************************
    //! Type of the parent plugin
    using Parent = detail::ConstBlockSliceFacade<Plugin>;
    //! Type of the slice
    using This = ConstBlockSlice<Plugin>;
    //! Type of gettable
    using GetAffordance = Gettable<This>;
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
     * \param parent_block Parent block for the slice
     * \param index Index of the slice
     */
    ConstBlockSlice(ParentBlock const& parent_block, std::size_t index) noexcept
        : Parent(parent_block, index), GetAffordance(), PluginType() {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief The copy constructor.
     *
     * \param other slice to copy
     */
    ConstBlockSlice(ConstBlockSlice const& other)
        : Parent(static_cast<Parent const&>(other)),
          GetAffordance(),
          PluginType(static_cast<PluginType const&>(other)){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other slice to move from
     */
    ConstBlockSlice(ConstBlockSlice&& other) noexcept
        : Parent(static_cast<Parent&&>(other)),
          GetAffordance(),
          PluginType(static_cast<PluginType&&>(other)){};
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~ConstBlockSlice() = default;
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
  /*!\brief Equality comparison between two BlockSlice objects.
   *
   * \param lhs The left-hand side slice.
   * \param rhs The right-hand side slice.
   * \return \a true if the slices are same, else \a false
   */
  template <class Plugin>
  inline constexpr auto operator==(BlockSlice<Plugin> const& lhs,
                                   BlockSlice<Plugin> const& rhs) noexcept
      -> bool {
    return static_cast<detail::BlockSliceFacade<Plugin> const&>(lhs) ==
           static_cast<detail::BlockSliceFacade<Plugin> const&>(rhs);
  }
  //****************************************************************************

  //**Equality operator*********************************************************
  /*!\brief Equality comparison between two ConstBlockSlice objects.
   *
   * \param lhs The left-hand side const slice.
   * \param rhs The right-hand side const slice.
   * \return \a true if the const slices are same, else \a false
   */
  template <class Plugin>
  inline constexpr auto operator==(ConstBlockSlice<Plugin> const& lhs,
                                   ConstBlockSlice<Plugin> const& rhs) noexcept
      -> bool {
    return static_cast<detail::ConstBlockSliceFacade<Plugin> const&>(lhs) ==
           static_cast<detail::ConstBlockSliceFacade<Plugin> const&>(rhs);
  }
  //****************************************************************************

  //============================================================================
  //
  //  GET FUNCTIONS
  //
  //============================================================================

  namespace detail {

    template <typename BlockVariableTag, typename SliceLike>
    inline constexpr decltype(auto) get_slice(SliceLike&& slice_like) noexcept {
      return std::forward<SliceLike>(slice_like)
          .parent()
          .template slice<BlockVariableTag>(
              std::forward<SliceLike>(slice_like).index());
    }

  }  // namespace detail

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name Get functions for Slice types */
  //@{

  //****************************************************************************
  /*!\brief Extract element from a BlockSlice
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the Block `block_slice` whose tag type is `Tag`.
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
      BlockSlice<Plugin>& block_slice) noexcept {
    return detail::get_slice<BlockVariableTag>(block_slice);
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockSlice<Plugin> const& block_slice) noexcept {
    return detail::get_slice<BlockVariableTag>(block_slice);
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockSlice<Plugin>&& block_slice) noexcept {
    return detail::get_slice<BlockVariableTag>(
        static_cast<BlockSlice<Plugin>&&>(block_slice));
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      BlockSlice<Plugin> const&& block_slice) noexcept {
    return detail::get_slice<BlockVariableTag>(
        static_cast<BlockSlice<Plugin> const&&>(block_slice));
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      ConstBlockSlice<Plugin> const& block_slice) noexcept {
    return detail::get_slice<BlockVariableTag>(block_slice);
  }
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
  /*!\name Block slice functions */
  //@{

  //****************************************************************************
  /*!\brief Slice backend for Block
   * \ingroup blocks
   *
   * \details
   * Implements slices on entities modeling block concept.
   *
   * \param block_like Data-structure modeling the block concept
   * \param index The index of the slice (int or from_end)
   *
   * \example
   * Given a block `b`, we can slice it using
   * \code
   * auto b_slice = block::slice(b, 5UL); // at index 5 from start
   * auto b_slice_from_end = block::slice(b, elastica::end - 2UL); // 2 from end
   * \endcode
   *
   * \note
   * not marked noexcept because we can have out of range indices
   */
  template <typename Plugin>
  inline decltype(auto) slice_backend(Block<Plugin>& block_like,
                                      std::size_t index) noexcept {
    using ReturnType =
        typename PluginFrom<Block<Plugin>>::template to<BlockSlice>::type;
    return ReturnType{block_like, index};
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  inline decltype(auto) slice_backend(Block<Plugin> const& block_like,
                                      std::size_t index) noexcept {
    using ReturnType =
        typename PluginFrom<Block<Plugin>>::template to<ConstBlockSlice>::type;
    return ReturnType{block_like, index};
  }
  //****************************************************************************

  //@}
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks
