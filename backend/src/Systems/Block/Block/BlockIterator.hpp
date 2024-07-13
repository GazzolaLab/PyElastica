#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>
#include <iterator>
#include <limits>

#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/Block/Types.hpp"
//

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Implementation of a iterator for blocks.
   * \ingroup blocks
   *
   * The BlockIterator represents a generic random-access iterator that can be
   * used for all block types in \elastica
   */
  template <typename Plugin>
  class BlockIterator {
   private:
    //**Type definitions********************************************************
    using ParentBlock = Block<Plugin>;
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! The iterator category.
    using IteratorCategory = std::random_access_iterator_tag;
    //! Type of the underlying elements.
    using ValueType =
        typename PluginFrom<ParentBlock>::template to<BlockSlice>::type;
    //! Pointer return type.
    using PointerType = ValueType;
    //! Reference return type.
    using ReferenceType = ValueType;
    //! Difference between two iterators.
    using DifferenceType = std::size_t;

    // STL iterator requirements
    //! The iterator category.
    using iterator_category = IteratorCategory;
    //! Type of the underlying elements.
    using value_type = ValueType;
    //! Pointer return type.
    using pointer = PointerType;
    //! Reference return type.
    using reference = ReferenceType;
    //! Difference between two iterators.
    using difference_type = DifferenceType;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief Default constructor for the BlockIterator class.
     */
    explicit constexpr BlockIterator() noexcept
        : block_(nullptr), curr_idx_(std::numeric_limits<std::size_t>::max()) {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief Constructor for the BlockIterator class.
     *
     * \param ptr Pointer to the block
     * \param idx Index of slicing
     */
    constexpr BlockIterator(ParentBlock* ptr, std::size_t idx) noexcept
        : block_(ptr), curr_idx_(idx) {}
    BlockIterator(const BlockIterator&) = default;
    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~BlockIterator() = default;
    //@}
    //**************************************************************************

    //**Assignment operators****************************************************
    /*!\name Assignment operators */
    //@{

    //**************************************************************************
    /*!\brief Addition assignment operator.
     *
     * \param inc The increment of the iterator.
     * \return Reference to the incremented iterator.
     */
    constexpr BlockIterator& operator+=(difference_type inc) noexcept {
      curr_idx_ += inc;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subtraction assignment operator.
     *
     * \param dec The decrement of the iterator.
     * \return Reference to the decremented iterator.
     */
    constexpr BlockIterator& operator-=(difference_type dec) noexcept {
      // does this by any chance go beyond bounds?
      curr_idx_ -= dec;
      return *this;
    };
    //**************************************************************************

    BlockIterator& operator=(const BlockIterator&) = default;
    //@}
    //**************************************************************************

    //**Increment/decrement operators*******************************************
    /*!\name Increment/decrement operators */
    //@{

    //**************************************************************************
    /*!\brief Pre-increment operator.
     *
     * \return Reference to the incremented iterator.
     */
    constexpr BlockIterator& operator++() noexcept {
      ++curr_idx_;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Post-increment operator.
     *
     * \return The previous position of the iterator.
     */
    constexpr const BlockIterator operator++(int) noexcept {
      return BlockIterator(block_, curr_idx_++);
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Pre-decrement operator.
     *
     * \return Reference to the decremented iterator.
     */
    constexpr BlockIterator& operator--() noexcept {
      --curr_idx_;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Post-decrement operator.
     *
     * \return The previous position of the iterator.
     */
    constexpr const BlockIterator operator--(int) noexcept {
      return BlockIterator(block_, curr_idx_--);
    };
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Access operators********************************************************
    /*!\name Access operators */
    //@{

    //**************************************************************************
    /*!\brief Direct access to the underlying slice.
     *
     * \param index Access index.
     * \return Accessed slice.
     */
    constexpr ReferenceType operator[](size_t index) const noexcept {
      // A static cast is necessary to properly return tagged systems. Tagged
      // systems may be implemented as deriving from a Block, in which case the
      // slice method() returns a non-tagged reference, which is then converted
      // to a tagged reference.
      return static_cast<ReferenceType>(slice(*block_, curr_idx_ + index));
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Direct access to the slice at the current iterator position.
     *
     * \return Current slice.
     */
    constexpr ReferenceType operator*() const noexcept {
      return static_cast<ReferenceType>(slice_backend(*block_, curr_idx_));
    };
    //@}
    //**************************************************************************

    //**Utility functions*******************************************************
    /*!\name Utility functions */
    //@{

    //**************************************************************************
    /*!\brief Low-level access to the underlying data of the iterator.
     *
     * \return Pointer to the parent block.
     */
    constexpr ParentBlock* data() const noexcept { return block_; }
    //**************************************************************************

    //************************************************************************
    /*!\brief Gets the parent block
     */
    inline constexpr auto parent() & noexcept -> ParentBlock& {
      return *block_;
    }
    //************************************************************************

    //************************************************************************
    /*!\brief Gets the parent block
     */
    inline constexpr auto parent() const& noexcept -> ParentBlock const& {
      return *block_;
    }
    //************************************************************************

    //************************************************************************
    /*!\brief Gets the parent block
     *
     * \note
     * This method is useful when xvalue block-slices are generated
     * on-the-fly via BlockRefs. If this overload is not present, then the
     * const& is picked up, and it is not possible to assign values to the
     * parent block anymore.
     */
    inline constexpr auto parent() && noexcept -> ParentBlock& {
      return *block_;
    }
    //************************************************************************

    //************************************************************************
    /*!\brief Gets the parent block
     *
     * \note
     * This method is provided for symmetry with the overload above.
     */
    inline constexpr auto parent() const&& noexcept -> ParentBlock const& {
      return *block_;
    }
    //************************************************************************

    //**************************************************************************
    /*!\brief Low-level access to the index of the iterator.
     *
     * \return Index into the current slice location
     */
    constexpr auto index() const noexcept -> std::size_t { return curr_idx_; }
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! Pointer to the held block.
    ParentBlock* block_;
    //! Current slicing index
    std::size_t curr_idx_;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //============================================================================
  //
  //  GLOBAL OPERATORS
  //
  //============================================================================

  //****************************************************************************
  /*!\name BlockIterator operators */
  //@{

  //****************************************************************************
  /*!\brief Equality comparison between two BlockIterator objects.
   *
   * \param lhs The left-hand side iterator.
   * \param rhs The right-hand side iterator.
   * \return \a true if the iterators refer to the same element, \a false if
   * not.
   */
  template <typename Plugin>
  constexpr inline bool operator==(const BlockIterator<Plugin>& lhs,
                                   const BlockIterator<Plugin>& rhs) noexcept {
    return lhs.data() == rhs.data() and lhs.index() == rhs.index();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief InEquality comparison between two BlockIterator objects.
   *
   * \param lhs The left-hand side iterator.
   * \param rhs The right-hand side iterator.
   * \return \a true if the iterators don't refer to the same element, \a false
   * if they do.
   */
  template <typename Plugin>
  constexpr inline bool operator!=(const BlockIterator<Plugin>& lhs,
                                   const BlockIterator<Plugin>& rhs) noexcept {
    return lhs.data() != rhs.data() or lhs.index() != rhs.index();
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
  /*!\brief Implementation of a iterator for blocks.
   * \ingroup blocks
   *
   * The ConstBlockIterator represents a generic random-access iterator that can
   * be used for all block types in \elastica
   */
  template <typename Plugin>
  class ConstBlockIterator {
   private:
    //**Type definitions********************************************************
    using ParentBlock = Block<Plugin>;
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! The iterator category.
    using IteratorCategory = std::random_access_iterator_tag;
    //! Type of the underlying elements.
    using ValueType =
        typename PluginFrom<ParentBlock>::template to<ConstBlockSlice>::type;
    //! Pointer return type.
    using PointerType = ValueType;
    //! Reference return type.
    using ReferenceType = ValueType;
    //! Difference between two iterators.
    using DifferenceType = std::size_t;

    // STL iterator requirements
    //! The iterator category.
    using iterator_category = IteratorCategory;
    //! Type of the underlying elements.
    using value_type = ValueType;
    //! Pointer return type.
    using pointer = PointerType;
    //! Reference return type.
    using reference = ReferenceType;
    //! Difference between two iterators.
    using difference_type = DifferenceType;
    //**************************************************************************

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief Default constructor for the ConstBlockIterator class.
     */
    explicit constexpr ConstBlockIterator() noexcept
        : block_(nullptr), curr_idx_(std::numeric_limits<std::size_t>::max()) {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief Constructor for the ConstBlockIterator class.
     *
     * \param ptr Pointer to the block
     * \param idx Index of slicing
     */
    constexpr ConstBlockIterator(ParentBlock const* ptr,
                                 std::size_t idx) noexcept
        : block_(ptr), curr_idx_(idx) {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief Conversion constructor from a BlockIterator instance.
     *
     * \param it The BlockIterator instance to be copied.
     */
    constexpr ConstBlockIterator(BlockIterator<Plugin> const& it) noexcept
        : block_(it.data()), curr_idx_(it.index()){};
    //**************************************************************************

    ConstBlockIterator(const ConstBlockIterator&) = default;
    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~ConstBlockIterator() = default;
    //@}
    //**************************************************************************

    //**Assignment operators****************************************************
    /*!\name Assignment operators */
    //@{

    //**************************************************************************
    /*!\brief Addition assignment operator.
     *
     * \param inc The increment of the iterator.
     * \return Reference to the incremented iterator.
     */
    constexpr ConstBlockIterator& operator+=(difference_type inc) noexcept {
      curr_idx_ += inc;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subtraction assignment operator.
     *
     * \param dec The decrement of the iterator.
     * \return Reference to the decremented iterator.
     */
    constexpr ConstBlockIterator& operator-=(difference_type dec) noexcept {
      // does this by any chance go beyond bounds?
      curr_idx_ -= dec;
      return *this;
    };
    //**************************************************************************

    ConstBlockIterator& operator=(const ConstBlockIterator&) = default;
    //@}
    //**************************************************************************

    //**Increment/decrement operators*******************************************
    /*!\name Increment/decrement operators */
    //@{

    //**************************************************************************
    /*!\brief Pre-increment operator.
     *
     * \return Reference to the incremented iterator.
     */
    constexpr ConstBlockIterator& operator++() noexcept {
      ++curr_idx_;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Post-increment operator.
     *
     * \return The previous position of the iterator.
     */
    constexpr const ConstBlockIterator operator++(int) noexcept {
      return ConstBlockIterator(block_, curr_idx_++);
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Pre-decrement operator.
     *
     * \return Reference to the decremented iterator.
     */
    constexpr ConstBlockIterator& operator--() noexcept {
      --curr_idx_;
      return *this;
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Post-decrement operator.
     *
     * \return The previous position of the iterator.
     */
    constexpr const ConstBlockIterator operator--(int) noexcept {
      return ConstBlockIterator(block_, curr_idx_--);
    };
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Access operators********************************************************
    /*!\name Access operators */
    //@{

    //**************************************************************************
    /*!\brief Direct access to the underlying slice.
     *
     * \param index Access index.
     * \return Accessed slice.
     */
    constexpr ReferenceType operator[](size_t index) const noexcept {
      return static_cast<ReferenceType>(slice(*block_, curr_idx_ + index));
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Direct access to the slice at the current iterator position.
     *
     * \return Current slice.
     */
    constexpr ReferenceType operator*() const noexcept {
      return static_cast<ReferenceType>(slice(*block_, curr_idx_));
    };
    //@}
    //**************************************************************************

    //**Utility functions*******************************************************
    /*!\name Utility functions */
    //@{

    //**************************************************************************
    /*!\brief Low-level access to the underlying data of the iterator.
     *
     * \return Pointer to the parent block.
     */
    constexpr ParentBlock const* data() const noexcept { return block_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets the parent block
     */
    inline constexpr auto parent() const noexcept -> ParentBlock const& {
      return *block_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Low-level access to the index of the iterator.
     *
     * \return Index into the current slice location
     */
    constexpr auto index() const noexcept -> std::size_t { return curr_idx_; }
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! Pointer to the held (const) block.
    ParentBlock const* block_;
    //! Current slicing index
    std::size_t curr_idx_;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //============================================================================
  //
  //  GLOBAL OPERATORS
  //
  //============================================================================

  //****************************************************************************
  /*!\name ConstBlockIterator operators */
  //@{

  //****************************************************************************
  /*!\brief Equality comparison between two ConstBlockIterator objects.
   *
   * \param lhs The left-hand side iterator.
   * \param rhs The right-hand side iterator.
   * \return \a true if the iterators refer to the same element, \a false if
   * not.
   */
  template <typename Plugin>
  constexpr inline bool operator==(
      const ConstBlockIterator<Plugin>& lhs,
      const ConstBlockIterator<Plugin>& rhs) noexcept {
    return lhs.data() == rhs.data() and lhs.index() == rhs.index();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief InEquality comparison between two ConstBlockIterator objects.
   *
   * \param lhs The left-hand side iterator.
   * \param rhs The right-hand side iterator.
   * \return \a true if the iterators don't refer to the same element, \a false
   * if they do.
   */
  template <typename Plugin>
  constexpr inline bool operator!=(
      const ConstBlockIterator<Plugin>& lhs,
      const ConstBlockIterator<Plugin>& rhs) noexcept {
    return lhs.data() != rhs.data() or lhs.index() != rhs.index();
  }
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks
