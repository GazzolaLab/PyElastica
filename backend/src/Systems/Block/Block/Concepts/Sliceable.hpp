#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Block/Concepts/Types.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/End.hpp"
//
#include <cstddef>  // size_t

namespace blocks {

  //**Customization*************************************************************
  /*!\name Block slice customization
   * \brief Customization of slicing operation
   * \ingroup block_concepts
   *
   * \details
   * The slice_backend() is a customization point for implementing
   * slicing backend of a block-like type.
   *
   * Customization is achieved by overloading this function for the target
   * block type. If not overload is provided, then a compiler/linker error is
   * raised.
   *
   * \example
   * The following shows an example of customizing the slice backend.
   * \snippet Transfer/Test_Transfer.cpp customization_eg
   *
   * \see blocks::slice()
   */
  //@{
  template <typename BlockLike>
  decltype(auto) slice_backend(Sliceable<BlockLike>& block_like,
                               std::size_t index);
  template <typename BlockLike>
  decltype(auto) slice_backend(Sliceable<BlockLike> const& block_like,
                               std::size_t index);
  template <typename BlockLike>
  decltype(auto) slice_backend(Sliceable<BlockLike>&& block_like,
                               std::size_t index);
  template <typename BlockLike>
  decltype(auto) slice_backend(Sliceable<BlockLike> const&& block_like,
                               std::size_t index);
  //@}
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>& sliceable,
                              std::size_t index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>& sliceable,
                              ::elastica::from_end index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>&& sliceable,
                              std::size_t index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>&& sliceable,
                              ::elastica::from_end index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const& sliceable,
                              std::size_t index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const& sliceable,
                              ::elastica::from_end index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const&& sliceable,
                              std::size_t index);
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const&& sliceable,
                              ::elastica::from_end index);
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Models the Sliceable concept
   * \ingroup block_concepts
   *
   * \details
   * The Sliceable template class represents the concept of slicing a
   * block-like data-structure using the blocks::slice() or #operator[] methods.
   * It provides a slice into the data held by the parent---and hence acts like
   * a ponter to the parents data. This slice cannot be further sliced or
   * viewed.
   *
   * \see blocks::slice(), Viewable, BlockSlice
   */
  template <typename BlockLike>
  class Sliceable : public elastica::CRTPHelper<BlockLike, Sliceable> {
   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<BlockLike, Sliceable>;
    //**************************************************************************

   public:
    //**Self methods************************************************************
    //! CRTP methods
    using CRTP::self;
    //**************************************************************************

    // NOTE : Since this is an interface class, we eschew templates for the
    // index types in favor of actual types below, for it to appear nicer in
    // Doxygen.

    //**Slice operators*********************************************************
    /*!\name slice operators */
    //@{

    //**************************************************************************
    /*!\brief Subscript operator for slicing the block
     *
     * \param index Index of slice
     * \return Slice of the current block
     */
    decltype(auto) operator[](std::size_t index) & {
      using ::blocks::slice;
      return slice(self(), index);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for slicing the block
     *
     * \param index Index of slice
     * \return Slice of the current block
     */
    decltype(auto) operator[](::elastica::from_end index) & {
      using ::blocks::slice;
      return slice(self(), index);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for slicing the block
     *
     * \param index Index of slice
     * \return Slice of the current block
     */
    decltype(auto) operator[](std::size_t index) const& {
      using ::blocks::slice;
      return slice(self(), index);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for slicing the block
     *
     * \param index Index of slice
     * \return Slice of the current block
     */
    decltype(auto) operator[](::elastica::from_end index) const& {
      using ::blocks::slice;
      return slice(self(), index);
    }
    //**************************************************************************

    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //============================================================================
  //
  //  BLOCK SLICE FUNCTION
  //
  //============================================================================

  //**Block slice***************************************************************
  /*!\name Block slice functions*/
  //@{

  //****************************************************************************
  /*!\brief Slice operator
   * \ingroup block_concepts
   *
   * \details
   * Implements slices on entities modeling Sliceable concept.
   *
   * \param sliceable Data-structure modeling the Sliceable concept
   * \param index The index of the slice (int or from_end)
   *
   * \example
   * Given a block `b`, we can slice it using
   * \code
   * auto b_slice = block::slice(b, 5UL); // Gets b[5], at index 5 from start
   * // Gets b[-2]
   * auto b_slice_from_end = block::slice(b, elastica::end - 2UL); // 2 from end
   * \endcode
   *
   * \note
   * not marked noexcept because we can have out of range indices
   *
   * \see blocks::Sliceable , blocks::BlockSlice
   */
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>& sliceable,
                              std::size_t index) /*noexcept*/ {
    return slice_backend(sliceable.self(),
                         units_check(sliceable.self(), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>& sliceable,
                              elastica::from_end index) /*noexcept*/ {
    return slice_backend(sliceable.self(),
                         units_check(sliceable.self(), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>&& sliceable,
                              std::size_t index) /*noexcept*/ {
    return slice_backend(
        static_cast<BlockLike&&>(sliceable),
        units_check(static_cast<BlockLike&&>(sliceable), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike>&& sliceable,
                              elastica::from_end index) /*noexcept*/ {
    return slice_backend(
        static_cast<BlockLike&&>(sliceable),
        units_check(static_cast<BlockLike&&>(sliceable), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const& sliceable,
                              std::size_t index) /*noexcept*/ {
    return slice_backend(sliceable.self(),
                         units_check(sliceable.self(), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const& sliceable,
                              elastica::from_end index) /*noexcept*/ {
    return slice_backend(sliceable.self(),
                         units_check(sliceable.self(), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const&& sliceable,
                              std::size_t index) /*noexcept*/ {
    return slice_backend(
        static_cast<BlockLike const&&>(sliceable),
        units_check(static_cast<BlockLike const&&>(sliceable), index));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Sliceable<BlockLike> const&& sliceable,
                              elastica::from_end index) /*noexcept*/ {
    return slice_backend(
        static_cast<BlockLike const&&>(sliceable),
        units_check(static_cast<BlockLike const&&>(sliceable), index));
  }
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks
