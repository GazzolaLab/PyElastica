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
#include <utility>
#include <stdexcept> // logic error

namespace blocks {

  //**Customization*************************************************************
  /*!\name Block slice customization
   * \brief Customization of viewing operation
   * \ingroup block_concepts
   *
   * \details
   * The slice_backend() is a customization point for implementing
   * viewing backend of a block-like type.
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
  decltype(auto) slice_backend(Viewable<BlockLike>& block_like,
                               std::size_t start_index,
                               std::size_t region_size);
  template <typename BlockLike>
  decltype(auto) slice_backend(Viewable<BlockLike> const& block_like,
                               std::size_t start_index,
                               std::size_t region_size);
  template <typename BlockLike>
  decltype(auto) slice_backend(Viewable<BlockLike>&& block_like,
                               std::size_t start_index,
                               std::size_t region_size);
  template <typename BlockLike>
  decltype(auto) slice_backend(Viewable<BlockLike> const&& block_like,
                               std::size_t start_index,
                               std::size_t region_size);
  //@}
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  // There are 16 combinations here, we only list the most important ones.
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              std::size_t start_index, std::size_t stop_index);
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              std::size_t start_index,
                              ::elastica::from_end stop_index);
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              ::elastica::from_end start_index,
                              std::size_t stop_index);
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              ::elastica::from_end start_index,
                              ::elastica::from_end stop_index);
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Viewable concept
   * \ingroup block_concepts
   *
   * \details
   * The Viewable template class represents the concept of getting a view (i.e.
   * a chunk of more than one system) into a
   * block-like data-structure using the blocks::slice() or #operator[] methods.
   * The view can be further viewed or sliced.
   *
   * \see blocks::slice(), Sliceable, BlockView
   */
  template <typename BlockLike>
  class Viewable : public elastica::CRTPHelper<BlockLike, Viewable> {
   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<BlockLike, Viewable>;
    //**************************************************************************

   public:
    //**Self methods************************************************************
    //! CRTP methods
    using CRTP::self;
    //**************************************************************************

    // NOTE : Since this is an interface class, we eschew templates for the
    // index types below in favor of actual types, for it to appear nicer in
    // Doxygen.

    //**Slice operators*********************************************************
    /*!\name slice operators */
    //@{

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](std::pair<std::size_t, std::size_t> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<std::size_t, ::elastica::from_end> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<::elastica::from_end, std::size_t> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<::elastica::from_end, ::elastica::from_end> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<std::size_t, std::size_t> index) const& {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<std::size_t, ::elastica::from_end> index) const& {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<::elastica::from_end, std::size_t> index) const& {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<::elastica::from_end, ::elastica::from_end> index) const& {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<std::size_t, decltype(::elastica::end)> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Subscript operator for viewing the block
     *
     * \param index The range of indices to view
     * \return Slice of the current block
     */
    decltype(auto) operator[](
        std::pair<::elastica::from_end, decltype(::elastica::end)> index) & {
      using ::blocks::slice;
      return slice(self(), index.first, index.second);
    }
    //**************************************************************************

    //@}
    //**************************************************************************
  };
  //****************************************************************************

  namespace detail {

    template <typename V, typename FI, typename SI>
    decltype(auto) slice_logic(V&& viewable, FI start_index, SI stop_index) {
      std::size_t const start_idx =
          units_check(std::forward<V>(viewable), start_index);
      // workaround to support slices with last index : stop_index - 1 does not
      // throw in check(), and once we have the numeric index, we add one to
      // recover original index.
      std::size_t const stop_idx =
          units_check(std::forward<V>(viewable), stop_index - 1UL) + 1UL;
      if (not(stop_idx > start_idx)) {
        throw std::logic_error(
            "Stop index must be greater than the start index");
      }
      return slice_backend(std::forward<V>(viewable), start_idx,
                           stop_idx - start_idx);
    }

  }  // namespace detail

  //============================================================================
  //
  //  BLOCK VIEW FUNCTION
  //
  //============================================================================

  //**Block view****************************************************************
  /*!\name Block view functions*/
  //@{

  //****************************************************************************
  /*!\brief View operator
   * \ingroup block_concepts
   *
   * \details
   * Implements contiguous views on entities modeling block concept.
   *
   * \param block_like Data-structure modeling the Viewable concept
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
   *
   * \note
   * not marked noexcept because we can have out of range indices
   *
   * \note
   * Views must have size >= 1 and hence the same indices to start and end will
   * raise an error.
   *
   * \see blocks::Viewable, blocks::BlockView
   */

  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              std::size_t start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              ::elastica::from_end start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              std::size_t start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>& viewable,
                              ::elastica::from_end start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike>& viewable, std::size_t start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike>& viewable, ::elastica::from_end start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const& viewable,
                              std::size_t start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const& viewable,
                              ::elastica::from_end start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const& viewable,
                              std::size_t start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const& viewable,
                              ::elastica::from_end start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike> const& viewable, std::size_t start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike> const& viewable, ::elastica::from_end start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(viewable.self(), start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>&& viewable,
                              std::size_t start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>&& viewable,
                              ::elastica::from_end start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>&& viewable,
                              std::size_t start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike>&& viewable, std::size_t start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike>&& viewable, ::elastica::from_end start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike>&& viewable,
                              ::elastica::from_end start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike&&>(viewable), start_index,
                               stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const&& viewable,
                              std::size_t start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const&& viewable,
                              ::elastica::from_end start_index,
                              std::size_t stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const&& viewable,
                              std::size_t start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(Viewable<BlockLike> const&& viewable,
                              ::elastica::from_end start_index,
                              ::elastica::from_end stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike> const&& viewable, std::size_t start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //****************************************************************************
  template <typename BlockLike>
  inline decltype(auto) slice(
      Viewable<BlockLike> const&& viewable, ::elastica::from_end start_index,
      decltype(::elastica::end) stop_index) /*noexcept*/ {
    return detail::slice_logic(static_cast<BlockLike const&&>(viewable),
                               start_index, stop_index);
  }
  //****************************************************************************

  //@}
  //****************************************************************************

}  // namespace blocks
