#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Block/Concepts/Types.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/End.hpp"
// #include "Utilities/PrettyType.hpp"
//
#include <cstddef>  // size_t
#include <stdexcept>

namespace blocks {

  //**Customization*************************************************************
  /*!\name Block units customization
   * \brief Customization of units
   * \ingroup block_concepts
   *
   * \details
   * The n_units() is a customization point for implementing
   * units backend of a block-like type.
   *
   * Customization is achieved by overloading this function for the target
   * block type. If not overload is provided, then a compiler/linker error is
   * raised.
   *
   * \example
   * The following shows an example of customizing the slice backend.
   * \snippet Transfer/Test_Transfer.cpp customization_eg
   *
   * \see blocks::n_units()
   */
  //@{
  template <typename BlockLike>
  auto n_units(Subdividable<BlockLike> const& block_like) noexcept
      -> std::size_t;
  //@}
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Models the Subdividable concept
   * \ingroup block_concepts
   *
   * \details
   * The Subdividable template class represents the concept of a data-structure
   * with multiple units/entities. It is useful as a template in supporting
   * generation of slices/views from the data.
   *
   * \see blocks::n_units()
   */
  template <typename BlockLike>
  class Subdividable : public elastica::CRTPHelper<BlockLike, Subdividable> {
   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<BlockLike, Subdividable>;
    //**************************************************************************

   public:
    //**Self methods************************************************************
    //! CRTP methods
    using CRTP::self;
    //**************************************************************************
  };
  //****************************************************************************

  //**Index check support*******************************************************
  /*!\name Block index check */
  /*!\brief Checks whether block unit indices are within range of accessible
   * values
   */
  //@{
  template <typename BlockLike>
  auto units_check(Subdividable<BlockLike> const& subdividable,
                   std::size_t index_to_be_sliced) -> std::size_t {
    using ::blocks::n_units;
    if (index_to_be_sliced < n_units(subdividable.self())) {
      return index_to_be_sliced;
    } else {
      throw std::out_of_range(
          "Index to be sliced exceeds the number of units inside a " +
          BlockLike::name());
          //pretty_type::name<BlockLike>());
    }
  }

  template <typename BlockLike>
  inline auto units_check(Subdividable<BlockLike> const& subdividable,
                          elastica::from_end index_to_be_sliced)
      -> std::size_t { /* This index cannot be 0 */
    using ::blocks::n_units;

    auto const fe = index_to_be_sliced.i;
    auto const un = n_units(subdividable.self());
    if (fe <= un) {
      return un - fe;
    } else {
      throw std::out_of_range(
          "Index to be sliced exceeds the number of units inside a " +
          BlockLike::name());
          // pretty_type::name<BlockLike>());
    }
  }
  // @}
  //****************************************************************************

}  // namespace blocks
