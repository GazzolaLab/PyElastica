#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/End.hpp"  // from_end
//
#include <cstddef>  // size_t
#include <stdexcept>

namespace elastica {

  //============================================================================
  //
  //  ACCESS API FUNCTIONS
  //
  //============================================================================

  // Defined in Access.hpp
  struct AnyPhysicalSystem;

  //**Index check functions*****************************************************
  /*!\name Index check functions */
  //@{
  /*!\brief Checks that an index is valid for a system
   * \ingroup systems
   */
  auto index_check(AnyPhysicalSystem& t, std::size_t index) -> std::size_t;
  auto index_check(AnyPhysicalSystem& t, from_end index) -> std::size_t;
  //@}
  //****************************************************************************

  //**Index check helpers*******************************************************
  /*!\name Index check helpers */
  //@{
  /*!\brief Helpers to validate indices for a system
   * \ingroup systems
   */
  inline auto index_check_helper(std::size_t max_index,
                                 std::size_t index_to_be_sliced)
      -> std::size_t {
    if (index_to_be_sliced < max_index) {
      return index_to_be_sliced;
    } else {
      throw std::out_of_range(
          "Index to be sliced (from the start) exceeds the number of dofs!");
    }
  }

  inline auto index_check_helper(std::size_t max_index,
                                 elastica::from_end index_to_be_sliced)
      -> std::size_t {
    const auto fe = index_to_be_sliced.i;
    if (fe <= max_index) {
      return max_index - fe;
    } else {
      throw std::out_of_range(
          "Index to be sliced (from the end) exceeds the number of dofs!");
    }
  }
  //@}
  //****************************************************************************

}  // namespace elastica
