#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/common/Components/Types.hpp"
//

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Tag representing unit cardinality
   * \ingroup systems
   *
   * \details
   * Indicates that one system of a plugin has only one Lagrangian degree
   * of freedom.
   */
  struct UnitCardinality {
    //! Index for unit cardinality
    static constexpr auto index() noexcept -> std::size_t { return 0UL; }
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Tag representing multiple cardinality
   * \ingroup systems
   *
   * \details
   * Indicates that one system of a plugin has multiple Lagrangian degrees
   * of freedom.
   *
   */
  struct MultipleCardinality {};
  //****************************************************************************

  //****************************************************************************
  /*!\brief Indicates unit cardinality of a plugin.
   * \ingroup systems
   *
   * \tparam Plugin A system plugin.
   * \see UnitCardinality
   */
  template <typename Plugin>
  struct HasUnitCardinality {
    //**Type definitions********************************************************
    //! Type of cardinality
    using cardinality = UnitCardinality;
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Indicates multiple cardinality of a plugin.
   * \ingroup systems
   *
   * \tparam Plugin A system plugin.
   * \see MultipleCardinality
   */
  template <typename Plugin>
  struct HasMultipleCardinality {
    //**Type definitions********************************************************
    //! Type of cardinality
    using cardinality = MultipleCardinality;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace elastica
