#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdint>

namespace elastica {

  //**************************************************************************
  /*!\brief The rotation convention followed by \elastica
   * \ingroup utils
   *
   * \details
   * TODO
   */
  struct RotationConvention {
    //! unique ID for the normal direction
    static constexpr std::uint8_t Normal = 0U;
    //! unique ID for the binormal direction
    static constexpr std::uint8_t Binormal = 1U;
    //! unique ID for the tangent direction
    static constexpr std::uint8_t Tangent = 2U;
  };
  //****************************************************************************

}  // namespace elastica
