#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <blaze/math/expressions/DVecScalarMultExpr.h>
#include <cmath>

#include "Utilities/DefineTypes.h"
#include "Utilities/Thresholds.hpp"
//
#include "Utilities/Math/Length.hpp"
#include "Utilities/Math/Vec3.hpp"

namespace elastica {

  //****************************************************************************
  /*!\brief Normalizes a vector
   * \ingroup math
   *
   * \return normalized vector
   */
  using blaze::normalize;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Checks if the vector v is normalized
   * \ingroup math
   *
   * \return \a true if normalized, else false
   */
  inline auto is_normalized(Vec3 const& v) noexcept -> bool {
    return std::abs(elastica::sqrLength(v) - static_cast<real_t>(1.0)) <
           normal_threshold;
  }
  //****************************************************************************

}  // namespace elastica
