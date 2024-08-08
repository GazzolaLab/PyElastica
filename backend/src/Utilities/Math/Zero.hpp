#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <blaze/math/Accuracy.h>
#include <blaze/math/expressions/DVecNormExpr.h>

#include "Utilities/Math/Vec3.hpp"

namespace elastica {

  //****************************************************************************
  /*!\brief Checks for a zero number
   */
  inline auto is_zero(real_t real_number) noexcept -> bool {
    return ::blaze::isZero(real_number);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Checks for a zero vector
   */
  inline auto is_zero(const Vec3& vec) noexcept -> bool {
    return ::blaze::isZero(vec);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Checks for a zero length vector
   *
   * Checks if the length of a vector is zero or as close to zero that it
   * can not be distinguished form zero
   */
  inline auto is_zero_length(const Vec3& vec) noexcept -> bool {
    //          return vec.sqrLength() < Limits<real_t>::fpuAccuracy();
    return ::blaze::sqrLength(vec) < ::blaze::accuracy;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns a zero vector
   */
  inline auto zero3() noexcept -> Vec3 {
    return Vec3{static_cast<real_t>(0.0), static_cast<real_t>(0.0),
                static_cast<real_t>(0.0)};
  }
  //****************************************************************************

}  // namespace elastica
