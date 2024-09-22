//=================================================================================================
/*!
 *  \file pe/math/shims/Equal.h
 *  \brief Header file for the equal shim
 *
 *  Copyright (C) 2009 Klaus Iglberger
 *
 *  This file is part of pe.
 *
 *  pe is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 *  pe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 * pe. If not, see <http://www.gnu.org/licenses/>.
 */
//=================================================================================================

#pragma once

//#include <cmath>

// Can use blaze's equal without any performance hits

/*
namespace elastica {
  constexpr bool relaxed = true;

  template <bool RF>  // Relaxation flag
  inline bool equal(float a, float b) {
    using std::max;
    if (RF == relaxed) {
      const float acc(static_cast<float>(1e-6));
      return (std::fabs(a - b) <= max(acc, acc * std::fabs(a)));
    } else {
      return a == b;
    }
  }

  template <bool RF>  // Relaxation flag
  inline bool equal(double a, double b) {
    using std::max;
    if (RF == relaxed) {
      const double acc(static_cast<double>(1e-8));
      return (std::fabs(a - b) <= max(acc, acc * std::fabs(a)));
    } else {
      return a == b;
    }
  }

  template <typename T1,  // Type of the left-hand side value/object
            typename T2>  // Type of the right-hand side value/object
  inline constexpr bool equal(const T1& a, const T2& b) {
    return equal<relaxed>(a, b);
  }

}  // namespace elastica
 */

#include <blaze/math/shims/Equal.h>

namespace elastica {

  //*************************************************************************************************
  /*!\brief Import of the blaze::equal() function into the elastica namespace.
  // \ingroup math_shims
  */
  using blaze::equal;
  //*************************************************************************************************

}  // namespace elastica
