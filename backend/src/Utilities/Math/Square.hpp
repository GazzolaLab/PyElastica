//=================================================================================================
/*!
 *  \file pe/math/shims/Square.h
 *  \brief Header file for the square shim
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

#include "Utilities/ForceInline.hpp"

namespace elastica {

  //*************************************************************************************************
  /*!\brief Squaring the given value/object.
   * \ingroup math_shims
   *
   * \param a The value/object to be squared.
   * \return The result of the square operation.
   *
   * The square shim represents an abstract interface for squaring a
   * value/object of any given data type. For values of built-in data type this
   * results in a plain multiplication.
   */

  template <typename T>
  ELASTICA_ALWAYS_INLINE constexpr decltype(auto) sq(const T& a) {
    return a * a;
  }
}  // namespace elastica
