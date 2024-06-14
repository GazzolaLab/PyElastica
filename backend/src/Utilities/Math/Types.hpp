//==============================================================================
/*!
//  \file
//  \brief Types belonging to the Math module
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <blaze/Forward.h>
//
#include "Utilities/DefineTypes.h"

namespace elastica {

  //****************************************************************************
  //! \brief 3D vector for use in \elastica interfaces
  //! \ingroup math
  using Vec3 = blaze::StaticVector<real_t, 3UL, blaze::columnVector,
                                   blaze::unaligned, blaze::unpadded>;
  //****************************************************************************

  //****************************************************************************
  //! \brief 3D matrix for use in \elastica interfaces
  //! \ingroup math
  using Rot3 = blaze::StaticMatrix<real_t, 3UL, 3UL, blaze::rowMajor,
                                   blaze::unaligned, blaze::unpadded>;
  //****************************************************************************

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations
  //
  //////////////////////////////////////////////////////////////////////////////

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  struct RotationConvention;
  /*! \endcond */
  //****************************************************************************
}  // namespace elastica
