#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/Geometry/detail/Types.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //////////////////////////////////////////////////////////////////////////
      //
      // Forward declarations of geometry types in the interface
      //
      //////////////////////////////////////////////////////////////////////////

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      template <typename /* Cosserat Rod Traits */, typename /* Block */>
      class WithCircularCosseratRod;

      template <typename /* Cosserat Rod Traits */, typename /* Block */>
      class WithSquareCosseratRod;
      /*! \endcond */
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
