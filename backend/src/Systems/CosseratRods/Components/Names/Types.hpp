#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/common/Components/Types.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //////////////////////////////////////////////////////////////////////////
      //
      // Forward declarations of name types in the interface
      //
      //////////////////////////////////////////////////////////////////////////

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      struct CosseratRodName;

      template <typename Traits, typename Block>
      using CosseratRodNameAdapted =
          ::elastica::detail::NameAdapter<Traits, Block, CosseratRodName>;

      /*! \endcond */
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
