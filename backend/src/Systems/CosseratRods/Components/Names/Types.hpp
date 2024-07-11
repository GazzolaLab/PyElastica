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
      struct CosseratRodWithoutDampingName;

      template <typename Traits, typename Block>
      using CosseratRodNameAdapted =
          ::elastica::detail::NameAdapter<Traits, Block, CosseratRodName>;

      template <typename Traits, typename Block>
      using CosseratRodWithoutDampingNameAdapted =
          ::elastica::detail::NameAdapter<Traits, Block,
                                          CosseratRodWithoutDampingName>;
      /*! \endcond */
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
