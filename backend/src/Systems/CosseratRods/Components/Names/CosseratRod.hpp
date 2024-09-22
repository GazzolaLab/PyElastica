#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/common/Components/NameAdapter.hpp"
//
#include <string>

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Provides a name to CosseratRod
       * \ingroup cosserat_rod_component
       */
      struct CosseratRodName {
       public:
        //**********************************************************************
        /*!\brief Human-readable name of the current plugin and all derivates
         *
         * \note This is intended to work with pretty_type::name<>
         */
        static std::string name() { return "CosseratRod"; }
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
