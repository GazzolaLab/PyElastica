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
      /*!\brief Provides a name to CosseratRodWithoutDamping
       * \ingroup cosserat_rod_component
       */
      struct CosseratRodWithoutDampingName {
       public:
        //**********************************************************************
        /*!\brief Human-readable name of the current plugin and all derivates
         *
         * \note This is intended to work with pretty_type::name<>
         */
        static std::string name() { return "CosseratRodWithoutDamping"; }
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
