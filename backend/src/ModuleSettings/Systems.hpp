#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdlib>

namespace elastica {

  //============================================================================
  //
  //  SYSTEMS CONFIGURATION
  //
  //============================================================================

#include "Configuration/Systems.hpp"

  namespace detail {

    inline bool systems_warnings_enabled() /* noexcept*/ {
      // if set to 0 also return true
      if (const char* env_v = std::getenv(ENV_ELASTICA_NO_SYSTEM_WARN))
        return not bool(env_v[0]);
      else
        return true;
    }

  }  // namespace detail

}  // namespace elastica
