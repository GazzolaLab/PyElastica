#pragma once

//******************************************************************************
// Includes
//******************************************************************************

namespace elastica {

  //============================================================================
  //
  //  PARALLEL CONFIGURATION
  //
  //============================================================================

#include "Configuration/Parallel.hpp"

//******************************************************************************
/*!\brief Compilation switch for TBB parallelization.
 * \ingroup parallel
 *
 * This compilation switch enables/disables TBB parallelization. In case TBB is
 * enabled during compilation, \elastica attempts to parallelize computations,
 * if requested by the configuration. In case no parallelization is enabled,
 * all computations are performed on a single compute core, even if the
 * configuration requests it
 */
#if defined(ELASTICA_USE_SHARED_MEMORY_PARALLELIZATION) && \
    defined(ELASTICA_USE_TBB)
#define ELASTICA_TBB_PARALLEL_MODE 1
#else
#define ELASTICA_TBB_PARALLEL_MODE 0
#endif
  //****************************************************************************

}  // namespace elastica
