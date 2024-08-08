#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdint>

namespace elastica {

  namespace cosserat_rod {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Optimization levels
     * \ingroup cosserat_rods
     *
     * Define level of optimization for the kernels employed within the
     * cosserat rod module
     */
    enum class OptimizationLevel : std::uint8_t { basic = 0, A = 1 };
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica
