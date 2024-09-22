#pragma once

#include <type_traits>

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Marks a rank of a tensor
    // \ingroup cosserat_rod
    */
    template <unsigned int I>
    struct Rank : std::integral_constant<unsigned int, I> {};
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica
