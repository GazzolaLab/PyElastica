#pragma once

//******************************************************************************
// Includes
//******************************************************************************

///
#include "Systems/CosseratRods/Traits/Types.hpp"
#include "Utilities/DefineTypes.h"
///
//#include "Systems/CosseratRods/Traits/DataTraits/Rank.hpp"
// Not used, since we directly use blaze's implementation of a vector
//#include "Systems/CosseratRods/Traits/DataType/ScalarTag.hpp"
///

#define ELASTICA_USE_BLAZE 1  // todo : from CMAKE

#if defined(ELASTICA_USE_BLAZE)
#include "DataType/BlazeBackend/DataTypeTraits.hpp"
#include "Operations/BlazeBackend/OpsTraits.hpp"
#include "Operations/BlazeBackend/OptimizationLevel.hpp"
#endif  // ELASTICA_USE_BLAZE

///
#include "Utilities/NonCreatable.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Traits implementing data-structures for Cosserat rods in \elastica
     * \ingroup cosserat_rod_traits
     *
     * \details
     * DataOpsTraits is the customization point for altering the data-structures
     * to be used within the Cosserat rod hierarchy implemented using @ref
     * blocks in \elastica. It defines (domain-specific) types corresponding to
     * a Cosserat rod (such as a matrix, vector etc.), and is intended for use
     * as a template parameter in CosseratRodTraits.
     *
     * \see elastica::cosserat_rod::CosseratRodTraits
     */
    struct DataOpsTraits : private NonCreatable {
      //**Type definitions******************************************************
      //! Real type
      using real_type = real_t;
      //! Type of index
      using index_type = std::size_t;
      // using OnRod = ScalarTag<real_type>;
      //! Type to track indices on a Cosserat rod, with shape (1, )
      using Index = VectorTag<index_type>;
      //! Type to track scalars on a Cosserat rod, with shape (1, )
      using Scalar = VectorTag<real_type>;
      //! Type to track vectors on a Cosserat rod, with shape (d)
      using Vector = MatrixTag<real_type>;
      //! Type to track matrices on a Cosserat rod, with shape (d, d)
      using Matrix = TensorTag<real_type>;
      //************************************************************************

      //**Operation definitions*************************************************
      using Operations = OpsTraits<OptimizationLevel::basic>;
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod
}  // namespace elastica
