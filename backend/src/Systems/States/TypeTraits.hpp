#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///
#include "Systems/States/Types.hpp"
///
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Systems/States/TypeTraits/HasOrder.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"
#include "Systems/States/TypeTraits/IsPrimitive.hpp"
#include "Systems/States/TypeTraits/IsScalar.hpp"
#include "Systems/States/TypeTraits/IsVectorized.hpp"
#include "Systems/States/TypeTraits/SupportsVectorizedOperations.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup states_tt Type Traits
 * \ingroup states
 * \brief Type traits belonging to states
 */
//******************************************************************************

namespace elastica {

  namespace states {

    //**************************************************************************
    /*!\brief Namespace for TypeTraits for States
    // \ingroup states_tt
    */
    namespace tt {}
    //**************************************************************************

  }  // namespace states

}  // namespace elastica
