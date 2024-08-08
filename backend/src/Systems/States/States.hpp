#pragma once

//******************************************************************************
// Includes
//******************************************************************************

// Types always first
#include "Systems/States/Types.hpp"
//
#include "Systems/States/Expressions/SE3.hpp"
#include "Systems/States/Expressions/SO3.hpp"
#include "Systems/States/Expressions/States.hpp"
#include "Systems/States/TypeTraits.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup states States
 * \ingroup systems
 * \brief Handles temporally evolving states
 *
 * The states module contains a generic interface for handling a collection of
 * temporally evolving states in \elastica, such as the positions of different
 * Lagrangian entities (such as rods, rigid bodies etc.). Its equipped with a
 * simple expression-template system to ensure correct semantics while
 * temporally integrating states, whether its in SE3, SO3 groups or a collection
 * of SE3 and SO3 groups.
 */
//******************************************************************************

namespace elastica {

  //****************************************************************************
  /*!\brief State manipulation and expressions
  // \ingroup states
  */
  namespace states {}
  //****************************************************************************

}  // namespace elastica
