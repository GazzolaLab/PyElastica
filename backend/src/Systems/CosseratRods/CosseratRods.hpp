#pragma once

//******************************************************************************
// Includes
//******************************************************************************
////
#include "Systems/CosseratRods/Protocols.hpp"
#include "Systems/CosseratRods/Tags.hpp"
#include "Systems/CosseratRods/TypeTraits.hpp"
#include "Systems/CosseratRods/Types.hpp"
////
//// include implementations now
#include "Systems/CosseratRods/Block.hpp"
#include "Systems/CosseratRods/BlockInitializer.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/BlockView.hpp"
#include "Systems/CosseratRods/Components.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
#include "Systems/CosseratRods/CosseratRodTraits.hpp"
#include "Systems/CosseratRods/Initializers.hpp"
///
// include aliased classes now
// Developer note:
// These can be instantiated in C++ files to massively speed up compilation,
// but we don't do that here to give full flexibility to the user for
// initializing it. In our applications, we can do an instantiation if needed,
// according to the initializers we use.
///
#include "Systems/CosseratRods/Aliases.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup cosserat_rod Cosserat rods
 * \ingroup systems
 * \brief Data, operations and helpers for manipulating Cosserat rods
 */
//******************************************************************************

namespace elastica {

  //****************************************************************************
  /*!\brief Cosserat rod data-structures and routines
  // \ingroup cosserat_rod
  */
  namespace cosserat_rod {}
  //****************************************************************************

}  // namespace elastica
