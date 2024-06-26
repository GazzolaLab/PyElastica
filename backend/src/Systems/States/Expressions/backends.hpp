#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///
#include "Systems/States/Expressions/backends/Declarations.hpp"
///

#define ELASTICA_USE_BLAZE 1  // todo : from CMAKE

#if defined(ELASTICA_USE_BLAZE)
#include "Systems/States/Expressions/backends/blaze.hpp"
#endif  // ELASTICA_USE_BLAZE
