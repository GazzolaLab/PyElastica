#pragma once

#include "block.h"
#include "cosserat_rod_system.h"
#include "operations.h"

namespace elasticapp {

// BlockRodSystem is a CRTP mix of CosseratRodSystem, Block, and CosseratRodOperations
// It combines System functionality with Block data storage and rod-specific operations
// This allows Block to have access to all CosseratRodSystem variables and operations
// Block<CosseratRodSystem, CosseratRodOperations> inherits from:
// - CosseratRodSystem (system variables and methods)
// - CosseratRodOperations<Block<...>> (rod-specific operations)
// So it has:
// - All CosseratRodSystem methods
// - All Block methods
// - All CosseratRodOperations methods (compute_internal_forces_and_torques, etc.)
// - Access to CosseratRodSystem::Variables for template metaprogramming
using BlockRodSystem = Block<CosseratRodSystem, CosseratRodOperations>;
} // namespace elasticapp
