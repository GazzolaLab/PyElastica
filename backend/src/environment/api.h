#pragma once

#include "../api.h"  // For BlockRodSystem
#include "collision/collision_system.h"
#include "collision/course_detection/hash_grid.h"
#include "collision/fine_detection/max_contacts.h"
#include "collision/batching/union_find.h"

namespace elasticapp/environment {

// Re-export CollisionSystem from collision namespace for convenience
namespace collision {
    using DefaultCollisionSystem = CollisionSystem<HashGrid, MaxContacts, UnionFind>;
}

} // namespace elasticapp/environment
