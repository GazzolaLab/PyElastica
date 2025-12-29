#include "no_interaction.h"
#include "../types.h"

namespace elasticapp {
namespace collision {
namespace physics {

inline Eigen::Vector3d NoInteraction::compute_force(
    const Contact& contact,
    double& penetration_depth
) const {
    // No interaction - always return zero force
    penetration_depth = 0.0;
    return Eigen::Vector3d::Zero();
}

} // namespace physics
} // namespace collision
} // namespace elasticapp
