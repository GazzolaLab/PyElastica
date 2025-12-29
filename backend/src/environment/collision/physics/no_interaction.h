#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include "../types.h"

namespace elasticapp::environment::collision {
namespace physics {

/**
 * NoInteraction physics model for testing purposes.
 *
 * This model returns zero force for all contacts, allowing collision detection
 * to be tested without applying any forces. Useful for:
 * - Testing collision detection algorithms
 * - Validating contact geometry
 * - Debugging collision pipeline without force effects
 */
struct NoInteraction {
    /**
     * Default constructor (no parameters needed).
     */
    NoInteraction() = default;

    /**
     * Compute collision forces for a contact.
     *
     * Always returns zero force regardless of contact state.
     *
     * @param contact The contact information (unused, but required for interface compatibility)
     * @param penetration_depth Output: penetration depth (set to 0.0)
     * @return Zero force vector
     */
    inline Eigen::Vector3d compute_force(
        const collision::Contact& contact,
        double& penetration_depth
    ) const;
};

} // namespace physics
} // namespace elasticapp::environment::collision
