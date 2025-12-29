#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include "../types.h"

namespace elasticapp::environment::collision {
namespace physics {

/**
 * Linear spring-dashpot collision physics model.
 *
 * Implements the Haff and Werner model for DEM collision response.
 * Computes normal (repulsive) and tangential (friction) forces based on:
 * - Normal force: F_n = max(0, k_n * delta + eta_n * v_n)
 * - Tangential force: F_t = -k_t * d_t - eta_t * v_t (regularized by Coulomb limit)
 *
 * where:
 * - k_n: normal spring constant (stiffness)
 * - k_t: tangential spring constant (default: 0.0 for velocity-only friction)
 * - eta_n: normal damping coefficient
 * - eta_t: tangential damping coefficient
 * - mu: friction coefficient
 * - delta: penetration depth
 * - d_t: tangential displacement (accumulated over time)
 * - v_n: normal relative velocity
 * - v_t: tangential relative velocity
 */
struct LinearSpringDashpot {
    // Physics parameters
    double k_normal;      // Normal spring constant
    double k_tangential;  // Tangential spring constant
    double eta_normal;    // Normal damping coefficient
    double eta_tangential; // Tangential damping coefficient
    double friction;      // Static friction coefficient

    /**
     * Constructor with physics parameters (3 parameters - uses eta_normal for tangential).
     *
     * @param k_normal Normal spring constant for repulsion
     * @param eta_normal Normal damping coefficient (also used for tangential)
     * @param friction Static friction coefficient
     */
    LinearSpringDashpot(
        double k_normal,
        double eta_normal,
        double friction
    ) : k_normal(k_normal),
        k_tangential(0.0),  // Default: no tangential spring (velocity-only friction)
        eta_normal(eta_normal),
        eta_tangential(eta_normal),  // Use same value for tangential
        friction(friction) {}

    /**
     * Constructor with physics parameters (4 parameters - explicit tangential damping).
     *
     * @param k_normal Normal spring constant for repulsion
     * @param eta_normal Normal damping coefficient
     * @param eta_tangential Tangential damping coefficient
     * @param friction Static friction coefficient
     */
    LinearSpringDashpot(
        double k_normal,
        double eta_normal,
        double eta_tangential,
        double friction
    ) : k_normal(k_normal),
        k_tangential(0.0),  // Default: no tangential spring (velocity-only friction)
        eta_normal(eta_normal),
        eta_tangential(eta_tangential),
        friction(friction) {}

    /**
     * Constructor with physics parameters (5 parameters - explicit tangential spring and damping).
     *
     * @param k_normal Normal spring constant for repulsion
     * @param eta_normal Normal damping coefficient
     * @param k_tangential Tangential spring constant
     * @param eta_tangential Tangential damping coefficient
     * @param friction Static friction coefficient
     */
    LinearSpringDashpot(
        double k_normal,
        double eta_normal,
        double k_tangential,
        double eta_tangential,
        double friction
    ) : k_normal(k_normal),
        k_tangential(k_tangential),
        eta_normal(eta_normal),
        eta_tangential(eta_tangential),
        friction(friction) {}

    /**
     * Compute collision forces for a contact.
     *
     * This method implements the LinearSpringDashpot force model:
     * 1. Calculates penetration depth
     * 2. Computes normal force (spring + damping)
     * 3. Computes tangential force (spring + damping, regularized by Coulomb limit)
     * 4. Returns the net force vector
     *
     * Tangential force formula:
     * F_t = -k_tangential * tangential_displacement - eta_tangential * tangential_velocity
     * Then regularized: F_t = min(|F_t|, mu * F_n) * normalize(F_t)
     *
     * @param contact The contact information (position, normal, distance, velocities, tangential_displacement, etc.)
     * @param penetration_depth Output: penetration depth (positive if penetrating)
     * @return Net force vector to apply to the first body (force on second body is -net_force)
     */
    inline Eigen::Vector3d compute_force(
        const collision::Contact& contact,
        double& penetration_depth
    ) const;

    /**
     * Check if contact is penetrating.
     *
     * @param contact The contact information
     * @return True if bodies are penetrating (distance < 0)
     */
    inline static bool is_penetrating(const collision::Contact& contact);
};

} // namespace physics
} // namespace elasticapp::environment::collision
