#pragma once

#include "../types.h"
#include <vector>
#include <Eigen/Dense>
#include <cmath>

#include <omp.h>

namespace elasticapp::environment {
namespace collision {

/**
 * MaxContacts fine collision detection policy.
 *
 * Finds points of maximum overlap between potentially colliding pairs.
 * This is the default fine detection algorithm in the reference implementation.
 *
 * Based on the Elastica MaxContacts implementation, which creates the maximal
 * number of contact points necessary to handle collisions between rigid bodies
 * physically accurately.
 *
 * For sphere-sphere collisions (nodes), a single contact point is generated
 * at the point of maximum overlap along the line connecting the two centers.
 *
 * This is a CRTP policy class that will be mixed into CollisionSystem.
 */
class MaxContacts {
public:
    /**
     * Contact threshold for determining if two objects are in contact.
     *
     * Objects are considered in contact if their separation distance is less
     * than this threshold. This allows contacts even when objects are slightly
     * separated, which is important for stability in discrete time simulations.
     *
     * Value based on typical Elastica implementations (typically 1e-6 to 1e-9).
     */
    static constexpr double contact_threshold = 1e-8;

    /**
     * Default constructor.
     */
    MaxContacts() = default;

    /**
     * Destructor.
     */
    ~MaxContacts() = default;

    /**
     * Perform fine collision detection on candidate pairs.
     *
     * Performs precise intersection tests and finds points of maximum overlap.
     * Returns detailed contact information for confirmed collisions.
     *
     * Algorithm (for sphere-sphere):
     * 1. Compute normal vector from node2 to node1: normal = pos1 - pos2
     * 2. Compute separation distance: dist = length(normal) - r1 - r2
     *    - dist < 0: penetrating (overlapping)
     *    - dist > 0: separated
     * 3. If dist < contact_threshold, create contact:
     *    - Normalize normal vector
     *    - Contact position: pos2 + normal * (r2 + 0.5 * dist)
     *    - Contact normal points from node2 to node1
     *    - Distance stored as dist (negative when penetrating)
     *
     * Material properties (stiffness, damping, friction) are taken from
     * the physics model and stored in each Contact for later force computation.
     *
     * @param candidate_pairs Pairs of node indices from coarse detection
     * @param positions Node positions (3 x n_nodes)
     * @param velocities Node velocities (3 x n_nodes)
     * @param radii Node radii (1 x n_nodes)
     * @param physics_model Physics model containing material properties
     * @return Vector of Contact objects for confirmed collisions
     */
    template<typename PhysicsModel>
    std::vector<Contact> detect(
        const std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
        const Eigen::MatrixXd& positions,
        const Eigen::MatrixXd& velocities,
        const Eigen::MatrixXd& radii,
        const PhysicsModel& physics_model
    ) const {
        std::vector<Contact> contacts;
        contacts.reserve(candidate_pairs.size());  // Reserve space for potential contacts

        // Extract material properties from physics model
        const double stiffness = physics_model.k_normal;
        const double normal_damping = physics_model.eta_normal;
        const double tangential_damping = physics_model.eta_tangential;
        const double friction = physics_model.friction;

        // Process each candidate pair (parallelized with OpenMP)
        const std::size_t n_pairs = candidate_pairs.size();

        // Thread-local storage for contacts (each thread collects its own contacts)
        std::vector<std::vector<Contact>> thread_contacts(omp_get_max_threads());

        #pragma omp parallel for
        for (std::size_t idx = 0; idx < n_pairs; ++idx) {
            const auto& pair = candidate_pairs[idx];
            const std::size_t i1 = pair.first;
            const std::size_t i2 = pair.second;

            // Get node positions
            const Eigen::Vector3d pos1 = positions.col(i1);
            const Eigen::Vector3d pos2 = positions.col(i2);

            // Get node velocities
            const Eigen::Vector3d vel1 = velocities.col(i1);
            const Eigen::Vector3d vel2 = velocities.col(i2);

            // Get node radii
            const double r1 = std::abs(radii(i1));
            const double r2 = std::abs(radii(i2));

            // Detect sphere-sphere collision
            auto pair_contacts = detect_sphere_sphere(
                i1, i2,
                pos1, pos2,
                vel1, vel2,
                r1, r2,
                stiffness,
                normal_damping,
                tangential_damping,
                friction
            );

            // Add contacts to thread-local storage
            const int thread_id = omp_get_thread_num();
            thread_contacts[thread_id].insert(
                thread_contacts[thread_id].end(),
                pair_contacts.begin(),
                pair_contacts.end()
            );
        }

        // Combine thread-local contacts into final result
        for (const auto& thread_contact_vec : thread_contacts) {
            contacts.insert(contacts.end(), thread_contact_vec.begin(), thread_contact_vec.end());
        }

        return contacts;
    }

private:
    /**
     * Detect collision between two spheres (nodes).
     *
     * Implements the sphere-sphere collision detection algorithm from the
     * reference implementation.
     *
     * @param node1_idx Index of first node
     * @param node2_idx Index of second node
     * @param pos1 Position of first node
     * @param pos2 Position of second node
     * @param vel1 Velocity of first node
     * @param vel2 Velocity of second node
     * @param r1 Radius of first node
     * @param r2 Radius of second node
     * @param stiffness Contact stiffness
     * @param normal_damping Normal damping coefficient
     * @param tangential_damping Tangential damping coefficient
     * @param friction Friction coefficient
     * @return Contact object if collision detected, or empty optional
     */
    std::vector<Contact> detect_sphere_sphere(
        std::size_t node1_idx,
        std::size_t node2_idx,
        const Eigen::Vector3d& pos1,
        const Eigen::Vector3d& pos2,
        const Eigen::Vector3d& vel1,
        const Eigen::Vector3d& vel2,
        double r1,
        double r2,
        double stiffness,
        double normal_damping,
        double tangential_damping,
        double friction
    ) const;
};

} // namespace collision
} // namespace elasticapp::environment
