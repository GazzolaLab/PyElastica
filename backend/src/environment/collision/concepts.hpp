#pragma once

#include <concepts>
#include <vector>
#include <utility>
#include <cstddef>
#include <Eigen/Dense>
#include "types.h"
#include "physics/linear_spring_dashpot.h"

namespace elasticapp::environment {
namespace collision {

/**
 * Concept for CoarseDetectionPolicy.
 *
 * Coarse detection policies perform broad-phase collision detection to identify
 * candidate contact pairs. They should be fast but may produce false positives
 * that will be filtered by fine detection.
 *
 * Requirements:
 * - Must have a detect() method that takes positions and radii matrices
 * - Must return a vector of candidate node pairs (indices)
 * - Positions and radii are Eigen::MatrixXd
 */
template<typename T>
concept CoarseDetectionPolicy = requires(
    T policy,
    const Eigen::MatrixXd& positions,
    const Eigen::MatrixXd& radii
) {
    // Must have a detect method that takes positions and radii
    { policy.detect(positions, radii) } -> std::convertible_to<std::vector<std::pair<std::size_t, std::size_t>>>;
};

/**
 * Concept for FineDetectionPolicy.
 *
 * Fine detection policies perform narrow-phase collision detection on candidate
 * pairs from coarse detection. They determine actual contact geometry and properties.
 *
 * Requirements:
 * - Must have a detect() method that takes candidate pairs, positions, velocities,
 *   radii, and a physics model (any type)
 * - Must return a vector of Contact objects with full contact information
 * - Positions, velocities, and radii are Eigen::MatrixXd
 */
template<typename T, typename PhysicsModel>
concept FineDetectionPolicyFor = requires(
    T policy,
    const std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
    const Eigen::MatrixXd& positions,
    const Eigen::MatrixXd& velocities,
    const Eigen::MatrixXd& radii,
    const PhysicsModel& physics_model
) {
    { policy.detect(candidate_pairs, positions, velocities, radii, physics_model) }
        -> std::convertible_to<std::vector<Contact>>;
};

/**
 * Concept for FineDetectionPolicy (unconstrained version).
 *
 * A policy satisfies FineDetectionPolicy if it can work with any physics model type.
 * This is checked by requiring it to work with at least one model type (LinearSpringDashpot).
 */
template<typename T>
concept FineDetectionPolicy = FineDetectionPolicyFor<T, physics::LinearSpringDashpot>;

/**
 * Concept for BatchingPolicy.
 *
 * Batching policies group contacts into batches for efficient parallel processing.
 * Contacts in the same batch can be processed independently.
 *
 * Requirements:
 * - Must have a batch() method that takes a vector of contacts
 * - Must return a vector of batches, where each batch is a vector of contact indices
 */
template<typename T>
concept BatchingPolicy = requires(
    T policy,
    std::vector<Contact>& contacts
) {
    // Must have a batch method that takes contacts and returns batches
    { policy.batch(contacts) } -> std::convertible_to<std::vector<std::vector<std::size_t>>>;
};

} // namespace collision
} // namespace elasticapp::environment
