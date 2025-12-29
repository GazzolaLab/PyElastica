#pragma once

#include "../../api.h"  // For BlockRodSystem
#include "physics/linear_spring_dashpot.h"
#include "physics/no_interaction.h"
#include "course_detection/hash_grid.h"
#include "fine_detection/max_contacts.h"
#include "batching/union_find.h"
#include "types.h"
#include <variant>
#include <unordered_map>
#include <utility>  // For std::pair
#include <functional>  // For std::hash

namespace elasticapp/environment {
namespace collision {

// Hash function for std::pair<std::size_t, std::size_t> (needed for unordered_map)
// C++11 doesn't provide std::hash for pairs by default
struct PairHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
        // Combine hashes of both elements
        // Using a simple hash combination (can be improved with boost::hash_combine)
        return std::hash<std::size_t>()(p.first) ^ (std::hash<std::size_t>()(p.second) << 1);
    }
};

/**
 * CollisionSystem class for collision detection and resolution.
 *
 * This class orchestrates the collision detection and resolution pipeline:
 * 1. Data extraction from Block
 * 2. Coarse detection (HashGrid)
 * 3. Fine detection (MaxContacts)
 * 4. Contact batching (UnionFind)
 * 5. Contact resolution (PhysicsModel - runtime selectable)
 * 6. Force application to ExternalForces
 *
 * Uses CRTP (Curiously Recurring Template Pattern) to mix in policy classes
 * for coarse detection, fine detection, and batching strategies.
 *
 * Physics models are stored in a std::variant for runtime selection without
 * virtual function overhead.
 *
 * @tparam CoarseDetectionPolicy Policy for coarse collision detection
 * @tparam FineDetectionPolicy Policy for fine collision detection
 * @tparam BatchingPolicy Policy for contact batching
 */
template<typename CoarseDetectionPolicy, typename FineDetectionPolicy, typename BatchingPolicy>
class CollisionSystem : public CoarseDetectionPolicy, public FineDetectionPolicy, public BatchingPolicy {
public:
    /**
     * Variant type for physics models.
     * Currently supports LinearSpringDashpot, but can be extended with additional models.
     * This syntax is used to avoid using virtual functions and CRTP hierarchies.
     * Different structures can be considered once a more compositional approach to the physics model is needed.
     *
     * To add a new physics model:
     * 1. Implement the model class with a compute_force(Contact&, double&) method
     * 2. Add it to this variant: using PhysicsModel = std::variant<...>;
     */
    using PhysicsModel = std::variant<physics::LinearSpringDashpot, physics::NoInteraction>;

    /**
     * Constructor takes a physics model and detection frequency.
     *
     * @param model The collision physics model (e.g., LinearSpringDashpot instance)
     * @param detect_every Perform coarse detection every N steps (default: 1, meaning every step)
     */
    CollisionSystem(const PhysicsModel& model, std::size_t detect_every = 1) : model_(model), detect_every_(detect_every), step_counter_(0) {}

    /**
     * Resolve collisions for a BlockRodSystem.
     *
     * This method performs the full collision detection and resolution pipeline.
     */
    void resolve(BlockRodSystem& system);

    /**
     * Get the contact cache (non-const).
     *
     * @return Reference to the contact cache vector
     */
    std::vector<std::pair<std::size_t, std::size_t>>& contact_cache() {
        return contact_cache_;
    }

    /**
     * Get the contact cache (const).
     *
     * @return Const reference to the contact cache vector
     */
    const std::vector<std::pair<std::size_t, std::size_t>>& contact_cache() const {
        return contact_cache_;
    }

    /**
     * Get the detect_every parameter.
     *
     * @return The number of steps between coarse detection calls
     */
    std::size_t detect_every() const {
        return detect_every_;
    }

    /**
     * Set the detect_every parameter.
     *
     * @param detect_every Number of steps between coarse detection calls
     */
    void set_detect_every(std::size_t detect_every) {
        detect_every_ = detect_every;
    }

private:
    PhysicsModel model_;

    /**
     * Number of steps between coarse detection calls.
     *
     * Coarse detection (HashGrid) is only performed every detect_every_ steps.
     * Fine detection, batching, and resolution still occur every step using
     * the cached contact pairs from the last coarse detection.
     *
     * Default: 1 (detect every step)
     */
    std::size_t detect_every_;

    /**
     * Current step counter for tracking when to perform coarse detection.
     */
    mutable std::size_t step_counter_;

    /**
     * Contact cache for storing candidate pairs from coarse detection.
     *
     * This cache stores the candidate pairs identified by coarse detection,
     * allowing fine detection to reuse them without re-running coarse detection.
     * The cache is cleared when coarse detection is performed.
     */
    mutable std::vector<std::pair<std::size_t, std::size_t>> contact_cache_;

    /**
     * Previous contacts for tracking tangential displacement over time.
     *
     * Stores contacts from the previous resolve() call, keyed by normalized node pair.
     * This allows accumulation of tangential displacement between calls.
     * Key: Normalized node pair (min(node1, node2), max(node1, node2))
     * Value: Contact from previous call with accumulated displacement
     */
    std::unordered_map<std::pair<std::size_t, std::size_t>, Contact, PairHash> previous_contacts_;

    /**
     * Normalize a node pair to ensure consistent ordering.
     *
     * Returns (min(n1, n2), max(n1, n2)) to ensure the same pair
     * always maps to the same key regardless of order.
     *
     * @param n1 First node index
     * @param n2 Second node index
     * @return Normalized pair (min, max)
     */
    static std::pair<std::size_t, std::size_t> normalize_pair(
        std::size_t n1,
        std::size_t n2
    ) {
        return n1 < n2 ? std::make_pair(n1, n2) : std::make_pair(n2, n1);
    }
};

// Template method implementations
template<typename CoarseDetectionPolicy, typename FineDetectionPolicy, typename BatchingPolicy>
void CollisionSystem<CoarseDetectionPolicy, FineDetectionPolicy, BatchingPolicy>::resolve(
    BlockRodSystem& system) {

    // Get data from BlockRodSystem
    auto&& positions = system.template get<system::cosserat_rod::Position>();
    auto&& velocities = system.template get<system::cosserat_rod::Velocity>();
    auto&& radii_elem = system.template get<system::cosserat_rod::Radius>();
    auto&& external_forces = system.template get<system::cosserat_rod::ExternalForces>();

    const std::size_t n_nodes = positions.cols();
    const std::size_t n_elems = radii_elem.cols();

    // Convert element radii to node radii (interpolate: node i gets radius from element floor(i/2))
    // For nodes: node 0 and 1 share element 0, node 2 and 3 share element 1, etc.
    Eigen::MatrixXd radii(1, n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        std::size_t elem_idx = (i < n_elems) ? i : (n_elems - 1);
        if (i > 0 && i % 2 == 0 && elem_idx < n_elems - 1) {
            elem_idx = i / 2;
        } else if (i > 0) {
            elem_idx = std::min(i / 2, n_elems - 1);
        }
        radii(0, i) = radii_elem(0, elem_idx);
    }

    // Step 1: Coarse detection (only every detect_every_ steps)
    bool should_detect = (step_counter_ % detect_every_ == 0);

    if (should_detect) {
        // Clear contact cache and perform coarse detection
        contact_cache_.clear();
        // Call coarse detection via CRTP (inherited from CoarseDetectionPolicy)
        contact_cache_ = static_cast<CoarseDetectionPolicy&>(*this).detect(positions, radii);
    }
    // Otherwise, reuse cached contact pairs from previous detection

    // Step 2: Fine detection (always performed, uses cached pairs if detection was skipped)
    std::vector<Contact> contacts;
    if (!contact_cache_.empty()) {
        // Extract physics parameters from model and call fine detection via CRTP
        std::visit([&](const auto& physics_model) {
            contacts = static_cast<FineDetectionPolicy&>(*this).detect(contact_cache_, positions, velocities, radii, physics_model);
        }, model_);
    }

    // Step 3: Contact batching (via CRTP)
    std::vector<std::vector<std::size_t>> batches = static_cast<BatchingPolicy&>(*this).batch(contacts);

    // Step 4 & 5: Contact resolution and force application
    for (const auto& batch : batches) {
        for (std::size_t contact_idx : batch) {
            Contact& contact = contacts[contact_idx];

            // Update tangential displacement from previous contact if exists
            auto pair_key = normalize_pair(contact.node1_idx, contact.node2_idx);
            auto prev_it = previous_contacts_.find(pair_key);
            if (prev_it != previous_contacts_.end()) {
                // Accumulate tangential displacement
                const Eigen::Vector3d& prev_pos = prev_it->second.position;
                const Eigen::Vector3d& curr_pos = contact.position;
                const Eigen::Vector3d& prev_normal = prev_it->second.normal;

                // Project displacement onto tangent plane
                Eigen::Vector3d displacement = curr_pos - prev_pos;
                Eigen::Vector3d normal_displacement = displacement.dot(prev_normal) * prev_normal;
                contact.tangential_displacement = prev_it->second.tangential_displacement +
                                                 (displacement - normal_displacement);
            }

            // Compute force using physics model
            double penetration_depth;
            Eigen::Vector3d force;
            std::visit([&](const auto& physics_model) {
                force = physics_model.compute_force(contact, penetration_depth);
            }, model_);

            // Apply forces to external_forces
            if (penetration_depth > 0.0) {
                // Force on node1
                external_forces(0, contact.node1_idx) += force(0);
                external_forces(1, contact.node1_idx) += force(1);
                external_forces(2, contact.node1_idx) += force(2);

                // Force on node2 (opposite direction)
                external_forces(0, contact.node2_idx) -= force(0);
                external_forces(1, contact.node2_idx) -= force(1);
                external_forces(2, contact.node2_idx) -= force(2);
            }

            // Store contact for next iteration
            previous_contacts_[pair_key] = contact;
        }
    }

    // Increment step counter
    step_counter_++;
}

} // namespace collision
} // namespace elasticapp/environment
