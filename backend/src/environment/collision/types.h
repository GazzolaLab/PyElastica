#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace elasticapp::environment {
namespace collision {

/**
 * Contact information between two nodes.
 *
 * Represents a collision contact point between two nodes (spheres)
 * in the collision detection system.
 */
struct Contact {
    // Node indices in the block
    std::size_t node1_idx;  // Index of first node
    std::size_t node2_idx;  // Index of second node

    // Contact geometry
    Eigen::Vector3d position;  // Contact point position (world coordinates)
    Eigen::Vector3d normal;     // Contact normal (unit vector, pointing from node2 to node1)
    double distance;            // Signed distance: negative when penetrating (overlap)

    // Node velocities at contact point
    Eigen::Vector3d velocity1;  // Velocity of node1
    Eigen::Vector3d velocity2;  // Velocity of node2

    // Physics parameters (from material properties)
    double stiffness;           // Contact stiffness (k_normal)
    double normal_damping;      // Normal damping coefficient (eta_normal)
    double tangential_damping;  // Tangential damping coefficient (eta_tangential)
    double friction;            // Static friction coefficient (mu)

    // Tangential displacement tracking (for spring-dashpot model)
    Eigen::Vector3d tangential_displacement;  // Accumulated tangential displacement

    // Contact index within batch
    std::size_t index;  // Index within batch (default: 0)

    /**
     * Default constructor.
     */
    Contact() : node1_idx(0), node2_idx(0),
                position(Eigen::Vector3d::Zero()),
                normal(Eigen::Vector3d::Zero()),
                distance(0.0),
                velocity1(Eigen::Vector3d::Zero()),
                velocity2(Eigen::Vector3d::Zero()),
                stiffness(1.0),
                normal_damping(0.1),
                tangential_damping(0.1),
                friction(0.5),
                tangential_displacement(Eigen::Vector3d::Zero()),
                index(0) {}

    /**
     * Constructor with all parameters.
     */
    Contact(
        std::size_t n1, std::size_t n2,
        const Eigen::Vector3d& pos,
        const Eigen::Vector3d& n,
        double dist,
        const Eigen::Vector3d& v1,
        const Eigen::Vector3d& v2,
        double k, double eta_n, double eta_t, double mu
    ) : node1_idx(n1), node2_idx(n2),
        position(pos), normal(n), distance(dist),
        velocity1(v1), velocity2(v2),
        stiffness(k), normal_damping(eta_n),
        tangential_damping(eta_t), friction(mu),
        tangential_displacement(Eigen::Vector3d::Zero()),
        index(0) {}

    /**
     * Set the contact index within its batch.
     *
     * @param idx The index to set
     */
    void set_index(std::size_t idx) {
        index = idx;
    }
};

} // namespace collision
} // namespace elasticapp::environment
