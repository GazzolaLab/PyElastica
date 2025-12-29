#include "linear_spring_dashpot.h"
#include "../types.h"
#include <algorithm>
#include <cmath>

namespace elasticapp {
namespace collision {
namespace physics {

inline Eigen::Vector3d LinearSpringDashpot::compute_force(
    const Contact& contact,
    double& penetration_depth
) const {
    // Check if contact is penetrating
    if (!is_penetrating(contact)) {
        penetration_depth = 0.0;
        return Eigen::Vector3d::Zero();
    }

    // Penetration depth: positive when penetrating
    // contact.distance is negative when penetrating (overlap)
    // Matching CONTEXT: real_t delta(-contact.get_distance());
    penetration_depth = -contact.distance;

    // Get contact normal (unit vector pointing from body2 to body1)
    const Eigen::Vector3d& normal = contact.normal;

    // Calculate relative velocity matching CONTEXT/COLLISION.md documentation:
    // tangential_velocity = (other_elem_velocity - this_elem_velocity) - normal_velocity * normal_direction
    // where other_elem is node2 and this_elem is node1
    // So: relative_velocity = velocity2 - velocity1
    const Eigen::Vector3d rel_velocity = contact.velocity2 - contact.velocity1;

    // Normal component of relative velocity
    // Matching CONTEXT/COLLISION.md: normal_velocity = (other_elem_velocity - this_elem_velocity).dot(normal_direction)
    const double rel_vel_normal = rel_velocity.dot(normal);

    // Tangential component of relative velocity
    // Matching CONTEXT/COLLISION.md: tangential_velocity = (other_elem_velocity - this_elem_velocity) - normal_velocity * normal_direction
    const Eigen::Vector3d rel_vel_tangential = rel_velocity - rel_vel_normal * normal;

    // Normal force magnitude (ReLU operation - no negative forces)
    // Matching CONTEXT/COLLISION.md: normal_force = k_normal * penetration + eta_normal * normal_velocity
    // Use physics parameters from LinearSpringDashpot instance, not from contact
    // (Contact may have material properties, but we use the model's parameters)
    const double f_normal_mag = std::max(0.0,
        k_normal * penetration_depth +
        eta_normal * rel_vel_normal
    );

    // Normal force vector
    // Matching CONTEXT: Vec3 fN(fNabs * contact.get_normal());
    const Eigen::Vector3d f_normal = f_normal_mag * normal;

    // Tangential force calculation (full spring-dashpot model)
    // Matching CONTEXT/COLLISION.md exactly:
    // auto tangential_force = -k_tangential * tangential_displacement_vec;
    // tangential_force += -eta_tangential * tangential_velocity;
    // This enables positive k_tangential for full spring-dashpot friction model
    Eigen::Vector3d f_tangential = -k_tangential * contact.tangential_displacement
                                   - eta_tangential * rel_vel_tangential;

    // Apply Coulomb friction limit
    // Matching reference: regularized_tangential_force_mag = min(tangential_force_mag, mu_static * normal_force)
    const double f_tangential_mag = f_tangential.norm();
    const double regularized_f_tangential_mag = std::min(
        f_tangential_mag,
        friction * f_normal_mag
    );

    // Normalize tangential force if magnitude is significant
    // Matching reference: if (tangential_force_mag > 1e-12) {
    //     tangential_force *= regularized_tangential_force_mag / tangential_force_mag;
    // }
    if (f_tangential_mag > 1e-12) {
        f_tangential *= regularized_f_tangential_mag / f_tangential_mag;
    } else {
        f_tangential = Eigen::Vector3d::Zero();
    }

    // Net force (to be applied to body1, body2 gets -net_force)
    // Matching CONTEXT: Vec3 net_force = (fN + fT);
    return f_normal + f_tangential;
}

inline bool LinearSpringDashpot::is_penetrating(const Contact& contact) {
    // Contact is penetrating if distance < 0 (overlap)
    return contact.distance < 0.0;
}

} // namespace physics
} // namespace collision
} // namespace elasticapp
