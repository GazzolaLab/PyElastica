#include "max_contacts.h"
#include <algorithm>
#include <cmath>

namespace elasticapp {
namespace collision {

std::vector<Contact> MaxContacts::detect_sphere_sphere(
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
) const {
    std::vector<Contact> contacts;

    // Ordering: Normal from node2 to node1 always (matching reference implementation)
    // This ensures consistent normal direction regardless of pair order
    Eigen::Vector3d normal = pos1 - pos2;
    const double normal_length = normal.norm();

    // Avoid division by zero for coincident nodes
    if (normal_length < 1e-12) {
        return contacts;  // Nodes are coincident, skip
    }

    // Compute separation distance: dist = length(normal) - r1 - r2
    // - dist < 0: penetrating (overlapping)
    // - dist > 0: separated
    // - dist = 0: touching
    const double dist = normal_length - r1 - r2;

    // Check if in contact (dist < contact_threshold)
    // This allows contacts even when slightly separated for stability
    if (dist < contact_threshold) {
        // Normalize the normal vector
        normal /= normal_length;

        // Compute contact position
        // Contact point lies on the line connecting centers, at the point of maximum overlap
        // Formula from reference: gPos = pos2 + normal * (r2 + 0.5 * dist)
        // When dist < 0 (penetrating), this places the contact point between the surfaces
        const double k = r2 + 0.5 * dist;
        const Eigen::Vector3d contact_pos = pos2 + normal * k;

        // Create contact
        // Note: distance is stored as dist (negative when penetrating, positive when separated)
        // The Contact struct expects distance to be negative when penetrating
        Contact contact(
            node1_idx, node2_idx,
            contact_pos,
            normal,
            dist,  // dist is already negative when penetrating
            vel1,
            vel2,
            stiffness,
            normal_damping,
            tangential_damping,
            friction
        );

        contacts.push_back(contact);
    }

    return contacts;
}

// Template implementation moved to header file (max_contacts.h) due to template nature

} // namespace collision
} // namespace elasticapp
