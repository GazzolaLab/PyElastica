#pragma once

#include <cmath>
#include <cstddef>
#include "cosserat_rod_system.h"
#include "cosserat_equations.h"
#include "traits.h"

// Include OpenMP headers if threading is enabled
#ifdef ELASTICAPP_USE_THREADING
#include <omp.h>
#endif

namespace elasticapp {

// Thread management utilities (only available when threading is enabled)
#ifdef ELASTICAPP_USE_THREADING
// Set the number of OpenMP threads to use
// This affects all subsequent parallel regions
// Args:
//   num_threads: Number of threads to use (0 = use OpenMP default, typically all CPU cores)
inline void set_num_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    // If num_threads is 0 or negative, OpenMP will use its default
}

// Get the current number of threads in the current parallel region
// Returns 1 if called outside a parallel region
inline int get_num_threads() {
    return omp_get_num_threads();
}

// Get the maximum number of threads that can be used
inline int get_max_threads() {
    return omp_get_max_threads();
}

// Get the current thread number (0 to num_threads-1)
// Returns 0 if called outside a parallel region
inline int get_thread_num() {
    return omp_get_thread_num();
}
#endif // ELASTICAPP_USE_THREADING

// Default empty operations class using CRTP pattern
// This class can be extended with operations that work on the derived Block type
//
// Example usage with custom operations:
//   template<typename Derived>
//   class MyOperations {
//   public:
//       void my_operation() {
//           auto& block = static_cast<Derived&>(*this);
//           // Access block members and perform operations
//       }
//   };
//
//   using MyBlock = Block<CosseratRodSystem, MyOperations>;
template<typename Derived>
class DefaultOperations {
public:
    // Access to the derived class
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

protected:
    // Protected constructor to prevent direct instantiation
    DefaultOperations() = default;
    ~DefaultOperations() = default;

    // Prevent copying/moving (can be enabled in derived class if needed)
    DefaultOperations(const DefaultOperations&) = default;
    DefaultOperations(DefaultOperations&&) = default;
    DefaultOperations& operator=(const DefaultOperations&) = default;
    DefaultOperations& operator=(DefaultOperations&&) = default;
};

// Alias for backward compatibility
template<typename Derived>
using Operations = DefaultOperations<Derived>;

// CosseratRodOperations class for Cosserat rod-specific operations
// This class provides operations for computing forces, torques, accelerations, etc.
template<typename Derived>
class CosseratRodOperations {
public:
    // Compute internal forces and torques
    void compute_internal_forces_and_torques(double time) {
        (void)time;  // Suppress unused parameter warning
        auto& block = static_cast<Derived&>(*this);
        // Compute internal forces and torques separately
        compute_geometry_from_state(block);
        compute_all_dilatations(block);
        compute_shear_stretch_strains(block);
        compute_internal_shear_stretch_stresses_from_model(block);
        compute_internal_forces(block);

        compute_bending_twist_strains(block);
        compute_internal_bending_twist_stresses_from_model(block);
        compute_dilatation_rate(block);
        compute_internal_torques(block);
    }

    // Update accelerations based on forces
    void update_accelerations(double time) {
        (void)time;  // Suppress unused parameter warning
        auto& block = static_cast<Derived&>(*this);

        // Get all required variables
        auto acceleration = block.template get<system::cosserat_rod::Acceleration>();
        auto internal_forces = block.template get<system::cosserat_rod::InternalForces>();
        auto external_forces = block.template get<system::cosserat_rod::ExternalForces>();
        auto mass = block.template get<system::cosserat_rod::Mass>();
        auto alpha = block.template get<system::cosserat_rod::Alpha>();
        auto inv_mass_second_moment_of_inertia = block.template get<system::cosserat_rod::InvMassSecondMomentOfInertia>();
        auto internal_torques = block.template get<system::cosserat_rod::InternalTorques>();
        auto external_torques = block.template get<system::cosserat_rod::ExternalTorques>();
        auto dilatation = block.template get<system::cosserat_rod::Dilatation>();

        // Update translational acceleration: a = (F_internal + F_external) / m
        // acceleration is OnNode, Vector (3 rows, n_nodes cols)
        const Eigen::Index n_nodes = acceleration.cols();
        for (Eigen::Index i = 0; i < 3; ++i) {
            for (Eigen::Index k = 0; k < n_nodes; ++k) {
                acceleration(i, k) = (internal_forces(i, k) + external_forces(i, k)) / mass(0, k);
            }
        }

        // Update angular acceleration: alpha = inv_J * (tau_internal + tau_external) * dilatation
        // First zero out alpha
        alpha.setZero();

        // inv_mass_second_moment_of_inertia is OnElement, Matrix (9 rows, n_elems cols)
        // Stored as 9xN where each column is a flattened 3x3 matrix: [I00, I01, I02, I10, I11, I12, I20, I21, I22]
        // alpha is OnElement, Vector (3 rows, n_elems cols)
        const Eigen::Index n_elems = alpha.cols();
        for (Eigen::Index i = 0; i < 3; ++i) {
            for (Eigen::Index j = 0; j < 3; ++j) {
                for (Eigen::Index k = 0; k < n_elems; ++k) {
                    // Map 3x3 matrix index (i, j) to flattened 9x1 index: i*3 + j
                    Eigen::Index inv_J_idx = i * 3 + j;
                    alpha(i, k) += inv_mass_second_moment_of_inertia(inv_J_idx, k)
                                 * (internal_torques(j, k) + external_torques(j, k))
                                 * dilatation(0, k);
                }
            }
        }
    }

    // Zero out external forces and torques
    // time parameter is included to match Python signature, but not used in implementation
    void zeroed_out_external_forces_and_torques(double time) {
        (void)time;  // Suppress unused parameter warning
        auto& block = static_cast<Derived&>(*this);
        auto external_forces = block.template get<system::cosserat_rod::ExternalForces>();
        auto external_torques = block.template get<system::cosserat_rod::ExternalTorques>();

        // Zero out all external forces (OnNode, Vector: 3 rows, n_nodes columns)
        // Use explicit loop to ensure values are set (matching Python implementation)
        const Eigen::Index n_nodes = external_forces.cols();
        for (Eigen::Index i = 0; i < 3; ++i) {
            for (Eigen::Index k = 0; k < n_nodes; ++k) {
                external_forces(i, k) = 0.0;
            }
        }

        // Zero out all external torques (OnElement, Vector: 3 rows, n_elems columns)
        const Eigen::Index n_elems = external_torques.cols();
        for (Eigen::Index i = 0; i < 3; ++i) {
            for (Eigen::Index k = 0; k < n_elems; ++k) {
                external_torques(i, k) = 0.0;
            }
        }
    }

    // Update kinematics (position, director) using velocity and omega
    // Equivalent to Python: update_kinematics(time, prefac)
    void update_kinematics(double prefac) {
        auto& block = static_cast<Derived&>(*this);

        // Get variable views from block using accessible type names
        auto position = block.template get<system::cosserat_rod::Position>();
        auto velocity = block.template get<system::cosserat_rod::Velocity>();
        auto director = block.template get<system::cosserat_rod::Director>();
        auto omega = block.template get<system::cosserat_rod::Omega>();

        // Update position: x += prefac * v
        // position is (3, n_nodes), velocity is (3, n_nodes)
        // Eigen operations are automatically vectorized if SIMD is enabled
        position += prefac * velocity;

        // Update director using rotation matrix from omega
        // director is stored as (9, n_elems) - flattened 3x3 matrices
        // omega is (3, n_elems)
        const std::size_t n_elems = director.cols();
        constexpr std::size_t dim = 3;

        // Parallelize loop if threading is enabled
        #ifdef ELASTICAPP_USE_THREADING
        #ifdef ELASTICAPP_NUM_THREADS
        #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS)
        #else
        #pragma omp parallel for
        #endif
        #endif
        for (std::size_t k = 0; k < n_elems; ++k) {
            // Compute scaled omega: prefac * omega
            double omega_x = prefac * omega(0, k);
            double omega_y = prefac * omega(1, k);
            double omega_z = prefac * omega(2, k);

            // Compute rotation angle (theta = ||omega||)
            double theta = std::sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);

            // If theta is very small, use identity rotation
            if (theta < 1e-14) {
                continue; // No rotation needed
            }

            // Normalize axis
            double inv_theta = 1.0 / theta;
            double ux = omega_x * inv_theta;
            double uy = omega_y * inv_theta;
            double uz = omega_z * inv_theta;

            // Precompute sin and cos
            double sin_theta = std::sin(theta);
            double cos_theta = std::cos(theta);
            double one_minus_cos = 1.0 - cos_theta;

            // Build rotation matrix R using Rodrigues formula
            // R = I + sin(theta) * [u]_× + (1 - cos(theta)) * [u]_×^2
            // where [u]_× is the skew-symmetric matrix of u

            // Rotation matrix elements
            double R00 = cos_theta + one_minus_cos * ux * ux;
            double R01 = one_minus_cos * ux * uy + sin_theta * uz;
            double R02 = one_minus_cos * ux * uz - sin_theta * uy;
            double R10 = one_minus_cos * ux * uy - sin_theta * uz;
            double R11 = cos_theta + one_minus_cos * uy * uy;
            double R12 = one_minus_cos * uy * uz + sin_theta * ux;
            double R20 = one_minus_cos * ux * uz + sin_theta * uy;
            double R21 = one_minus_cos * uy * uz - sin_theta * ux;
            double R22 = cos_theta + one_minus_cos * uz * uz;

            // Extract current director (3x3 matrix stored as 9 elements)
            // Storage order: [d00, d10, d20, d01, d11, d21, d02, d12, d22]
            // (column-major order for 3x3 matrix)
            double d00 = director(0, k);
            double d10 = director(1, k);
            double d20 = director(2, k);
            double d01 = director(3, k);
            double d11 = director(4, k);
            double d21 = director(5, k);
            double d02 = director(6, k);
            double d12 = director(7, k);
            double d22 = director(8, k);

            // Apply rotation: R @ director (matrix multiplication)
            // New director = R * old_director
            director(0, k) = R00 * d00 + R01 * d01 + R02 * d02;
            director(1, k) = R10 * d00 + R11 * d01 + R12 * d02;
            director(2, k) = R20 * d00 + R21 * d01 + R22 * d02;
            director(3, k) = R00 * d10 + R01 * d11 + R02 * d12;
            director(4, k) = R10 * d10 + R11 * d11 + R12 * d12;
            director(5, k) = R20 * d10 + R21 * d11 + R22 * d12;
            director(6, k) = R00 * d20 + R01 * d21 + R02 * d22;
            director(7, k) = R10 * d20 + R11 * d21 + R12 * d22;
            director(8, k) = R20 * d20 + R21 * d21 + R22 * d22;
        }
    }

    // Update dynamics (velocity, omega) using acceleration and alpha
    // Equivalent to Python: update_dynamics(time, prefac)
    void update_dynamics(double prefac) {
        auto& block = static_cast<Derived&>(*this);

        // Get variable views from block using accessible type names
        auto velocity = block.template get<system::cosserat_rod::Velocity>();
        auto acceleration = block.template get<system::cosserat_rod::Acceleration>();
        auto omega = block.template get<system::cosserat_rod::Omega>();
        auto alpha = block.template get<system::cosserat_rod::Alpha>();

        // Update velocity: v += prefac * a
        // velocity is (3, n_nodes), acceleration is (3, n_nodes)
        // Eigen operations are automatically vectorized if SIMD is enabled
        velocity += prefac * acceleration;

        // Update omega: ω += prefac * α
        // omega is (3, n_elems), alpha is (3, n_elems)
        // Eigen operations are automatically vectorized if SIMD is enabled
        omega += prefac * alpha;
    }

protected:
    // Protected constructor to prevent direct instantiation
    CosseratRodOperations() = default;
    ~CosseratRodOperations() = default;

    // Prevent copying/moving (can be enabled in derived class if needed)
    CosseratRodOperations(const CosseratRodOperations&) = default;
    CosseratRodOperations(CosseratRodOperations&&) = default;
    CosseratRodOperations& operator=(const CosseratRodOperations&) = default;
    CosseratRodOperations& operator=(CosseratRodOperations&&) = default;
};

} // namespace elasticapp
