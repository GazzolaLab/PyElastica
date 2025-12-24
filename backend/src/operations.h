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
    inline void compute_internal_forces_and_torques(double time) {
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

    // Compute strains
    inline void compute_strains(double time) {
        (void)time;  // Suppress unused parameter warning
        auto& block = static_cast<Derived&>(*this);
        // Compute internal forces and torques separately
        compute_geometry_from_state(block);
        compute_all_dilatations(block);
        compute_shear_stretch_strains(block);
        compute_bending_twist_strains(block);
    }

    // Update accelerations based on forces
    inline void update_accelerations(double time) {
        (void)time;  // Suppress unused parameter warning
        auto& block = static_cast<Derived&>(*this);

        // Get all required variables
        auto&& acceleration = block.template get<system::cosserat_rod::Acceleration>();
        auto&& internal_forces = block.template get<system::cosserat_rod::InternalForces>();
        auto&& external_forces = block.template get<system::cosserat_rod::ExternalForces>();
        auto&& mass = block.template get<system::cosserat_rod::Mass>();
        auto&& alpha = block.template get<system::cosserat_rod::Alpha>();
        auto&& inv_mass_second_moment_of_inertia = block.template get<system::cosserat_rod::InvMassSecondMomentOfInertia>();
        auto&& internal_torques = block.template get<system::cosserat_rod::InternalTorques>();
        auto&& external_torques = block.template get<system::cosserat_rod::ExternalTorques>();
        auto&& dilatation = block.template get<system::cosserat_rod::Dilatation>();

        // Update translational acceleration: a = (F_internal + F_external) / m
        // acceleration is OnNode, Vector (3 rows, n_nodes cols)
        const IndexType n_nodes = acceleration.cols();
        for (IndexType i = 0; i < 3; ++i) {
            #pragma omp parallel for simd schedule(static)
            for (IndexType k = 0; k < n_nodes; ++k) {
                acceleration(i, k) = (internal_forces(i, k) + external_forces(i, k)) / mass(0, k);
            }
        }

        // Update angular acceleration: alpha = inv_J * (tau_internal + tau_external) * dilatation
        // First zero out alpha
        alpha.setZero();

        // inv_mass_second_moment_of_inertia is OnElement, Matrix (9 rows, n_elems cols)
        // Stored as 9xN where each column is a flattened 3x3 matrix: [I00, I01, I02, I10, I11, I12, I20, I21, I22]
        // alpha is OnElement, Vector (3 rows, n_elems cols)
        const IndexType n_elems = alpha.cols();
        for (IndexType i = 0; i < 3; ++i) {
            for (IndexType j = 0; j < 3; ++j) {
                #pragma omp parallel for simd schedule(static)
                for (IndexType k = 0; k < n_elems; ++k) {
                    // Map 3x3 matrix index (i, j) to flattened 9x1 index: i*3 + j
                    IndexType inv_J_idx = i * 3 + j;
                    alpha(i, k) += inv_mass_second_moment_of_inertia(inv_J_idx, k)
                                 * (internal_torques(j, k) + external_torques(j, k))
                                 * dilatation(0, k);
                }
            }
        }
    }

    // Zero out external forces and torques
    // time parameter is included to match Python signature, but not used in implementation
    inline void zeroed_out_external_forces_and_torques(double time) {
        (void)time;  // Suppress unused parameter warning

        auto& block = static_cast<Derived&>(*this);
        auto&& external_forces = block.template get<system::cosserat_rod::ExternalForces>();
        auto&& external_torques = block.template get<system::cosserat_rod::ExternalTorques>();

        external_forces.setZero();
        external_torques.setZero();
        return;

        // Zero out all external forces (OnNode, Vector: 3 rows, n_nodes columns)
        // Use explicit loop to ensure values are set (matching Python implementation)
        const IndexType n_nodes = external_forces.cols();
        for (IndexType i = 0; i < 3; ++i) {
            #pragma omp parallel for simd schedule(static)
            for (IndexType k = 0; k < n_nodes; ++k) {
                external_forces(i, k) = 0.0;
            }
        }

        // Zero out all external torques (OnElement, Vector: 3 rows, n_elems columns)
        const IndexType n_elems = external_torques.cols();
        for (IndexType i = 0; i < 3; ++i) {
            #pragma omp parallel for simd schedule(static)
            for (IndexType k = 0; k < n_elems; ++k) {
                external_torques(i, k) = 0.0;
            }
        }
    }

    // Update kinematics (position, director) using velocity and omega
    // Equivalent to Python: update_kinematics(time, prefac)
    inline void update_kinematics(double prefac) {
        auto& block = static_cast<Derived&>(*this);

        // Get variable views from block using accessible type names
        auto&& position = block.template get<system::cosserat_rod::Position>();
        auto&& velocity = block.template get<system::cosserat_rod::Velocity>();
        auto&& director = block.template get<system::cosserat_rod::Director>();
        auto&& omega = block.template get<system::cosserat_rod::Omega>();

        // Update position: x += prefac * v
        // position is (3, n_nodes), velocity is (3, n_nodes)
        const IndexType n_nodes = position.cols();
        for (IndexType i = 0; i < 3; ++i) {
            #pragma omp parallel for simd schedule(static) // num_threads(2)
            for (IndexType k = 0; k < n_nodes; ++k) {
                position(i, k) += prefac * velocity(i, k);
            }
        }
        // position += prefac * velocity;

        // Update director using rotation matrix from omega
        // director is stored as (9, n_elems) - flattened 3x3 matrices
        // omega is (3, n_elems)
        const std::size_t n_elems = director.cols();
        constexpr std::size_t dim = 3;

        // Parallelize loop if threading is enabled
        #pragma omp parallel for simd schedule(static)
        for (std::size_t k = 0; k < n_elems; ++k) {
            // Match Python implementation: _get_rotation_matrix in _rotations.py
            // Step 1: Get unscaled omega components
            double v0 = omega(0, k);
            double v1 = omega(1, k);
            double v2 = omega(2, k);

            // Step 2: Compute theta = ||omega|| (magnitude before scaling)
            double theta = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

            // Step 3: Normalize axis (add epsilon to prevent division by zero, matching Python)
            double norm = theta + 1e-14;
            v0 /= norm;
            v1 /= norm;
            v2 /= norm;

            // Step 4: Scale theta by prefac (matching Python: theta *= scale)
            theta *= prefac;

            // Step 5: Precompute sin and cos (matching Python: u_prefix = sin(theta), u_sq_prefix = 1.0 - cos(theta))
            double u_prefix = std::sin(theta);
            double u_sq_prefix = 1.0 - std::cos(theta);

            // Step 6: Build rotation matrix using exact Python formulas
            // Python: rot_mat[0, 0, k] = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2)
            // This is equivalent to: cos(theta) + (1 - cos(theta)) * v0^2
            // but we use Python's formula for exact numerical match
            double R00 = 1.0 - u_sq_prefix * (v1 * v1 + v2 * v2);
            double R11 = 1.0 - u_sq_prefix * (v0 * v0 + v2 * v2);
            double R22 = 1.0 - u_sq_prefix * (v0 * v0 + v1 * v1);

            // Off-diagonal elements (matching Python exactly)
            double R01 = u_prefix * v2 + u_sq_prefix * v0 * v1;
            double R10 = -u_prefix * v2 + u_sq_prefix * v0 * v1;
            double R02 = -u_prefix * v1 + u_sq_prefix * v0 * v2;
            double R20 = u_prefix * v1 + u_sq_prefix * v0 * v2;
            double R12 = u_prefix * v0 + u_sq_prefix * v1 * v2;
            double R21 = -u_prefix * v0 + u_sq_prefix * v1 * v2;

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
    // Optimized: Use direct loop instead of Eigen operations for better performance
    // This matches the Python Numba implementation which uses simple nested loops
    inline void update_dynamics(double prefac) {
        auto& block = static_cast<Derived&>(*this);

        // Get variable views once (these are lightweight Eigen block expressions)
        // The get<>() calls are template functions that should be fully inlined
        // However, to minimize overhead, we compute dimensions once and reuse
        auto&& velocity = block.template get<system::cosserat_rod::Velocity>();
        auto&& acceleration = block.template get<system::cosserat_rod::Acceleration>();
        auto&& omega = block.template get<system::cosserat_rod::Omega>();
        auto&& alpha = block.template get<system::cosserat_rod::Alpha>();

        // Cache dimensions (avoid repeated .cols() calls)
        const IndexType n_nodes = velocity.cols();
        const IndexType n_elems = omega.cols();

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    velocity(0, k) += prefac * acceleration(0, k);
                }
            }
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    velocity(1, k) += prefac * acceleration(1, k);
                }
            }
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    velocity(2, k) += prefac * acceleration(2, k);
                }
            }
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    omega(0, k) += prefac * alpha(0, k);
                }
            }
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    omega(1, k) += prefac * alpha(1, k);
                }
            }
            #pragma omp section
            {
                #ifdef ELASTICAPP_COMPONENT_THREADS
                #pragma omp parallel for simd schedule(static) num_threads(ELASTICAPP_COMPONENT_THREADS)
                #endif
                for (IndexType k = 0; k < n_nodes; ++k) {
                    omega(2, k) += prefac * alpha(2, k);
                }
            }
        }
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
