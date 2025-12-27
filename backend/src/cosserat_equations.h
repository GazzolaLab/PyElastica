#pragma once

#include <cstddef>
#include <cmath>
#include "cosserat_rod_system.h"
#include "traits.h"
#include "math/eigen_detail/eigen_linear_algebra.hpp"
#include "math/eigen_detail/eigen_calculus.hpp"
#include <omp.h>

namespace elasticapp {

// Forward declaration
template<SystemModel SystemType, template<typename> class OperationsType>
class Block;

// Compute geometry from state (lengths, tangents, radius)
// Updates: lengths, tangents, radius
template<typename BlockType>
inline void compute_geometry_from_state(BlockType& block) {
    // Get variable views
    auto&& position = block.template get<system::cosserat_rod::Position>();
    auto&& volume = block.template get<system::cosserat_rod::Volume>();
    auto&& lengths = block.template get<system::cosserat_rod::Lengths>();
    auto&& tangents = block.template get<system::cosserat_rod::Tangents>();
    auto&& radius = block.template get<system::cosserat_rod::Radius>();

    // Compute position differences using difference_kernel
    // position_diff = position[:, 1:] - position[:, :-1]
    // This is equivalent to: difference_kernel(position)
    auto position_diff = difference_kernel(position);

    // Compute lengths = norm(position_diff) + 1e-14
    // FIXME: 1e-14 is added to fix ghost lengths, which is 0, and causes division by zero error!
    auto lengths_vec = batch_norm(position_diff);
    const IndexType n_elems = lengths.cols();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        lengths(0, k) = lengths_vec(k) + 1e-14;
    }

    // Compute tangents = position_diff / lengths
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        const double len = lengths(0, k);
        tangents(0, k) = position_diff(0, k) / len;
        tangents(1, k) = position_diff(1, k) / len;
        tangents(2, k) = position_diff(2, k) / len;
    }

    // Compute radius from volume conservation: radius = sqrt(volume / (lengths * pi))
    const double pi = 3.14159265358979323846;
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        radius(0, k) = std::sqrt(volume(0, k) / (lengths(0, k) * pi));
    }
}

// Compute all dilatations (element and voronoi)
// Updates: lengths, tangents, radius, dilatation, voronoi_dilatation
template<typename BlockType>
inline void compute_all_dilatations(BlockType& block) {
    // Get variable views
    auto&& lengths = block.template get<system::cosserat_rod::Lengths>();
    auto&& rest_lengths = block.template get<system::cosserat_rod::RestLengths>();
    auto&& dilatation = block.template get<system::cosserat_rod::Dilatation>();
    auto&& rest_voronoi_lengths = block.template get<system::cosserat_rod::RestVoronoiLengths>();
    auto&& voronoi_dilatation = block.template get<system::cosserat_rod::VoronoiDilatation>();

    // Compute dilatation = lengths / rest_lengths
    const IndexType n_elems = lengths.cols();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        dilatation(0, k) = lengths(0, k) / rest_lengths(0, k);
    }

    // Compute voronoi_lengths from average of lengths
    // voronoi_lengths = 0.5 * (lengths[k+1] + lengths[k]) for k in [0, n_voronoi-1]
    auto voronoi_lengths = average_kernel(lengths);

    // Compute voronoi_dilatation = voronoi_lengths / rest_voronoi_lengths
    const IndexType n_voronoi = voronoi_dilatation.cols();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_voronoi; ++k) {
        voronoi_dilatation(0, k) = voronoi_lengths(0, k) / rest_voronoi_lengths(0, k);
    }
}

// Compute shear/stretch strains (sigma)
// Updates: lengths, tangents, radius, dilatation, voronoi_dilatation, sigma
template<typename BlockType>
inline void compute_shear_stretch_strains(BlockType& block) {
    // Get variable views
    auto&& dilatation = block.template get<system::cosserat_rod::Dilatation>();
    auto&& director = block.template get<system::cosserat_rod::Director>();
    auto&& tangents = block.template get<system::cosserat_rod::Tangents>();
    auto&& sigma = block.template get<system::cosserat_rod::Sigma>();

    // Compute sigma = dilatation * batch_matvec(director_collection, tangents) - z_vector
    // director is stored as (9, n_elems) - flattened 3x3 matrices
    // Storage order: [d00, d10, d20, d01, d11, d21, d02, d12, d22] (column-major)
    // tangents is (3, n_elems)
    // sigma is (3, n_elems)
    const IndexType n_elems = sigma.cols();

    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        // Extract 3x3 director matrix for element k
        // director[:, k] is (row-wise flattened 3x3 matrix)
        //  [d00, d01, d02]
        //  [d10, d11, d12]
        //  [d20, d21, d22]
        double d00 = director(0, k);
        double d01 = director(1, k);
        double d02 = director(2, k);
        double d10 = director(3, k);
        double d11 = director(4, k);
        double d12 = director(5, k);
        double d20 = director(6, k);
        double d21 = director(7, k);
        double d22 = director(8, k);

        // Compute director @ tangents for element k
        // result = director_matrix * tangents[:, k]
        double result0 = d00 * tangents(0, k) + d01 * tangents(1, k) + d02 * tangents(2, k);
        double result1 = d10 * tangents(0, k) + d11 * tangents(1, k) + d12 * tangents(2, k);
        double result2 = d20 * tangents(0, k) + d21 * tangents(1, k) + d22 * tangents(2, k);

        // Compute sigma = dilatation * result - z_vector
        const double dil = dilatation(0, k);
        sigma(0, k) = dil * result0;
        sigma(1, k) = dil * result1;
        sigma(2, k) = dil * result2 - 1.0;
    }
}

// Compute internal shear/stretch stresses from model
// Updates: lengths, tangents, radius, dilatation, voronoi_dilatation, sigma, internal_stress
template<typename BlockType>
inline void compute_internal_shear_stretch_stresses_from_model(BlockType& block) {
    // Get variable views
    auto&& shear_matrix = block.template get<system::cosserat_rod::ShearMatrix>();
    auto&& sigma = block.template get<system::cosserat_rod::Sigma>();
    auto&& rest_sigma = block.template get<system::cosserat_rod::RestSigma>();
    auto&& internal_stress = block.template get<system::cosserat_rod::InternalStress>();

    // Compute sigma_diff = sigma - rest_sigma
    // sigma is (3, n_elems), rest_sigma is (3, n_elems)
    const IndexType n_elems = sigma.cols();

    // Compute internal_stress = batch_matvec(shear_matrix, sigma - rest_sigma)
    // shear_matrix is stored as (9, n_elems) - flattened 3x3 matrices
    // Storage order: [m00, m10, m20, m01, m11, m21, m02, m12, m22] (column-major)
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        // Compute sigma_diff = sigma - rest_sigma for element k
        double sigma_diff0 = sigma(0, k) - rest_sigma(0, k);
        double sigma_diff1 = sigma(1, k) - rest_sigma(1, k);
        double sigma_diff2 = sigma(2, k) - rest_sigma(2, k);

        // Extract 3x3 shear_matrix for element k
        // shear_matrix[:, k] is [m00, m10, m20, m01, m11, m21, m02, m12, m22]
        double m00 = shear_matrix(0, k);
        double m10 = shear_matrix(1, k);
        double m20 = shear_matrix(2, k);
        double m01 = shear_matrix(3, k);
        double m11 = shear_matrix(4, k);
        double m21 = shear_matrix(5, k);
        double m02 = shear_matrix(6, k);
        double m12 = shear_matrix(7, k);
        double m22 = shear_matrix(8, k);

        // Compute shear_matrix @ sigma_diff for element k
        // result = shear_matrix * sigma_diff
        internal_stress(0, k) = m00 * sigma_diff0 + m01 * sigma_diff1 + m02 * sigma_diff2;
        internal_stress(1, k) = m10 * sigma_diff0 + m11 * sigma_diff1 + m12 * sigma_diff2;
        internal_stress(2, k) = m20 * sigma_diff0 + m21 * sigma_diff1 + m22 * sigma_diff2;
    }
}

// Compute internal forces for Cosserat rod
// This is a template function that will be implemented later
template<typename BlockType>
inline void compute_internal_forces(BlockType& block) {
    // Get variable views
    auto&& director = block.template get<system::cosserat_rod::Director>();
    auto&& internal_stress = block.template get<system::cosserat_rod::InternalStress>();
    auto&& dilatation = block.template get<system::cosserat_rod::Dilatation>();
    auto&& internal_forces = block.template get<system::cosserat_rod::InternalForces>();
    auto&& cosserat_internal_stress = block.template get<system::cosserat_rod::ScratchVectorA>();

    // Compute cosserat_internal_stress = director^T @ internal_stress / dilatation
    // director is stored as (9, n_elems) - flattened 3x3 matrices
    // Storage order: [d00, d10, d20, d01, d11, d21, d02, d12, d22] (column-major)
    // internal_stress is (3, n_elems) where n_elems includes ghost elements
    // cosserat_internal_stress is (3, n_elems)
    // Note: internal_stress has adjusted width (width - 1 for OnElement), but we need full width
    // Get the full width from director which also has adjusted width
    const IndexType n_elems = director.cols();

    // Temporary matrix for cosserat_internal_stress
    // MatrixType cosserat_internal_stress(3, n_elems);

    // Compute cosserat_internal_stress = director^T @ internal_stress
    // Python: cosserat_internal_stress[i, k] = sum_j(director_collection[j, i, k] * internal_stress[j, k])
    // In C++: director_collection[j, i, k] = director(j*3 + i, k)
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_elems; ++k) {
            double sum = 0.0;
            for (IndexType j = 0; j < 3; ++j) {
                // director_collection[j, i, k] = director(j*3 + i, k)
                IndexType director_idx = j * 3 + i;
                sum += director(director_idx, k) * internal_stress(j, k);
            }
            cosserat_internal_stress(i, k) = sum / dilatation(0, k);
        }
    }


    // Reset ghost values for cosserat_internal_stress (OnElement)
    // Note: ghost_elems_idx() returns indices in full block coordinate system
    // But cosserat_internal_stress has adjusted width (width - 1 for OnElement)
    // So we need to map ghost indices to adjusted coordinate system
    // For OnElement: adjusted_index = full_index (since we just remove the last column, not ghost columns)
    // Actually, ghost elements are in the middle, so we need to check if they're within bounds
    auto ghost_elems = block.ghost_elems_idx();
    for (std::size_t ghost_col : ghost_elems) {
        IndexType data_col = static_cast<IndexType>(ghost_col);
        // Check bounds (ghost indices might be out of bounds for adjusted width)
        if (data_col >= 0 && data_col < n_elems) {
            for (IndexType i = 0; i < 3; ++i) {
                cosserat_internal_stress(i, data_col) = 0.0;
            }
        }
    }

    // Compute internal_forces = two_point_difference_kernel(cosserat_internal_stress)
    // internal_forces is OnNode (3, n_nodes) where n_nodes = n_elems + 1
    two_point_difference_kernel(internal_forces, cosserat_internal_stress);
}

// Compute bending/twist strains (kappa)
// Updates: kappa
template<typename BlockType>
inline void compute_bending_twist_strains(BlockType& block) {
    // Get variable views
    auto&& director = block.template get<system::cosserat_rod::Director>();
    auto&& rest_voronoi_lengths = block.template get<system::cosserat_rod::RestVoronoiLengths>();
    auto&& kappa = block.template get<system::cosserat_rod::Kappa>();

    // director is stored as (9, n_elems) - flattened 3x3 matrices
    // Storage order: [d00, d10, d20, d01, d11, d21, d02, d12, d22] (column-major)
    // rest_voronoi_lengths is (1, n_voronoi) where n_voronoi = n_elems - 1
    // kappa is (3, n_voronoi)
    const IndexType n_voronoi = kappa.cols();

    // Compute temp = inv_rotate(director_collection)
    // inv_rotate computes relative rotation between consecutive director frames
    // Python: temp = _inv_rotate(director_collection)
    // temp has shape (3, n_voronoi) where n_voronoi = n_elems - 1
    double temp_x, temp_y, temp_z;

    // Compute inv_rotate manually: relative rotation between consecutive directors
    // Python implementation computes cross products between consecutive director rows
    #pragma omp parallel for schedule(static)
    for (IndexType k = 0; k < n_voronoi; ++k) {
        // Extract director matrices for k and k+1
        // director[:, k] is (row-wise flattened 3x3 matrix)
        //  [d00, d01, d02]
        //  [d10, d11, d12]
        //  [d20, d21, d22]

        // For director[k]:
        double d00_k = director(0, k);
        double d01_k = director(1, k);
        double d02_k = director(2, k);
        double d10_k = director(3, k);
        double d11_k = director(4, k);
        double d12_k = director(5, k);
        double d20_k = director(6, k);
        double d21_k = director(7, k);
        double d22_k = director(8, k);

        // For director[k+1]:
        double d00_k1 = director(0, k + 1);
        double d01_k1 = director(1, k + 1);
        double d02_k1 = director(2, k + 1);
        double d10_k1 = director(3, k + 1);
        double d11_k1 = director(4, k + 1);
        double d12_k1 = director(5, k + 1);
        double d20_k1 = director(6, k + 1);
        double d21_k1 = director(7, k + 1);
        double d22_k1 = director(8, k + 1);

        // Compute inv_rotate: cross product between consecutive director frames
        temp_x = (d20_k1 * d10_k + d21_k1 * d11_k + d22_k1 * d12_k) -
                     (d10_k1 * d20_k + d11_k1 * d21_k + d12_k1 * d22_k);
        temp_y = (d00_k1 * d20_k + d01_k1 * d21_k + d02_k1 * d22_k) -
                     (d20_k1 * d00_k + d21_k1 * d01_k + d22_k1 * d02_k);
        temp_z = (d10_k1 * d00_k + d11_k1 * d01_k + d12_k1 * d02_k) -
                     (d00_k1 * d10_k + d01_k1 * d11_k + d02_k1 * d12_k);

        double trace = (d00_k1 * d00_k + d01_k1 * d01_k + d02_k1 * d02_k) +
                (d10_k1 * d10_k + d11_k1 * d11_k + d12_k1 * d12_k) +
                (d20_k1 * d20_k + d21_k1 * d21_k + d22_k1 * d22_k);

        // Clip the trace to between -1 and 3.
        // Any deviation beyond this is numerical error
        trace = std::clamp(trace, -1.0, 3.0);
        double cos = 0.5 * trace - 0.5;
        double theta = std::acos(cos) + 1e-14;
        double magnitude = -0.5 * theta / std::sin(theta) / rest_voronoi_lengths(0, k);
        kappa(0, k) = temp_x * magnitude;
        kappa(1, k) = temp_y * magnitude;
        kappa(2, k) = temp_z * magnitude;
    }
}

// Compute internal bending/twist stresses from model
// Updates: kappa, internal_couple
template<typename BlockType>
inline void compute_internal_bending_twist_stresses_from_model(BlockType& block) {
    // Get variable views
    auto&& kappa = block.template get<system::cosserat_rod::Kappa>();
    auto&& rest_kappa = block.template get<system::cosserat_rod::RestKappa>();
    auto&& bend_matrix = block.template get<system::cosserat_rod::BendMatrix>();
    auto&& internal_couple = block.template get<system::cosserat_rod::InternalCouple>();

    // kappa is (3, n_voronoi), rest_kappa is (3, n_voronoi)
    // bend_matrix is stored as (9, n_voronoi) - flattened 3x3 matrices
    // internal_couple is (3, n_voronoi)
    const IndexType n_voronoi = kappa.cols();

    // Compute diff_kappa = kappa - rest_kappa
    // MatrixType diff_kappa(3, n_voronoi);
    auto&& diff_kappa = block.template get<system::cosserat_rod::ScratchVectorA>();

    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_voronoi; ++k) {
            diff_kappa(i, k) = kappa(i, k) - rest_kappa(i, k);
        }
    }

    // Compute internal_couple = batch_matvec(bend_matrix, diff_kappa)
    // bend_matrix is stored as (9, n_voronoi) - flattened 3x3 matrices
    // Storage order: [m00, m10, m20, m01, m11, m21, m02, m12, m22] (column-major)
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_voronoi; ++k) {
        // Extract 3x3 bend_matrix for voronoi k
        // bend_matrix[:, k] is [m00, m10, m20, m01, m11, m21, m02, m12, m22]
        double m00 = bend_matrix(0, k);
        double m10 = bend_matrix(1, k);
        double m20 = bend_matrix(2, k);
        double m01 = bend_matrix(3, k);
        double m11 = bend_matrix(4, k);
        double m21 = bend_matrix(5, k);
        double m02 = bend_matrix(6, k);
        double m12 = bend_matrix(7, k);
        double m22 = bend_matrix(8, k);

        // Compute bend_matrix @ diff_kappa for voronoi k
        // result = bend_matrix * diff_kappa[:, k]
        internal_couple(0, k) = m00 * diff_kappa(0, k) + m01 * diff_kappa(1, k) + m02 * diff_kappa(2, k);
        internal_couple(1, k) = m10 * diff_kappa(0, k) + m11 * diff_kappa(1, k) + m12 * diff_kappa(2, k);
        internal_couple(2, k) = m20 * diff_kappa(0, k) + m21 * diff_kappa(1, k) + m22 * diff_kappa(2, k);
    }
}

// Compute dilatation rate
// Updates: dilatation_rate
template<typename BlockType>
inline void compute_dilatation_rate(BlockType& block) {
    // Get variable views
    auto&& position = block.template get<system::cosserat_rod::Position>();
    auto&& velocity = block.template get<system::cosserat_rod::Velocity>();
    auto&& lengths = block.template get<system::cosserat_rod::Lengths>();
    auto&& rest_lengths = block.template get<system::cosserat_rod::RestLengths>();
    auto&& dilatation_rate = block.template get<system::cosserat_rod::DilatationRate>();

    // position is (3, n_nodes), velocity is (3, n_nodes)
    // lengths is (1, n_elems), rest_lengths is (1, n_elems)
    // dilatation_rate is (1, n_elems)
    const IndexType n_nodes = position.cols();
    const IndexType n_elems = lengths.cols();

    // Compute r_dot_v = batch_dot(position, velocity)
    // This is the dot product of position and velocity at each node
    // MatrixType r_dot_v(1, n_nodes);
    auto&& r_dot_v = block.template get<system::cosserat_rod::ScratchScalarA>();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_nodes; ++k) {
        r_dot_v(0, k) = position(0, k) * velocity(0, k) +
                       position(1, k) * velocity(1, k) +
                       position(2, k) * velocity(2, k);
    }

    // Compute r_plus_one_dot_v = batch_dot(position[..., 1:], velocity[..., :-1])
    // Dot product of position[1:] and velocity[:-1] (both have n_elems elements)
    // MatrixType r_plus_one_dot_v(1, n_elems);
    auto&& r_plus_one_dot_v = block.template get<system::cosserat_rod::ScratchScalarB>();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        r_plus_one_dot_v(0, k) = position(0, k + 1) * velocity(0, k) +
                                position(1, k + 1) * velocity(1, k) +
                                position(2, k + 1) * velocity(2, k);
    }

    // Compute r_dot_v_plus_one = batch_dot(position[..., :-1], velocity[..., 1:])
    // Dot product of position[:-1] and velocity[1:] (both have n_elems elements)
    // MatrixType r_dot_v_plus_one(1, n_elems);
    auto&& r_dot_v_plus_one = block.template get<system::cosserat_rod::ScratchScalarC>();
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        r_dot_v_plus_one(0, k) = position(0, k) * velocity(0, k + 1) +
                                 position(1, k) * velocity(1, k + 1) +
                                 position(2, k) * velocity(2, k + 1);
    }

    // Compute dilatation_rate for each element
    // dilatation_rate[k] = (r_dot_v[k] + r_dot_v[k + 1] - r_dot_v_plus_one[k] - r_plus_one_dot_v[k])
    //                     / lengths[k] / rest_lengths[k]
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        dilatation_rate(0, k) = (r_dot_v(0, k) + r_dot_v(0, k + 1) -
                                 r_dot_v_plus_one(0, k) - r_plus_one_dot_v(0, k)) /
                                lengths(0, k) / rest_lengths(0, k);
    }
}

// Compute internal torques for Cosserat rod
// This is a template function that will be implemented later
template<typename BlockType>
inline void compute_internal_torques(BlockType& block) {
    // Get variable views
    auto&& voronoi_dilatation = block.template get<system::cosserat_rod::VoronoiDilatation>();
    auto&& internal_couple = block.template get<system::cosserat_rod::InternalCouple>();
    auto&& kappa = block.template get<system::cosserat_rod::Kappa>();
    auto&& rest_voronoi_lengths = block.template get<system::cosserat_rod::RestVoronoiLengths>();
    auto&& director = block.template get<system::cosserat_rod::Director>();
    auto&& tangents = block.template get<system::cosserat_rod::Tangents>();
    auto&& internal_stress = block.template get<system::cosserat_rod::InternalStress>();
    auto&& rest_lengths = block.template get<system::cosserat_rod::RestLengths>();
    auto&& mass_second_moment_of_inertia = block.template get<system::cosserat_rod::MassSecondMomentOfInertia>();
    auto&& omega = block.template get<system::cosserat_rod::Omega>();
    auto&& dilatation = block.template get<system::cosserat_rod::Dilatation>();
    auto&& dilatation_rate = block.template get<system::cosserat_rod::DilatationRate>();
    auto&& internal_torques = block.template get<system::cosserat_rod::InternalTorques>();

    // voronoi_dilatation is (1, n_voronoi), internal_couple is (3, n_voronoi)
    // kappa is (3, n_voronoi), rest_voronoi_lengths is (1, n_voronoi)
    // director is (9, n_elems), tangents is (3, n_elems)
    // internal_stress is (3, n_elems), rest_lengths is (1, n_elems)
    // mass_second_moment_of_inertia is (9, n_elems), omega is (3, n_elems)
    // dilatation is (1, n_elems), dilatation_rate is (1, n_elems)
    // internal_torques is (3, n_elems)
    const IndexType n_voronoi = voronoi_dilatation.cols();
    const IndexType n_elems = internal_torques.cols();
    const IndexType n_nodes = n_elems + 1;

    // Scratch buffers
    auto&& voronoi_dilatation_inv_cube_cached = block.template get<system::cosserat_rod::ScratchScalarC>();
    auto&& scratch_vec_a_voronoi = block.template get<system::cosserat_rod::ScratchVectorA>();
    auto&& bend_twist_couple_2D = block.template get<system::cosserat_rod::ScratchVectorB>();
    auto&& bend_twist_couple_3D = block.template get<system::cosserat_rod::ScratchVectorC>();
    auto&& shear_stretch_couple = block.template get<system::cosserat_rod::ScratchVectorD>();
    auto&& lagrangian_transport = block.template get<system::cosserat_rod::ScratchVectorE>();
    auto&& unsteady_dilatation = block.template get<system::cosserat_rod::ScratchVectorF>();
    auto&& director_tangents = block.template get<system::cosserat_rod::ScratchVectorA>();

    // Compute voronoi_dilatation_inv_cube_cached = 1.0 / voronoi_dilatation^3
    // MatrixType voronoi_dilatation_inv_cube_cached(1, n_voronoi);
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_voronoi; ++k) {
        double voronoi_dil = voronoi_dilatation(0, k);
        voronoi_dilatation_inv_cube_cached(0, k) = 1.0 / (voronoi_dil * voronoi_dil * voronoi_dil);
    }

    // Compute bend_twist_couple_2D = difference_kernel(internal_couple * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx)
    // First compute the product
    // MatrixType internal_couple_scaled(3, n_voronoi);
    auto internal_couple_scaled = scratch_vec_a_voronoi;
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_voronoi; ++k) {
            internal_couple_scaled(i, k) = internal_couple(i, k) * voronoi_dilatation_inv_cube_cached(0, k);
        }
    }

    // Reset ghost values for internal_couple_scaled (OnVoronoi)
    auto ghost_voronoi = block.ghost_voronoi_idx();
    for (std::size_t ghost_col : ghost_voronoi) {
        IndexType data_col = static_cast<IndexType>(ghost_col);
        if (data_col >= 0 && data_col < n_voronoi) {
            for (IndexType i = 0; i < 3; ++i) {
                internal_couple_scaled(i, data_col) = 0.0;
            }
        }
    }

    // Apply difference_kernel (two_point_difference_kernel)
    // MatrixType bend_twist_couple_2D(3, n_nodes);
    two_point_difference_kernel(bend_twist_couple_2D, internal_couple_scaled);

    // Compute bend_twist_couple_3D = quadrature_kernel((kappa x internal_couple) * rest_voronoi_lengths * voronoi_dilatation_inv_cube_cached, ghost_voronoi_idx)
    // First compute kappa x internal_couple
    // MatrixType kappa_cross_internal_couple(3, n_voronoi);
    auto kappa_cross_internal_couple = scratch_vec_a_voronoi;
    batch_cross(kappa_cross_internal_couple, kappa, internal_couple);

    // Multiply by rest_voronoi_lengths and voronoi_dilatation_inv_cube_cached
    // MatrixType bend_twist_couple_3D_input(3, n_voronoi);
    auto bend_twist_couple_3D_input = scratch_vec_a_voronoi;
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_voronoi; ++k) {
            bend_twist_couple_3D_input(i, k) = kappa_cross_internal_couple(i, k) *
                                              rest_voronoi_lengths(0, k) *
                                              voronoi_dilatation_inv_cube_cached(0, k);
        }
    }

    // Reset ghost values
    for (std::size_t ghost_col : ghost_voronoi) {
        IndexType data_col = static_cast<IndexType>(ghost_col);
        if (data_col >= 0 && data_col < n_voronoi) {
            for (IndexType i = 0; i < 3; ++i) {
                bend_twist_couple_3D_input(i, data_col) = 0.0;
            }
        }
    }

    // Apply quadrature_kernel (trapezoidal)
    // MatrixType bend_twist_couple_3D(3, n_nodes);
    quadrature_kernel(bend_twist_couple_3D, bend_twist_couple_3D_input);

    // Compute shear_stretch_couple = (Q^T * tangents) x internal_stress * rest_lengths
    // First compute Q^T * tangents (same as director^T @ tangents)
    // MatrixType director_tangents(3, n_elems);
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_elems; ++k) {
            double sum = 0.0;
            for (IndexType j = 0; j < 3; ++j) {
                IndexType director_idx = j * 3 + i;
                sum += director(director_idx, k) * tangents(j, k);
            }
            director_tangents(i, k) = sum;
        }
    }

    // Compute cross product: (Q^T * tangents) x internal_stress
    // MatrixType shear_stretch_couple(3, n_elems);
    batch_cross(shear_stretch_couple, director_tangents, internal_stress);

    // Multiply by rest_lengths
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_elems; ++k) {
            shear_stretch_couple(i, k) *= rest_lengths(0, k);
        }
    }

    // Compute J_omega_upon_e = batch_matvec(mass_second_moment_of_inertia, omega) / dilatation
    // MatrixType J_omega_upon_e(3, n_elems);
    auto J_omega_upon_e = director_tangents;
    #pragma omp parallel for simd schedule(static)
    for (IndexType k = 0; k < n_elems; ++k) {
        // Extract 3x3 mass_second_moment_of_inertia for element k
        double m00 = mass_second_moment_of_inertia(0, k);
        double m10 = mass_second_moment_of_inertia(1, k);
        double m20 = mass_second_moment_of_inertia(2, k);
        double m01 = mass_second_moment_of_inertia(3, k);
        double m11 = mass_second_moment_of_inertia(4, k);
        double m21 = mass_second_moment_of_inertia(5, k);
        double m02 = mass_second_moment_of_inertia(6, k);
        double m12 = mass_second_moment_of_inertia(7, k);
        double m22 = mass_second_moment_of_inertia(8, k);

        // Compute mass_second_moment_of_inertia @ omega
        J_omega_upon_e(0, k) = (m00 * omega(0, k) + m01 * omega(1, k) + m02 * omega(2, k)) / dilatation(0, k);
        J_omega_upon_e(1, k) = (m10 * omega(0, k) + m11 * omega(1, k) + m12 * omega(2, k)) / dilatation(0, k);
        J_omega_upon_e(2, k) = (m20 * omega(0, k) + m21 * omega(1, k) + m22 * omega(2, k)) / dilatation(0, k);
    }

    // Compute lagrangian_transport = (J * omega / dilatation) x omega
    // MatrixType lagrangian_transport(3, n_elems);
    batch_cross(lagrangian_transport, J_omega_upon_e, omega);

    // Compute unsteady_dilatation = J_omega_upon_e * dilatation_rate / dilatation
    // MatrixType unsteady_dilatation(3, n_elems);
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_elems; ++k) {
            unsteady_dilatation(i, k) = J_omega_upon_e(i, k) * dilatation_rate(0, k) / dilatation(0, k);
        }
    }

    // Compute internal_torques = sum of all components
    // Note: bend_twist_couple_2D and bend_twist_couple_3D are (3, n_nodes), but we need (3, n_elems)
    // So we need to take the element values (columns 0 to n_elems-1)
    for (IndexType i = 0; i < 3; ++i) {
        #pragma omp parallel for simd schedule(static)
        for (IndexType k = 0; k < n_elems; ++k) {
            internal_torques(i, k) = bend_twist_couple_2D(i, k) +
                                     bend_twist_couple_3D(i, k) +
                                     shear_stretch_couple(i, k) +
                                     lagrangian_transport(i, k) +
                                     unsteady_dilatation(i, k);
        }
    }
}

} // namespace elasticapp
