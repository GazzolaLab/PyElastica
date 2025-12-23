#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#ifdef ELASTICAPP_USE_THREADING
#include <omp.h>
#endif

namespace elastica {

using ElasticaMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// For 3D tensors, we represent them as matrices with manual indexing
struct ElasticaTensor {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_;
    std::size_t pages_;
    std::size_t rows_;
    std::size_t cols_;

    ElasticaTensor(std::size_t pages, std::size_t rows, std::size_t cols)
        : data_(pages * rows, cols), pages_(pages), rows_(rows), cols_(cols) {}

    double& operator()(std::size_t page, std::size_t row, std::size_t col) {
        return data_(page * rows_ + row, col);
    }

    const double& operator()(std::size_t page, std::size_t row, std::size_t col) const {
        return data_(page * rows_ + row, col);
    }

    std::size_t pages() const { return pages_; }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
};

//**************************************************************************
/*!\brief Batchwise matrix logarithmic operator (inverse rotation).
//
// Batchwise for rotation matrix R computes the corresponding rotation
// axis vector {theta (u)} using the matrix log() operator.
//
// \param[out] rot_axis_vector_batch(3, n_elems) rotation axis vector batch
// \param[in] rot_matrix_batch(3, 3, n_elems) rotation matrix batch
*/
template <typename MT, typename TT>
void batch_inv_rotate(MT& rot_axis_vector_batch, const TT& rot_matrix_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = rot_matrix_batch.cols();
    using ValueType = double;

    assert(rot_matrix_batch.pages() == dimension);
    assert(rot_matrix_batch.rows() == dimension);
    assert(rot_axis_vector_batch.rows() == dimension);
    assert(rot_axis_vector_batch.cols() == n_elems);

    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) if(!omp_in_parallel())
    #else
    #pragma omp parallel for if(!omp_in_parallel())
    #endif
    #endif
    for (std::size_t k = 0; k < n_elems; ++k) {
        // Compute trace: tr(R) = R[0,0] + R[1,1] + R[2,2]
        double trace = rot_matrix_batch(0, 0, k) +
                      rot_matrix_batch(1, 1, k) +
                      rot_matrix_batch(2, 2, k);

        // Clip trace to [-1, 3] for numerical stability
        trace = std::max(-1.0, std::min(3.0, trace));

        // theta = acos((tr(R) - 1) / 2)
        double theta = std::acos(0.5 * (trace - 1.0) - 1e-12);

        // Compute R - R^T (skew-symmetric part)
        // Extract vector from skew-symmetric matrix
        rot_axis_vector_batch(0, k) = rot_matrix_batch(2, 1, k) - rot_matrix_batch(1, 2, k);
        rot_axis_vector_batch(1, k) = rot_matrix_batch(0, 2, k) - rot_matrix_batch(2, 0, k);
        rot_axis_vector_batch(2, k) = rot_matrix_batch(1, 0, k) - rot_matrix_batch(0, 1, k);

        // theta (u) = -theta * inv_skew([R - RT]) / (2 * sin(theta))
        double sin_theta = std::sin(theta) + 1e-14;
        double magnitude = -0.5 * theta / sin_theta;

        rot_axis_vector_batch(0, k) *= magnitude;
        rot_axis_vector_batch(1, k) *= magnitude;
        rot_axis_vector_batch(2, k) *= magnitude;
    }
}

//**************************************************************************
/*!\brief Batchwise matrix exponential operator (Rodrigues formula).
//
// Batchwise for rotation axis vector {theta u} computes the corresponding
// rotation matrix R using the matrix exp() operator (Rodrigues formula):
// R = I + sin(theta) * U + (1 - cos(theta)) * U^2
//
// \param[out] rot_matrix_batch(3, 3, n_elems) rotation matrix batch
// \param[in] rot_axis_vector_batch(3, n_elems) rotation axis vector batch
*/
template <typename TT, typename MT>
void exp_batch(TT& rot_matrix_batch, const MT& rot_axis_vector_batch) {
    constexpr std::size_t dimension(3UL);
    using ValueType = double;

    assert(rot_axis_vector_batch.rows() == dimension);
    assert(rot_matrix_batch.pages() == dimension);
    assert(rot_matrix_batch.rows() == dimension);

    const std::size_t n_elems = rot_axis_vector_batch.cols();

    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) if(!omp_in_parallel())
    #else
    #pragma omp parallel for if(!omp_in_parallel())
    #endif
    #endif
    for (std::size_t k = 0; k < n_elems; ++k) {
        // Compute theta = ||axis||
        double v0 = rot_axis_vector_batch(0, k);
        double v1 = rot_axis_vector_batch(1, k);
        double v2 = rot_axis_vector_batch(2, k);
        double theta = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);

        // Normalize axis
        double norm = theta + 1e-14;
        v0 /= norm;
        v1 /= norm;
        v2 /= norm;

        // Precompute sin and cos
        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        double one_minus_cos = 1.0 - cos_theta;

        // Build rotation matrix using Rodrigues formula
        // Diagonal elements: R[ii] = cos(theta) + (1 - cos(theta)) * u[i]^2
        rot_matrix_batch(0, 0, k) = cos_theta + one_minus_cos * v0 * v0;
        rot_matrix_batch(1, 1, k) = cos_theta + one_minus_cos * v1 * v1;
        rot_matrix_batch(2, 2, k) = cos_theta + one_minus_cos * v2 * v2;

        // Off-diagonal elements
        rot_matrix_batch(0, 1, k) = one_minus_cos * v0 * v1 + sin_theta * v2;
        rot_matrix_batch(1, 0, k) = one_minus_cos * v0 * v1 - sin_theta * v2;

        rot_matrix_batch(0, 2, k) = one_minus_cos * v0 * v2 - sin_theta * v1;
        rot_matrix_batch(2, 0, k) = one_minus_cos * v0 * v2 + sin_theta * v1;

        rot_matrix_batch(1, 2, k) = one_minus_cos * v1 * v2 + sin_theta * v0;
        rot_matrix_batch(2, 1, k) = one_minus_cos * v1 * v2 - sin_theta * v0;
    }
}

} // namespace elastica
