#pragma once

#include <Eigen/Dense>
#include "traits.h"
#include <cassert>
#include <omp.h>

namespace elasticapp {

//**************************************************************************
/*!\brief Simple 2-point difference rule with zero at end points.
//
// Discrete 2-point difference in elasticapp of a function f:[a,b]-> R, i.e
// D f[a,b] -> df[a,b] where f satisfies the conditions
// f(a) = f(b) = 0.0. Operates from rod's elemental space to nodal space.
//
// \param[out] out_matrix(3, n_nodes) difference values
// \param[in] in_matrix(3, n_elems) vector batch
// where n_nodes = n_elems + 1
*/
template <typename MT1, typename MT2>
inline void two_point_difference_kernel(MT1& out_matrix, const MT2& in_matrix) {
    constexpr std::size_t dimension(3UL);
    assert(in_matrix.rows() == dimension);
    assert(out_matrix.rows() == dimension);
    const std::size_t n_elems = in_matrix.cols();
    const std::size_t n_nodes = n_elems + 1UL;
    assert(out_matrix.cols() == n_nodes);

    // First column: f(a) = in_matrix[:, 0]
    out_matrix.col(0) = in_matrix.col(0);

    // Last column: f(b) = -in_matrix[:, n_elems-1]
    out_matrix.col(n_nodes - 1) = -in_matrix.col(n_elems - 1);

    // Middle columns: difference between consecutive elements
    if (n_elems > 1) {
        out_matrix.block(0, 1, dimension, n_elems - 1) =
            in_matrix.block(0, 1, dimension, n_elems - 1) -
            in_matrix.block(0, 0, dimension, n_elems - 1);
    }
}

//**************************************************************************
/*!\brief Simple trapezoidal quadrature rule with zero at end points.
//
// Discrete integral of a function in elasticapp
// f : [a,b] -> R, ∫[a,b] f -> R
// where f satisfies the conditions f(a) = f(b) = 0.0.
// Operates from rod's elemental space to nodal space.
//
// \param[out] out_matrix(3, n_nodes) quadrature values
// \param[in] in_matrix(3, n_elems) vector batch
// where n_nodes = n_elems + 1
*/
template <typename MT1, typename MT2>
inline void quadrature_kernel(MT1& out_matrix, const MT2& in_matrix) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = in_matrix.cols();
    assert(in_matrix.rows() == dimension);
    assert(out_matrix.rows() == dimension);
    const std::size_t n_nodes = n_elems + 1UL;
    assert(out_matrix.cols() == n_nodes);

    using ValueType = typename MT1::Scalar;

    // First column: 0.5 * in_matrix[:, 0]
    out_matrix.col(0) = ValueType(0.5) * in_matrix.col(0);

    // Last column: 0.5 * in_matrix[:, n_elems-1]
    out_matrix.col(n_nodes - 1) = ValueType(0.5) * in_matrix.col(n_elems - 1);

    // Middle columns: 0.5 * (in_matrix[:, k] + in_matrix[:, k-1])
    out_matrix.block(0, 1, dimension, n_elems - 1) =
        ValueType(0.5) *
        (in_matrix.block(0, 1, dimension, n_elems - 1) +
            in_matrix.block(0, 0, dimension, n_elems - 1));
}

//**************************************************************************
/*!\brief Average between consecutive elements.
//
// Computes average of consecutive elements: average[k] = 0.5 * (vector[k+1] + vector[k])
//
// \param[in] in_vector(1, n_elems) or (n_elems,) scalar array
//
// \return output_vector(1, n_voronoi) where n_voronoi = n_elems - 1
*/
template <typename MT>
auto average_kernel(const MT& in_vector) {
    const std::size_t n_elems = in_vector.cols();
    const std::size_t n_voronoi = n_elems > 0 ? n_elems - 1UL : 0UL;

    MatrixType result(1, n_voronoi);

    #pragma omp parallel for if(!omp_in_parallel())
    for (std::size_t k = 0; k < n_voronoi; ++k) {
        result(0, k) = 0.5 * (in_vector(0, k + 1) + in_vector(0, k));
    }

    return result;
}

} // namespace elasticapp
