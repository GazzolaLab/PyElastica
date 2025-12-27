#pragma once

#include "traits.h"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <omp.h>

namespace elasticapp {

//**************************************************************************
/*!\brief Vector Difference
//
// \param[in] in_vector(3, n_nodes)
//
// \return output_vector(3, n_elems) where n_elems = n_nodes - 1
*/
template<typename V>
inline auto difference_kernel(const V& in_vector) {
    constexpr std::size_t dimension(3UL);
    assert(in_vector.rows() == dimension);
    const std::size_t n_nodes = in_vector.cols();
    const std::size_t n_elems = n_nodes - 1UL;

    MatrixType result(dimension, n_elems);
    result = in_vector.block(0, 1, dimension, n_elems) -
             in_vector.block(0, 0, dimension, n_elems);
    return result;
}

//**************************************************************************
/*!\brief Batchwise matrix-vector product.
//
// Computes: matvec_batch{ik} = matrix_batch{ijk} * vector_batch{jk}
//
// \param[out] matvec_batch(3, n_elems) matrix-vector product
// \param[in] matrix_batch(3, 3, n_elems)
// \param[in] vector_batch(3, n_elems)
*/
template <typename MT1, typename TT, typename MT2>
inline void batch_matvec(MT1& matvec_batch,
                  const TT& matrix_batch,
                  const MT2& vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = matrix_batch.cols();

    assert(matvec_batch.rows() == dimension);
    assert(matvec_batch.cols() == n_elems);
    assert(matrix_batch.pages() == dimension);
    assert(matrix_batch.rows() == dimension);
    assert(vector_batch.rows() == dimension);
    assert(vector_batch.cols() == n_elems);

    #pragma omp parallel for if(!omp_in_parallel())
    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t k = 0; k < n_elems; ++k) {
            double sum = 0.0;
            for (std::size_t j = 0; j < dimension; ++j) {
                sum += matrix_batch(i, j, k) * vector_batch(j, k);
            }
            matvec_batch(i, k) = sum;
        }
    }
}

//**************************************************************************
/*!\brief Batchwise matrix-matrix product.
//
// Computes: matmul_batch{ilk} = first_matrix_batch{ijk} * second_matrix_batch{jlk}
//
// \param[out] matmul_batch(3, 3, n_elems) matrix-matrix product
// \param[in] first_matrix_batch(3, 3, n_elems)
// \param[in] second_matrix_batch(3, 3, n_elems)
*/
template <typename TT1, typename TT2, typename TT3>
inline void batch_matmul(TT1& matmul_batch,
                  const TT2& first_matrix_batch,
                  const TT3& second_matrix_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_matrix_batch.cols();

    assert(matmul_batch.pages() == dimension);
    assert(matmul_batch.rows() == dimension);
    assert(matmul_batch.cols() == n_elems);
    assert(first_matrix_batch.pages() == dimension);
    assert(first_matrix_batch.rows() == dimension);
    assert(second_matrix_batch.pages() == dimension);
    assert(second_matrix_batch.rows() == dimension);
    assert(second_matrix_batch.cols() == n_elems);

    for (std::size_t i = 0; i < dimension; ++i) {
        for (std::size_t l = 0; l < dimension; ++l) {
            for (std::size_t k = 0; k < n_elems; ++k) {
                double sum = 0.0;
                for (std::size_t j = 0; j < dimension; ++j) {
                    sum += first_matrix_batch(i, j, k) * second_matrix_batch(j, l, k);
                }
                matmul_batch(i, l, k) = sum;
            }
        }
    }
}

//**************************************************************************
/*!\brief Batchwise vector-vector cross product.
//
// Computes cross product for each column: cross_batch{ik} = first_vector_batch{ik} x second_vector_batch{ik}
//
// \param[out] cross_batch(3, n_elems) vector-vector cross product
// \param[in] first_vector_batch(3, n_elems)
// \param[in] second_vector_batch(3, n_elems)
*/
template <typename MT1, typename MT2, typename MT3>
inline void batch_cross(MT1& cross_batch,
                 const MT2& first_vector_batch,
                 const MT3& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_vector_batch.cols();

    assert(cross_batch.rows() == dimension);
    assert(cross_batch.cols() == n_elems);
    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    assert(second_vector_batch.cols() == n_elems);

    #pragma omp parallel for if(!omp_in_parallel())
    for (std::size_t k = 0; k < n_elems; ++k) {
        cross_batch(0, k) = first_vector_batch(1, k) * second_vector_batch(2, k) -
                           first_vector_batch(2, k) * second_vector_batch(1, k);
        cross_batch(1, k) = first_vector_batch(2, k) * second_vector_batch(0, k) -
                           first_vector_batch(0, k) * second_vector_batch(2, k);
        cross_batch(2, k) = first_vector_batch(0, k) * second_vector_batch(1, k) -
                           first_vector_batch(1, k) * second_vector_batch(0, k);
    }
}

//**************************************************************************
/*!\brief Batchwise vector-vector dot product.
//
// Computes: dot_batch{j} = first_vector_batch{ij} * second_vector_batch{ij}
//
// \param[in] first_vector_batch(3, n_elems)
// \param[in] second_vector_batch(3, n_elems)
//
// \return dot_batch(n_elems)
*/
template <typename MT1, typename MT2>
inline auto batch_dot(const MT1& first_vector_batch,
               const MT2& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_vector_batch.cols();

    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    assert(second_vector_batch.cols() == n_elems);

    VectorType result(n_elems);
    #pragma omp parallel for if(!omp_in_parallel())
    for (std::size_t k = 0; k < n_elems; ++k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            sum += first_vector_batch(i, k) * second_vector_batch(i, k);
        }
        result(k) = sum;
    }
    return result;
}

//**************************************************************************
/*!\brief Batchwise vector L2 norm.
//
// Computes: norm_batch{j} = (vector_batch{ij} * vector_batch{ij})^0.5
//
// \param[in] vector_batch(3, n_elems)
//
// \return norm_batch(n_elems)
*/
template <typename MT>
inline auto batch_norm(const MT& vector_batch) {
    constexpr std::size_t dimension(3UL);
    assert(vector_batch.rows() == dimension);

    const std::size_t n_elems = vector_batch.cols();
    VectorType result(n_elems);

    #pragma omp parallel for if(!omp_in_parallel())
    for (std::size_t k = 0; k < n_elems; ++k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            sum += vector_batch(i, k) * vector_batch(i, k);
        }
        result(k) = std::sqrt(sum);
    }
    return result;
}

}  // namespace elasticapp
