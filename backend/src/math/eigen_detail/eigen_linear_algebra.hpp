#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

#ifdef ELASTICAPP_USE_THREADING
#include <omp.h>
#endif

namespace elastica {

// Type aliases for Eigen matrices
using ElasticaMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ElasticaVector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

// For 3D tensors, we represent them as matrices with manual indexing
// Tensor(pages, rows, cols) is stored as Matrix(pages * rows, cols)
// Access: tensor(page, row, col) = matrix(page * rows + row, col)
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

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix() { return data_; }
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix() const { return data_; }
};

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

    ElasticaMatrix result(dimension, n_elems);
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
void batch_matvec(MT1& matvec_batch,
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

    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) collapse(2) if(!omp_in_parallel())
    #else
    #pragma omp parallel for collapse(2) if(!omp_in_parallel())
    #endif
    #endif
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
void batch_matmul(TT1& matmul_batch,
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
void batch_cross(MT1& cross_batch,
                 const MT2& first_vector_batch,
                 const MT3& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_vector_batch.cols();

    assert(cross_batch.rows() == dimension);
    assert(cross_batch.cols() == n_elems);
    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    assert(second_vector_batch.cols() == n_elems);

    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) if(!omp_in_parallel())
    #else
    #pragma omp parallel for if(!omp_in_parallel())
    #endif
    #endif
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
auto batch_dot(const MT1& first_vector_batch,
               const MT2& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_vector_batch.cols();

    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    assert(second_vector_batch.cols() == n_elems);

    ElasticaVector result(n_elems);
    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) if(!omp_in_parallel())
    #else
    #pragma omp parallel for if(!omp_in_parallel())
    #endif
    #endif
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
auto batch_norm(const MT& vector_batch) {
    constexpr std::size_t dimension(3UL);
    assert(vector_batch.rows() == dimension);

    const std::size_t n_elems = vector_batch.cols();
    ElasticaVector result(n_elems);

    #ifdef ELASTICAPP_USE_THREADING
    #ifdef ELASTICAPP_NUM_THREADS
    #pragma omp parallel for num_threads(ELASTICAPP_NUM_THREADS) if(!omp_in_parallel())
    #else
    #pragma omp parallel for if(!omp_in_parallel())
    #endif
    #endif
    for (std::size_t k = 0; k < n_elems; ++k) {
        double sum = 0.0;
        for (std::size_t i = 0; i < dimension; ++i) {
            sum += vector_batch(i, k) * vector_batch(i, k);
        }
        result(k) = std::sqrt(sum);
    }
    return result;
}

}  // namespace elastica
