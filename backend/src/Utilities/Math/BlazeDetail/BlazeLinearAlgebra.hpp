#pragma once

#include "blaze/Blaze.h"
#include "blaze_tensor/Blaze.h"

/* some of them have for loops until blaze_tensor comes up with 3D tensor */
/* products imp */
/* things would have been much simpler if there was einsum() in blaze */
/* NOTE: % operator corresponds to Schur product for matrices and tensors */
namespace elastica {
  //**************************************************************************
  /*!\brief Vector Diference
  //
  // \param[out] output_vector(3, n_elems)
  // \param[in] input_vector(3, n_elems)
  //
  // \return void/None
  //
  // \see TODO: fill later
  */
  template<typename V>
  inline auto difference_kernel(const V& in_vector){
    constexpr std::size_t dimension(3UL);
    assert(in_vector.rows() == dimension);
    const std::size_t n_nodes = in_vector.columns();
    const std::size_t n_elems = n_nodes - 1UL;

    return blaze::submatrix(in_vector, 0UL, 1UL, dimension, n_elems) -
           blaze::submatrix(in_vector, 0UL, 0UL, dimension, n_elems);
  }

  //**************************************************************************
  /*!\brief Batchwise matrix-vector product.
  //
  // \details
  // Computes a batchwise matrix-vector product given in indical notation: \n
  // matvec_batch{ik} = matrix_batch{ijk} * vector_batch{jk}
  //
  // \example
  // The following shows a typical use of the batch_matvec function
  // with the expected (correct) result also shown.
  // \snippet test_linalg.cpp batch_matvec_example
  //
  // \param[out] matvec_batch(3, n_elems) matrix-vector product
  // \param[in] matrix_batch(3, 3, n_elems)
  // \param[in] vector_batch(3, n_elems)
  //
  // \return void/None
  //
  // \see fill later?
  */
  template <typename MT1,  // blaze Matrix expression type 1
            typename TT,   // blaze Tensor expression type
            typename MT2>  // blaze Matrix expression type 2
  void batch_matvec(MT1& matvec_batch,
                    const TT& matrix_batch,
                    const MT2& vector_batch) {
    constexpr std::size_t dimension(3UL);
    //const std::size_t n_elems = matrix_batch.columns();

    // Written for debugging purpose
//    const std::size_t vbatch_columns = vector_batch.columns();
//    const std::size_t vbatch_rows = vector_batch.rows();
//    const std::size_t mbatch_pages = matrix_batch.pages();
//    const std::size_t mbatch_columns = matrix_batch.columns();
//    const std::size_t mbatch_rows = matrix_batch.rows();
//
//    const std::size_t size1 = blaze::pageslice(matrix_batch, 0UL).columns();
//    const std::size_t size0 = blaze::pageslice(matrix_batch, 0UL).rows();
    //    assert(matvec_batch.columns() == n_elems);
    //    assert(matvec_batch.rows() == dimension);
    //    assert(matrix_batch.pages() == dimension);
    //    assert(matrix_batch.rows() == dimension);
    //    assert(vector_batch.columns() == n_elems);
    //    assert(vector_batch.rows() == dimension);

    for (std::size_t i(0UL); i < dimension; ++i) {
      auto val1 = blaze::pageslice(matrix_batch, i);
      auto val2 = blaze::sum<blaze::columnwise>(val1 % vector_batch);
      blaze::row(matvec_batch, i) = val2;
    }
  }
  //**************************************************************************

  //**************************************************************************
  /*!\brief Batchwise matrix-matrix product.
  //
  // \details
  // Computes a batchwise matrix-matrix product given in
  // indical notation: \n
  // matmul_batch{ilk} = first_matrix_batch{ijk} * second_matrix_batch{jlk}
  //
  // \example
  // The following shows a typical use of the batch_matmul function
  // with the expected (correct) result also shown.
  // \snippet test_linalg.cpp batch_matmul_example
  //
  // \param[out] matmul_batch(3, 3, n_elems) matrix-matrix product
  // \param[in] first_matrix_batch(3, 3, n_elems)
  // \param[in] second_matrix_batch(3, 3, n_elems)
  //
  // \return void/None
  //
  // \see fill later?
  */
  template <typename TT1,  // blaze Tensor expression type 1
      typename TT2,  // blaze Tensor expression type 2
      typename TT3>  // blaze Tensor expression type 3
  void batch_matmul(TT1& matmul_batch,
                    const TT2& first_matrix_batch,
                    const TT3& second_matrix_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_matrix_batch.columns();
    assert(matmul_batch.pages() == dimension);
    assert(matmul_batch.rows() == dimension);
    assert(matmul_batch.columns() == n_elems);
    assert(first_matrix_batch.pages() == dimension);
    assert(first_matrix_batch.rows() == dimension);
    assert(second_matrix_batch.pages() == dimension);
    assert(second_matrix_batch.rows() == dimension);
    assert(second_matrix_batch.columns() == n_elems);

    for (std::size_t i(0UL); i < dimension; ++i)
      for (std::size_t j(0UL); j < dimension; ++j)
        /* loop over dimensions, lesser iterations, bigger slices */
        blaze::row(blaze::pageslice(matmul_batch, i), j) =
            blaze::sum<blaze::columnwise>(
                blaze::pageslice(first_matrix_batch, i) %
                blaze::trans(blaze::rowslice(second_matrix_batch, j)));
  }
  //**************************************************************************

  //**************************************************************************
  /*!\brief Batchwise vector-vector cross product.
  //
  // \details
  // Computes a batchwise vector-vector cross product given in
  // indical notation: \n
  // cross_batch{il} = LCT{ijk} * first_vector_batch{jl} *
  // second_vector_batch{kl} \n
  // where LCT is the Levi-Civita Tensor
  //
  // \example
  // The following shows a typical use of the batch_cross function
  // with the expected (correct) result also shown.
  // \snippet test_linalg.cpp batch_cross_example
  //
  // \param[out] cross_batch(3, n_elems) vector-vector cross product
  // \param[in] first_vector_batch(3, n_elems)
  // \param[in] second_vector_batch(3, n_elems)
  //
  // \return void/None
  //
  // \see fill later?
  */
  template <typename MT1,  // blaze Matrix expression type 1
      typename MT2,  // blaze Matrix expression type 2
      typename MT3>  // blaze Matrix expression type 3
  void batch_cross(MT1& cross_batch,
                   const MT2& first_vector_batch,
                   const MT3& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    //const std::size_t n_elems = first_vector_batch.columns();
    assert(cross_batch.rows() == dimension);
    //assert(cross_batch.columns() == n_elems);
    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    //assert(second_vector_batch.columns() == n_elems);

    for (std::size_t i(0UL); i < dimension; ++i)
      /* loop over dimensions, lesser iterations, bigger slices */
      /* remainder operator % cycles the indices across dimension*/
      blaze::row(cross_batch, i) =
          blaze::row(first_vector_batch, (i + 1UL) % dimension) *
          blaze::row(second_vector_batch, (i + 2UL) % dimension) -
          blaze::row(first_vector_batch, (i + 2UL) % dimension) *
          blaze::row(second_vector_batch, (i + 1UL) % dimension);
  }
  //**************************************************************************

  //**************************************************************************
  /*!\brief Batchwise vector-vector dot product.
  //
  // \details
  // Computes a batchwise vector-vector dot product given in indical
  // notation: \n
  // dot_batch{j} = first_vector_batch{ij} * second_vector_batch{ij}
  //
  // \example
  // The following shows a typical use of the batch_dot function
  // with the expected (correct) result also shown.
  // \snippet test_linalg.cpp batch_dot_example
  //
  // \param[in] first_vector_batch(3, n_elems)
  // \param[in] second_vector_batch(3, n_elems)
  //
  // \return dot_batch(n_elems): ET object pointer to the dot
  // product (for lazy evaluation).
  //
  // \see fill later?
  */
  template <typename MT1,  // blaze Matrix expression type 1
            typename MT2>  // blaze Matrix expression type 2
  auto batch_dot(const MT1& first_vector_batch,
                 const MT2& second_vector_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = first_vector_batch.columns();
    assert(first_vector_batch.rows() == dimension);
    assert(second_vector_batch.rows() == dimension);
    assert(second_vector_batch.columns() == n_elems);

    return blaze::trans(
        blaze::sum<blaze::columnwise>(first_vector_batch % second_vector_batch));
  }
  //**************************************************************************

  //**************************************************************************
  /*!\brief Batchwise vector L2 norm.
  //
  // \details
  // Computes a batchwise vector L2 norm given in indical notation: \n
  // norm_batch{j} = (vector_batch{ij} * vector_batch{ij})^0.5
  //
  // \example
  // The following shows a typical use of the batch_norm function
  // with the expected (correct) result also shown.
  // \snippet test_linalg.cpp batch_norm_example
  //
  // \param[in] vector_batch(3, n_elems)
  //
  // \return norm_batch(n_elems): ET object pointer to the vector L2 norm
  // (for lazy evaluation).
  //
  // \see fill later?
  */
  template <typename MT>  // blaze vector expression type
  auto batch_norm(const MT& vector_batch) {
    constexpr std::size_t dimension(3UL);
    //const std::size_t n_elems = vector_batch.columns();
    assert(vector_batch.rows() == dimension);

    return blaze::sqrt(blaze::trans(blaze::sum<blaze::columnwise>(vector_batch % vector_batch)));
  }

}  // namespace rod
