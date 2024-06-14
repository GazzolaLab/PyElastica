#pragma once

#include "blaze/Blaze.h"
#include "blaze_tensor/Blaze.h"

namespace elastica {
  // LOG AND EXP DEFINED AS PER ELASTICA'S ROTATIONAL DIRECTIONS!

  //**************************************************************************
  /*!\brief Batchwise matrix logarithmic operator.
  //
  // \details
  // Batchwise for rotation matrix R computes the corresponding rotation
  // axis vector {theta (u)} using the matrix log() operator: \n
  // if theta == 0:
  //      theta (u) = 0 \n
  // else:
  //      theta (u) = -theta * inv_skew[R - transpose(R)] / sin(theta) \n
  // where theta = acos(0.5 * (trace(R) - 1)) and inv_skew[] corresponds
  // to an inverse skew symmetric mapping from a skew symmetric matrix M
  // to vector V as:
  // <pre>
       |0 -z y|        |x|
   M = |z 0 -x| to V = |y|
       |-y x 0|        |z|
   </pre>
  //
  // \example
  // The following shows a typical use of the log_batch function
  // with the expected (correct) result also shown.
  // \snippet test_rotations.cpp log_batch_example
  //
  // \param[out] void/None
  // \param[in] rot_matrix_batch(3, 3, n_elems) rotation matrix batch
  //
  // \return rot_axis_vector_batch(3, n_elems) rotation axis vector batch
  //
  // \see fill later?
  */
  template <typename MT, typename TT>  // blaze Tensor expression type
  void batch_inv_rotate(MT& rot_axis_vector_batch, const TT& rot_matrix_batch) {
    constexpr std::size_t dimension(3UL);
    const std::size_t n_elems = rot_matrix_batch.columns();
    using ValueType = typename TT::ElementType;
    assert(rot_matrix_batch.pages() == dimension);
    assert(rot_matrix_batch.rows() == dimension);
    assert(rot_axis_vector_batch.rows() == dimension);
    assert(rot_axis_vector_batch.columns() == n_elems);

    /* hardcoded to avoid external memory passing */
    /* also now operates only on views */
    auto rot_matrix_trace_batch =
        blaze::row(blaze::pageslice(rot_matrix_batch, 0UL), 0UL) +
        blaze::row(blaze::pageslice(rot_matrix_batch, 1UL), 1UL) +
        blaze::row(blaze::pageslice(rot_matrix_batch, 2UL), 2UL);
    /* theta = acos((tr(R) - 1) / 2) */
    auto theta_batch = blaze::acos(ValueType(0.5) *
                                   (rot_matrix_trace_batch - ValueType(1.0) -
                                    ValueType(1e-12)));  // TODO refactor 1e-12
    auto R_minus_RT =
        rot_matrix_batch - blaze::trans(rot_matrix_batch, {1, 0, 2}); // TODO Something is wrong
    /* theta (u) = -theta * inv_skew([R - RT]) / 2 sin(theta) */
    for (std::size_t i(0UL); i < dimension; ++i) {
      auto inv_skew_symmetric_map = blaze::row(
          blaze::pageslice(R_minus_RT, (i + 2UL) % dimension),
          (i + 1UL) % dimension);  // % for the cyclic rotation of indices
      blaze::row(rot_axis_vector_batch, i) =
          ValueType(-0.5) * theta_batch * inv_skew_symmetric_map /
          (blaze::sin(theta_batch) +
           ValueType(1e-14));  // TODO refactor 1e-14
    }
  }
  //**************************************************************************

  //**************************************************************************
  /*!\brief Batchwise matrix exponential operator.
  //
  // \details
  // Batchwise for rotation axis vector {theta u} computes the corresponding
  // rotation matrix R using the matrix exp() operator (Rodrigues formula): \n
  // R = I - sin(theta) * U + (1 - cos(theta)) * U * U \n
  // Here a different tensorial form is implemented as follows:
  // if i == j:
  //      R{ij} = cos(theta) + (1 - cos(theta)) * (u{i})^2 \n
  // else:
  //      R{ij} = (1 - cos(theta)) * u{i} * u{j} - sin(theta) *
  //      skew_sym[u]{ij} \n
  // where I is the identity matrix and skew_sym[] corresponds to a skew
  // symmetric mapping from a vector V to a skew symmetric matrix M as:
  // <pre>
       |x|        |0 -z y|
   V = |y| to M = |z 0 -x|
       |z|        |-y x 0|
  </pre>
  //
  // \example
  // The following shows a typical use of the exp_batch function
  // with the expected (correct) result also shown.
  // \snippet test_rotations.cpp exp_batch_example
  //
  // \param[out] rot_matrix_batch(3, 3, n_elems) rotation matrix batch
  // \param[in] rot_axis_vector_batch(3, n_elems) rotation axis vector batch
  //
  // \return void/None
  //
  // \see fill later?
  */
  template <typename TT,  // blaze Tensor expression type
      typename MT>  // blaze Matrix expression type
  void exp_batch(TT& rot_matrix_batch, const MT& rot_axis_vector_batch) {
    constexpr std::size_t dimension(3UL);
    //const std::size_t n_elems = rot_axis_vector_batch.columns();
    using ValueType = typename MT::ElementType;
    assert(rot_axis_vector_batch.rows() == dimension);
    assert(rot_matrix_batch.pages() == dimension);
    assert(rot_matrix_batch.rows() == dimension);
    //assert(rot_matrix_batch.columns() == n_elems);

    auto theta_batch = blaze::sqrt(
        blaze::sum<blaze::columnwise>(blaze::pow(rot_axis_vector_batch, 2)));
    /* TODO refactor 1e-14 */
    auto unit_rot_axis_vector_batch =
        rot_axis_vector_batch %
        blaze::expand(blaze::pow(theta_batch + ValueType(1e-14), -1),
                      dimension);
    for (std::size_t i(0UL); i < dimension; ++i) {
      // if i == j:
      //      R{ij} = cos(theta) + (1 - cos(theta)) * (u{i})^2
      blaze::row(blaze::pageslice(rot_matrix_batch, i), i) =
          blaze::cos(theta_batch) +
          (ValueType(1.0) - blaze::cos(theta_batch)) *
              blaze::pow(blaze::row(unit_rot_axis_vector_batch, i), 2);
      // else:
      //      R{ij} = (1 - cos(theta)) * u{i} * u{j} - sin(theta) *
      //      skew_sym[u]{ij}
      for (std::size_t j : {i + 1UL, i + 2UL}) {
        auto skew_symmetric_map =
            std::pow(-1,
                     j - i) *  // Sign-bit to check order of entries
            blaze::row(unit_rot_axis_vector_batch,
                       dimension - i - (j % dimension));  // % source index;
        blaze::row(blaze::pageslice(rot_matrix_batch, i), j % dimension) =
            (ValueType(1.0) - blaze::cos(theta_batch)) *
                blaze::row(unit_rot_axis_vector_batch, i) *
                blaze::row(unit_rot_axis_vector_batch, j % dimension) -
            blaze::sin(theta_batch) * skew_symmetric_map;
      }
    }
  }
  //**************************************************************************

} // namespace elastica
