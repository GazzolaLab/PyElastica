#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/MatVec/Checks.hpp"
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename TT,         // blaze Tensor type
              bool SO>             // Storage order
    void lazy_matrix_vector_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseTensor<TT> const& input_tensor,
        blaze::DenseMatrix<MT2, SO> const& input_matrix,
        const std::size_t start_idx, const std::size_t stop_idx) {
      using ValueType = typename MT1::ElementType;
      constexpr std::size_t dimension(3UL);
      Expects(stop_idx > start_idx);

      auto& out_matrix(*output_matrix);
      auto const& in_tensor(*input_tensor);
      auto const& in_matrix(*input_matrix);

      ValueType vector_cache[dimension];

      for (auto idx = start_idx; idx < stop_idx; ++idx) {
        // Load vector
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          vector_cache[dim] = in_matrix(dim, idx);
        }

        // Loop over "pages" and process elements
        UNROLL_LOOP(dimension)
        for (auto page_idx = 0UL; page_idx < dimension; ++page_idx) {
          out_matrix(page_idx, idx) =
              op(in_tensor(page_idx, 0UL, idx), in_tensor(page_idx, 1UL, idx),
                 in_tensor(page_idx, 2UL, idx), vector_cache[0UL],
                 vector_cache[1UL], vector_cache[2UL]);
        }

      }  // idx
    }
    //**************************************************************************

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename TT,         // blaze Tensor type
              bool SO>             // Storage order
    void lazy_matrix_vector_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_matrix,
        blaze::DenseTensor<TT> const& in_tensor,
        blaze::DenseMatrix<MT2, SO> const& in_matrix) {
      detail::matrix_vector_kernel_checks(*out_matrix, *in_tensor, *in_matrix);
      lazy_matrix_vector_kernel_scalar(std::move(op), *out_matrix, *in_tensor,
                                       *in_matrix, 0UL,
                                       (*out_matrix).columns());
    }

  }  // namespace cosserat_rod

}  // namespace elastica
