#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecVec/Checks.hpp"
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/expressions/DenseMatrix.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename MT3,        // blaze Vector type
              bool SO>             // Storage order
    void lazy_vector_vector_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix1,
        blaze::DenseMatrix<MT3, SO> const& input_matrix2,
        const std::size_t start_idx, const std::size_t stop_idx) {
      using ValueType = typename MT1::ElementType;
      constexpr std::size_t dimension(3UL);
      Expects(stop_idx > start_idx);

      auto& out_matrix(*output_matrix);
      auto const& in_matrix1(*input_matrix1);
      auto const& in_matrix2(*input_matrix2);

      ValueType vector_cache[2UL][dimension];

      for (auto idx = start_idx; idx < stop_idx; ++idx) {
        // Load vector
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          vector_cache[0UL][dim] = in_matrix1(dim, idx);
          vector_cache[1UL][dim] = in_matrix2(dim, idx);
        }

        // Can use index apply, but need to capture by reference. Eschew that
        // approach in favor of macro-based simplicity.

#define PERFORM_OP(INDEX)                                                     \
  out_matrix(INDEX, idx) = op.template operator()<INDEX>(                     \
      vector_cache[0UL][0UL], vector_cache[0UL][1UL], vector_cache[0UL][2UL], \
      vector_cache[1UL][0UL], vector_cache[1UL][1UL], vector_cache[1UL][2UL])

        PERFORM_OP(0UL);
        PERFORM_OP(1UL);
        PERFORM_OP(2UL);

#undef PERFORM_OP
      }  // dofs
    }

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename MT3,        // blaze Vector type
              bool SO>             // Storage order
    void lazy_vector_vector_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_vector,
        blaze::DenseMatrix<MT2, SO> const& in_vector1,
        blaze::DenseMatrix<MT3, SO> const& in_vector2) {
      detail::vector_vector_kernel_checks(*out_vector, *in_vector1,
                                          *in_vector2);
      lazy_vector_vector_kernel_scalar(std::move(op), *out_vector, *in_vector1,
                                       *in_vector2, 0UL,
                                       (*out_vector).columns());
    }

  }  // namespace cosserat_rod

}  // namespace elastica
