#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecScalar/Checks.hpp"
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename VT,         // blaze Vector type
              bool SO,             // Storage order
              bool TF>
    void lazy_vector_scalar_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix,
        blaze::DenseVector<VT, TF> const& input_vector,
        const std::size_t start_idx, const std::size_t stop_idx) {
      constexpr std::size_t dimension(3UL);
      Expects(stop_idx > start_idx);

      auto& out_matrix(*output_matrix);
      auto const& in_matrix(*input_matrix);
      auto const& in_vector(*input_vector);

      for (auto idx = start_idx; idx < stop_idx; ++idx) {
        // Load vector
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          out_matrix(dim, idx) = op(in_matrix(dim, idx), in_vector[idx]);
        }
      }
    }

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename VT,         // blaze Vector type
              bool SO,             // Storage order
              bool TF>
    void lazy_vector_scalar_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_vector,
        blaze::DenseMatrix<MT2, SO> const& in_vector,
        blaze::DenseVector<VT, TF> const& in_scalar) {
      detail::vector_scalar_kernel_checks(*out_vector, *in_vector, *in_scalar);
      lazy_vector_scalar_kernel_scalar(std::move(op), *out_vector, *in_vector,
                                       *in_scalar, 0UL,
                                       (*out_vector).columns());
    }

  }  // namespace cosserat_rod

}  // namespace elastica
