#pragma once

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
              bool SO>             // Storage order
    void lazy_twopoint_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix,
        const std::size_t start_idx, const std::size_t stop_idx) {
      using ValueType = typename MT1::ElementType;
      constexpr std::size_t dimension(3UL);
      const std::size_t stop = stop_idx - 1UL;
      Expects(stop_idx > start_idx);

      auto const& in_matrix(*input_matrix);
      auto& out_matrix(*output_matrix);

      UNROLL_LOOP(dimension)
      for (auto dim = 0UL; dim < dimension; ++dim) {
        // edge case first
        {
          out_matrix(dim, start_idx) = op(
              in_matrix(dim, start_idx),
              (start_idx ? in_matrix(dim, start_idx - 1UL) : ValueType(0.0)));
        }

        for (auto idx = (start_idx + 1UL); idx < stop; ++idx) {
          // middle case
          out_matrix(dim, idx) =
              op(in_matrix(dim, idx), in_matrix(dim, idx - 1UL));
        }

        // edge case list
        {
          auto const idx = stop;
          out_matrix(dim, idx) =
              op((idx == in_matrix.columns()) ? ValueType(0.0)
                                              : in_matrix(dim, idx),
                 in_matrix(dim, idx - 1UL));
        }
      }
    }
    //**************************************************************************

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              bool SO>             // Storage order
    void lazy_twopoint_kernel_scalar(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_matrix,
        blaze::DenseMatrix<MT2, SO> const& in_matrix) {
      constexpr std::size_t dimension(3UL);
      const std::size_t n_outputs = (*out_matrix).columns();

      Expects((*in_matrix).rows() == dimension);
      Expects((*out_matrix).rows() == dimension);
      Expects(n_outputs == (*in_matrix).columns() + 1UL);

      lazy_twopoint_kernel_scalar(std::move(op), *out_matrix, *in_matrix, 0UL,
                                  n_outputs);
    }

  }  // namespace cosserat_rod

}  // namespace elastica
