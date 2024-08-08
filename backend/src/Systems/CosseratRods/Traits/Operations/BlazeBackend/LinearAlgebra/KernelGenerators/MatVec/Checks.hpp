#pragma once

#include "ErrorHandling/ExpectsAndEnsures.hpp"
//
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze_tensor/math/expressions/DenseTensor.h>
//
#include <cstddef>  // size_t

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename MT1,  // blaze Matrix expression type 1
                typename MT2,  // blaze Matrix expression type 2
                typename TT,   // blaze Tensor type
                bool SO>       // Storage order
      void matrix_vector_kernel_checks(
          blaze::DenseMatrix<MT1, SO>& out_matrix,
          blaze::DenseTensor<TT> const& in_tensor,
          blaze::DenseMatrix<MT2, SO> const& in_matrix) {
        constexpr std::size_t dimension(3UL);
        const std::size_t n_outputs = (*out_matrix).columns();

        Expects((*in_matrix).rows() == dimension);
        Expects((*out_matrix).rows() == dimension);
        Expects((*in_tensor).rows() == dimension);
        Expects((*in_tensor).pages() == dimension);
        Expects(n_outputs == (*in_matrix).columns());
        Expects(n_outputs == (*in_tensor).columns());
      }

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
