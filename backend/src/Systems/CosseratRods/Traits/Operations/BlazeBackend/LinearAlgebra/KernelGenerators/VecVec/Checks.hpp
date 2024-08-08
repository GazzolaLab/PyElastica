#pragma once

#include "ErrorHandling/ExpectsAndEnsures.hpp"
//
#include <blaze/math/expressions/DenseMatrix.h>
//
#include <cstddef>  // size_t

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename MT1,  // blaze Matrix expression type 1
                typename MT2,  // blaze Matrix expression type 2
                typename MT3,  // blaze Vector type
                bool SO>       // Storage order
      void vector_vector_kernel_checks(
          blaze::DenseMatrix<MT1, SO>& out_vector,
          blaze::DenseMatrix<MT2, SO> const& in_vector1,
          blaze::DenseMatrix<MT3, SO> const& in_vector2) {
        constexpr std::size_t dimension(3UL);
        const std::size_t n_outputs = (*out_vector).columns();
        Expects((*out_vector).rows() == dimension);
        Expects((*in_vector1).rows() == dimension);
        Expects((*in_vector2).rows() == dimension);
        Expects(n_outputs == (*in_vector1).columns());
        Expects(n_outputs == (*in_vector2).columns());
      }

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
