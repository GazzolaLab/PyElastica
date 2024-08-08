#pragma once

#include "ErrorHandling/ExpectsAndEnsures.hpp"
//
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>
//
#include <cstddef>  // size_t

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename MT1,  // blaze Matrix expression type 1
                typename MT2,  // blaze Matrix expression type 2
                typename VT,   // blaze Vector type
                bool SO,       // Storage order
                bool TF>
      void vector_scalar_kernel_checks(
          blaze::DenseMatrix<MT1, SO>& out_vector,
          blaze::DenseMatrix<MT2, SO> const& in_vector,
          blaze::DenseVector<VT, TF> const& in_scalar) {
        constexpr std::size_t dimension(3UL);
        const std::size_t n_outputs = (*out_vector).columns();
        Expects((*out_vector).rows() == dimension);
        Expects((*in_vector).rows() == dimension);
        Expects(n_outputs == (*in_vector).columns());
        Expects(n_outputs == (*in_scalar).size());
      }

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
