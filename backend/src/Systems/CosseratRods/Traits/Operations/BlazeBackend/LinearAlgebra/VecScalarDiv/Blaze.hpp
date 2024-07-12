#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecScalar/Checks.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/VecScalarDiv/BaseTemplate.hpp"
//
#include <blaze/Blaze.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct VecScalarDivOp<VecScalarDivKind::blaze> {
      template <typename MT1, typename MT2, typename VT, bool SO, bool TF>
      static inline auto apply(blaze::DenseMatrix<MT1, SO>& out,
                               blaze::DenseMatrix<MT2, SO> const& num,
                               blaze::DenseVector<VT, TF> const& den) noexcept
          -> void {
        auto& out_matrix(*out);
        auto& numerator(*num);
        auto& denominator(*den);

        detail::vector_scalar_kernel_checks(out_matrix, numerator, denominator);

        auto tp_denominator = blaze::trans(denominator);
        blaze::row(out_matrix, 0UL) =
            blaze::row(numerator, 0UL) / tp_denominator;
        blaze::row(out_matrix, 1UL) =
            blaze::row(numerator, 1UL) / tp_denominator;
        blaze::row(out_matrix, 2UL) =
            blaze::row(numerator, 2UL) / tp_denominator;
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
