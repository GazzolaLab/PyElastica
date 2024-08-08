#pragma once

#include "Utilities/DefineTypes.h"
#include "Utilities/ForceInline.hpp"
//
#include <blaze/math/SIMD.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename M, typename V>
    ELASTICA_ALWAYS_INLINE decltype(auto) matrix_vector_product_op(
        blaze::SIMDPack<M> const& m1, blaze::SIMDPack<M> const& m2,
        blaze::SIMDPack<M> const& m3, blaze::SIMDPack<V> const& v1,
        blaze::SIMDPack<V> const& v2, blaze::SIMDPack<V> const& v3) noexcept {
      return (*m1) * (*v1) + (*m2) * (*v2) + (*m3) * (*v3);
    }

    ELASTICA_ALWAYS_INLINE auto matrix_vector_product_op(
        elastica::real_t const m1, elastica::real_t const m2,
        elastica::real_t const m3, elastica::real_t const v1,
        elastica::real_t const v2, elastica::real_t const v3) noexcept
        -> elastica::real_t {
      return m1 * v1 + m2 * v2 + m3 * v3;
    }

    struct MatrixVectorProductOperation {
      explicit inline MatrixVectorProductOperation() = default;

      template <typename M, typename V>
      ELASTICA_ALWAYS_INLINE decltype(auto) operator()(
          M const& m1, M const& m2, M const& m3, V const& v1, V const& v2,
          V const& v3) const noexcept {
        return matrix_vector_product_op(m1, m2, m3, v1, v2, v3);
      }
    };

  }  // namespace cosserat_rod

}  // namespace elastica
