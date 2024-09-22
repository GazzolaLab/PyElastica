#pragma once

#include "Utilities/DefineTypes.h"
#include "Utilities/ForceInline.hpp"
//
#include <blaze/math/SIMD.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename T>
    ELASTICA_ALWAYS_INLINE decltype(auto) quadrature_op(
        blaze::SIMDPack<T> const& e_1, blaze::SIMDPack<T> const& e_2) noexcept {
      using V = blaze::ValueType_t<T>;
      return ::blaze::set(V(0.5)) * (*e_1 + *e_2);
    }

    ELASTICA_ALWAYS_INLINE auto quadrature_op(
        elastica::real_t const e_1, elastica::real_t const e_2) noexcept
        -> elastica::real_t {
      return ::elastica::real_t(0.5) * (e_1 + e_2);
    }

    struct QuadratureOperation {
      explicit inline QuadratureOperation() = default;

      template <typename T>
      ELASTICA_ALWAYS_INLINE decltype(auto) operator()(
          T const& e_1, T const& e_2) const noexcept {
        return quadrature_op(e_1, e_2);
      }
    };

  }  // namespace cosserat_rod

}  // namespace elastica
