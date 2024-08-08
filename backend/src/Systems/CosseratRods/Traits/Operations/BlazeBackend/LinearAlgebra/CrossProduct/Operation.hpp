#pragma once

#include "Utilities/DefineTypes.h"
#include "Utilities/ForceInline.hpp"
//
#include <blaze/math/SIMD.h>
#include <utility>  // size_t

namespace elastica {

  namespace cosserat_rod {

    template <typename M, typename V>
    ELASTICA_ALWAYS_INLINE decltype(auto) vector_cross_product_op(
        std::integral_constant<std::size_t, 0UL> /*meta*/,
        blaze::SIMDPack<M> const&, blaze::SIMDPack<M> const& m2,
        blaze::SIMDPack<M> const& m3, blaze::SIMDPack<V> const&,
        blaze::SIMDPack<V> const& v2, blaze::SIMDPack<V> const& v3) noexcept {
      return (*m2) * (*v3) - (*m3) * (*v2);
    }

    template <typename M, typename V>
    ELASTICA_ALWAYS_INLINE decltype(auto) vector_cross_product_op(
        std::integral_constant<std::size_t, 1UL> /*meta*/,
        blaze::SIMDPack<M> const& m1, blaze::SIMDPack<M> const&,
        blaze::SIMDPack<M> const& m3, blaze::SIMDPack<V> const& v1,
        blaze::SIMDPack<V> const&, blaze::SIMDPack<V> const& v3) noexcept {
      return (*m3) * (*v1) - (*m1) * (*v3);
    }

    template <typename M, typename V>
    ELASTICA_ALWAYS_INLINE decltype(auto) vector_cross_product_op(
        std::integral_constant<std::size_t, 2UL> /*meta*/,
        blaze::SIMDPack<M> const& m1, blaze::SIMDPack<M> const& m2,
        blaze::SIMDPack<M> const&, blaze::SIMDPack<V> const& v1,
        blaze::SIMDPack<V> const& v2, blaze::SIMDPack<V> const&) noexcept {
      return (*m1) * (*v2) - (*m2) * (*v1);
    }

    ELASTICA_ALWAYS_INLINE auto vector_cross_product_op(
        std::integral_constant<std::size_t, 0UL> /*meta*/,
        elastica::real_t const, elastica::real_t const m2,
        elastica::real_t const m3, elastica::real_t const,
        elastica::real_t const v2, elastica::real_t const v3) noexcept
        -> elastica::real_t {
      return m2 * v3 - m3 * v2;
    }

    ELASTICA_ALWAYS_INLINE auto vector_cross_product_op(
        std::integral_constant<std::size_t, 1UL> /*meta*/,
        elastica::real_t const m1, elastica::real_t const,
        elastica::real_t const m3, elastica::real_t const v1,
        elastica::real_t const, elastica::real_t const v3) noexcept
        -> elastica::real_t {
      return m3 * v1 - m1 * v3;
    }

    ELASTICA_ALWAYS_INLINE auto vector_cross_product_op(
        std::integral_constant<std::size_t, 2UL> /*meta*/,
        elastica::real_t const m1, elastica::real_t const m2,
        elastica::real_t const, elastica::real_t const v1,
        elastica::real_t const v2, elastica::real_t const) noexcept
        -> elastica::real_t {
      return m1 * v2 - m2 * v1;
    }

    struct VectorCrossProductOperation {
      explicit inline VectorCrossProductOperation() = default;

      template <std::size_t Idx, typename M, typename V>
      ELASTICA_ALWAYS_INLINE decltype(auto) operator()(
          M const& m1, M const& m2, M const& m3, V const& v1, V const& v2,
          V const& v3) const noexcept {
        return vector_cross_product_op(
            std::integral_constant<std::size_t, Idx>{}, m1, m2, m3, v1, v2, v3);
      }
    };

  }  // namespace cosserat_rod

}  // namespace elastica
