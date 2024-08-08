#pragma once

#include <array>

#include "Equal.hpp"
#include "Identity.hpp"
#include "Rot3.hpp"
#include "Trans.hpp"
#include "Vec3.hpp"
//#include "elastica/ErrorHandling/Assert.hpp"

namespace elastica {

  inline auto is_orthogonal(const Rot3& input_R) -> bool {
    return is_identity(input_R * trans(input_R));
  }
  auto orthogonalize(const Rot3& input_R) -> Rot3;
  // auto make_orthogonal_bases(const Vec3& arg_normal) noexcept -> Rot3;

  namespace detail {
    // ?
    auto make_orthogonal_bases_impl(Vec3 const&, Vec3 const&) noexcept
        -> std::array<Vec3, 3UL>;
    auto permute(std::array<Vec3, 3UL> const& vectors,
                 std::array<std::size_t, 3> axes) -> std::array<Vec3, 3UL>;
    auto rot_matrix_from(std::array<Vec3, 3UL> const& row_vectors) -> Rot3;
  }  // namespace detail

  auto make_orthogonal_bases_from_binormal(Vec3 const& vector) noexcept -> Rot3;
  auto make_orthogonal_bases_from_normal(Vec3 const& vector) noexcept -> Rot3;
  auto make_orthogonal_bases_from_normal_and_tangent(Vec3 const&, Vec3 const&)
      -> Rot3;
  auto make_orthogonal_bases_from_tangent(Vec3 const& vector) noexcept -> Rot3;

}  // namespace elastica
