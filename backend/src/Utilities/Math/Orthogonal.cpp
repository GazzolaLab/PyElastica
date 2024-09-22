#include "Utilities/Math/Orthogonal.hpp"

#include <stdexcept>

#include "Utilities/Math/Dot.hpp"
#include "Utilities/Math/Length.hpp"
#include "Utilities/Math/Normalize.hpp"
#include "Utilities/Math/Zero.hpp"

namespace elastica {

  auto orthogonalize(const Rot3& input_R) -> Rot3 {
    // TODO : very general, need not do a QR
    //  https://stackoverflow.com/a/23082112
    throw std::runtime_error("Not implemented");
    //    Rot3 Q, R;
    //    blaze::qr(input_R, Q, R);
    //    ELASTICA_ASSERT(input_R == Q * R, "QR fails");
    return input_R;
  }

  namespace detail {
    using _internal_RT = std::array<Vec3, 3UL>;

    // v1 is normal, v2 is binormal, v3 is tangent
    // so normal x binormal => tangent
    auto make_orthogonal_bases_impl(Vec3 const& normal_guess,
                                    Vec3 const& binormal_guess = {0.8, 0.6,
                                                                  0.0}) noexcept
        -> _internal_RT {
      Vec3 v1 = normalize(normal_guess);

      // Any initial guess thats not in the normal direction
      Vec3 v2 = binormal_guess;

      // if v1 % v2 is near 0.0, then this call is invalid.
      // So don't normalize here, rather normalize at the end
      Vec3 v3(v1 % v2);

      v2 = v3 % v1;
      // If its collinear, iterate till its not
      // should be parallel threshold but thats in simluator
      // and we want to make it not depend on simluator
      if /*while*/ (is_zero_length(v2)) {
        v2 = {0.6, -0.8, 0.0};
        v3 = normalize(v1 % v2);
        v2 = v3 % v1;
      }
      return {v1, normalize(v2), normalize(v3)};
    }

    auto permute(_internal_RT const& vectors, std::array<std::size_t, 3> axes)
        -> _internal_RT {
      return {vectors[axes[0UL]], vectors[axes[1UL]], vectors[axes[2UL]]};
    }

    auto rot_matrix_from(_internal_RT const& row_vectors) -> Rot3 {
      return Rot3{
          {row_vectors[0UL][0UL], row_vectors[0UL][1UL], row_vectors[0UL][2UL]},
          {row_vectors[1UL][0UL], row_vectors[1UL][1UL], row_vectors[1UL][2UL]},
          {row_vectors[2UL][0UL], row_vectors[2UL][1UL],
           row_vectors[2UL][2UL]}};
    }
  }  // namespace detail

  auto make_orthogonal_bases_from_binormal(Vec3 const& vector) noexcept
      -> Rot3 {
    // bases[1] always contains the vector
    // bases[1] x bases[2] = bases[3]
    // bases[0] , bases[1], bases[2]
    constexpr std::size_t d1 = 2UL;
    constexpr std::size_t d2 = 0UL;
    constexpr std::size_t d3 = 1UL;

    using namespace detail;
    return rot_matrix_from(
        permute(make_orthogonal_bases_impl(vector), {d1, d2, d3}));
  }

  auto make_orthogonal_bases_from_normal(Vec3 const& vector) noexcept -> Rot3 {
    // bases[1] always contains the vector
    // bases[1] x bases[2] = bases[3]
    // bases[0] , bases[1], bases[2]
    constexpr std::size_t d1 = 0UL;
    constexpr std::size_t d2 = 1UL;
    constexpr std::size_t d3 = 2UL;

    using namespace detail;
    return rot_matrix_from(
        permute(make_orthogonal_bases_impl(vector), {d1, d2, d3}));
  }

  auto make_orthogonal_bases_from_normal_and_tangent(
      Vec3 const& proposed_normal, Vec3 const& proposed_tangent) -> Rot3 {
    if (not is_zero(::elastica::dot(proposed_normal, proposed_tangent))) {
      throw std::logic_error(
          "The proposed normal and tangent are non-orthogonal!");
    }
    // bases[1] always contains the vector
    // bases[1] x bases[2] = bases[3]
    // bases[0] , bases[1], bases[2]
    constexpr std::size_t d1 = 0UL;
    constexpr std::size_t d2 = 1UL;
    constexpr std::size_t d3 = 2UL;

    using namespace detail;
    return rot_matrix_from(
        permute(make_orthogonal_bases_impl(proposed_normal,
                                           proposed_tangent % proposed_normal),
                {d1, d2, d3}));
  }

  auto make_orthogonal_bases_from_tangent(Vec3 const& vector) noexcept -> Rot3 {
    // bases[1] always contains the vector
    // bases[1] x bases[2] = bases[3]
    // bases[0] , bases[1], bases[2]
    constexpr std::size_t d1 = 1UL;
    constexpr std::size_t d2 = 2UL;
    constexpr std::size_t d3 = 0UL;

    using namespace detail;
    return rot_matrix_from(
        permute(make_orthogonal_bases_impl(vector), {d1, d2, d3}));
  }

  //  auto make_orthogonal_bases(const Vec3& arg_normal) noexcept -> Rot3 {
  //    // TODO : Doesn't work, seems like a bug in blaze
  //    Vec3 normal = normalize(arg_normal);
  //
  //    // Any initial guess thats not in the normal direction
  //    Vec3 binormal{0.8, 0.6, 0.0};
  //
  //    Vec3 tangent(normalize(binormal % normal));
  //
  //    binormal = normal % tangent;
  //    // If its collinear, iterate till its not
  //    // should be parallel threshold but thats in simluator
  //    // and we want to make it not depend on simluator
  //    if /*while*/ (equal(length(binormal), 0.0)) {
  //      // TODO : Arbitrary, revisit
  //      //      std::cout << binormal << " " << tangent << std::endl;
  //      binormal = {0.6, -0.8, 0.0};
  //      tangent = normalize(binormal % normal);
  //      binormal = normal % tangent;
  //    }
  //
  //    // Normal is in the 2nd row, d2
  //    return Rot3{{binormal[0], binormal[1], binormal[2]},
  //                {normal[0], normal[1], normal[2]},
  //                {tangent[0], tangent[1], tangent[2]}};
  //  }

}  // namespace elastica
