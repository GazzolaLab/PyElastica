#pragma once

#include <cmath>
#include <cstdint>
//
#include <blaze/config/Vectorization.h>
#include <blaze/system/Optimizations.h>  // for padding
//
#include <blaze/math/DynamicMatrix.h>
#include <blaze_tensor/math/DynamicTensor.h>
//
#include "Systems/States/Expressions/backends/blaze/SO3PrimitiveAddAssign/BaseTemplate.hpp"
#include "Systems/States/Expressions/backends/blaze/SO3PrimitiveAddAssign/Scalar.hpp"
#include "Systems/States/Expressions/backends/blaze/Size.hpp"
//
#include "ErrorHandling/Assert.hpp"
//
#include "Utilities/Unroll.hpp"

namespace elastica {

  namespace states {

    namespace detail {

      template <>
      struct SO3AddAssign<SO3AddAssignKind::simd> {
        template <typename T, typename RHSVectorBatch>
        static auto apply(blaze::DynamicTensor<T>& lhs_matrix_batch,
                          RHSVectorBatch const& rhs_vector_batch) noexcept
            -> void {
          using ::elastica::states::size_backend;
          const std::size_t n_dofs = size_backend(lhs_matrix_batch);
          constexpr bool remainder(!::blaze::usePadding ||
                                   !blaze::IsPadded_v<RHSVectorBatch>);
          constexpr std::size_t SIMDSIZE(blaze::SIMDTrait<T>::size);

          const std::size_t i_dof_pos(
              remainder ? blaze::prevMultiple(n_dofs, SIMDSIZE) : n_dofs);
          ELASTICA_ASSERT(i_dof_pos <= n_dofs, "Invalid end calculation");

          // first do loops of size simd width

          // begin has (row, page) as the syntax
          typename blaze::DynamicTensor<T>::Iterator q_it[9UL] = {
              lhs_matrix_batch.begin(0UL, 0UL),
              lhs_matrix_batch.begin(1UL, 0UL),
              lhs_matrix_batch.begin(2UL, 0UL),
              lhs_matrix_batch.begin(0UL, 1UL),
              lhs_matrix_batch.begin(1UL, 1UL),
              lhs_matrix_batch.begin(2UL, 1UL),
              lhs_matrix_batch.begin(0UL, 2UL),
              lhs_matrix_batch.begin(1UL, 2UL),
              lhs_matrix_batch.begin(2UL, 2UL)};

          typename RHSVectorBatch::ConstIterator u_it[3UL] = {
              rhs_vector_batch.begin(0UL), rhs_vector_batch.begin(1UL),
              rhs_vector_batch.begin(2UL)};
          //
          using SIMDType = typename blaze::DynamicTensor<T>::SIMDType;

          // decide how much to unroll here
          // unrolling more probably wont help because of register pressure
          SIMDType q[9UL];
          SIMDType u[3UL];
          SIMDType c_alpha, alpha, beta;

#if !(BLAZE_SVML_MODE || BLAZE_SLEEF_MODE)
          // a temporary aligned register to pipe contents into
          alignas(::blaze::AlignmentOf_v<T>) T temp[SIMDSIZE];
#endif

          std::size_t i_dof = 0UL;

          for (; i_dof < i_dof_pos; i_dof += SIMDSIZE) {
            {
              // 1. load data into registers

              // 1.1 load u
              UNROLL_LOOP(3UL)
              for (std::size_t dim = 0UL; dim < 3UL; ++dim) {
                u[dim] = u_it[dim].load();
              }

              // 1.2 load q
              UNROLL_LOOP(9UL)
              for (std::size_t dim = 0UL; dim < 9UL; ++dim) {
                q[dim] = q_it[dim].load();
              }
            }

            {
              // 2. compute the angle of rotation
              // blaze dispatches to std
              beta = ::blaze::sqrt(u[0UL] * u[0UL] + u[1UL] * u[1UL] +
                                   u[2UL] * u[2UL]);

              // 3. normalize the axis of rotation
              /*Clang-Tidy: Use range-based for loop instead*/
              UNROLL_LOOP(3UL)
              for (std::size_t dim = 0UL; dim < 3UL; ++dim) { /* NOLINT */
                // bad access pattern
                // TODO : refactor magic number
                auto arg = (beta + ::blaze::set(T(1e-14)));
                u[dim] /= arg;
              }

              // compute trigonometric entities
#if BLAZE_SVML_MODE || BLAZE_SLEEF_MODE
              c_alpha = ::blaze::cos(beta);
              beta = ::blaze::sin(beta);
#else
              // no intrinsics, so we unpack the SIMD vector, compute sin and
              // cos on each via regular math functions and then repack it
#pragma unroll SIMDSIZE
              for (std::size_t simd_idx = 0UL; simd_idx < SIMDSIZE; ++simd_idx)
                temp[simd_idx] = ::blaze::cos(beta[simd_idx]);

              c_alpha = ::blaze::loada(temp);

#pragma unroll SIMDSIZE
              for (std::size_t simd_idx = 0UL; simd_idx < SIMDSIZE; ++simd_idx)
                temp[simd_idx] = ::blaze::sin(beta[simd_idx]);

              beta = ::blaze::loada(temp);
#endif
              alpha = ::blaze::set(T(1.0)) - c_alpha;
            }

            {
              // 4. compute the rotated batch
              /*Clang-Tidy: Use range-based for loop instead*/
              UNROLL_LOOP(3UL)
              for (std::size_t j_dim = 0UL; j_dim < 3UL; ++j_dim) { /* NOLINT */
                // Force evaluate it
                const auto com = (q[j_dim] * u[0UL] + q[3UL + j_dim] * u[1UL] +
                                  q[6UL + j_dim] * u[2UL])
                                     .eval();

                q_it[0UL + j_dim].stream(
                    c_alpha * q[j_dim] + alpha * u[0UL] * com +
                    beta * (q[3UL + j_dim] * u[2UL] - q[6UL + j_dim] * u[1UL]));

                q_it[3UL + j_dim].stream(
                    c_alpha * q[3UL + j_dim] + alpha * u[1UL] * com +
                    beta * (q[6UL + j_dim] * u[0UL] - q[j_dim] * u[2UL]));

                q_it[6UL + j_dim].stream(
                    c_alpha * q[6UL + j_dim] + alpha * u[2UL] * com +
                    beta * (q[j_dim] * u[1UL] - q[3UL + j_dim] * u[0UL]));

              }  // jdim
            }

            {
              // 5. advance all the iterators to the next SIMD lane
              /*Clang-Tidy: Use range-based for loop instead*/
              UNROLL_LOOP(3UL)
              for (std::size_t dim = 0UL; dim < 3UL; ++dim) /* NOLINT */
                u_it[dim] += SIMDSIZE;

              UNROLL_LOOP(9UL)
              for (std::size_t dim = 0UL; dim < 9UL; ++dim) /* NOLINT */
                q_it[dim] += SIMDSIZE;
            }

          }  // i_dof

          // then do the last loops, peeling them off to serial scalar
          // implementation UNTESTED
          if (remainder)
            SO3AddAssign<SO3AddAssignKind::scalar>::internal_apply(
                lhs_matrix_batch, rhs_vector_batch, i_dof, n_dofs);
        }
      };

    }  // namespace detail

  }  // namespace states

}  // namespace elastica
