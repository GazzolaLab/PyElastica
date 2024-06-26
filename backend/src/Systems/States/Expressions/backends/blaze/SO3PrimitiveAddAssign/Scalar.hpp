#pragma once

#include <blaze_tensor/math/DynamicTensor.h>
#include <cmath>
#include <cstdint>

#include "Systems/States/Expressions/backends/blaze/SO3PrimitiveAddAssign/BaseTemplate.hpp"
#include "Systems/States/Expressions/backends/blaze/Size.hpp"
//
#include "Utilities/Unroll.hpp"

namespace elastica {

  namespace states {

    namespace detail {

      template <>
      struct SO3AddAssign<SO3AddAssignKind::scalar> {
        template <typename T, typename RHSVectorBatch>
        static auto internal_apply(blaze::DynamicTensor<T>& lhs_matrix_batch,
                                   RHSVectorBatch const& rhs_vector_batch,
                                   const std::size_t index_start,
                                   const std::size_t index_fin) noexcept
            -> void {
          // 0. allocate memory for rolling window
          T q[9UL];
          /*
           * structure is
           * q0 q3 q6
           * q1 q4 q7
           * q2 q5 q8
           */
          T u[3UL];
          // T theta, alpha, beta;  // doesnt matter the extra memory

          for (std::size_t i_dof = index_start; i_dof < index_fin; ++i_dof) {
            // 1. load data into registers
            UNROLL_LOOP(3UL)
            for (std::size_t dim = 0UL; dim < 3UL; ++dim) {
              // bad access pattern
              u[dim] = rhs_vector_batch(dim, i_dof);
            }

            UNROLL_LOOP(3UL)  // Needs to be timed
            for (std::size_t k_dim = 0UL; k_dim < 3UL; ++k_dim) {
              UNROLL_LOOP(3UL)
              for (std::size_t j_dim = 0UL; j_dim < 3UL; ++j_dim) {
                // bad access pattern
                // convention according to SO3PrimitiveAssign
                q[3UL * k_dim + j_dim] = lhs_matrix_batch(k_dim, j_dim, i_dof);
              }
            }

            // Now work only with registers

            // 2. compute the angle of rotation
            // blaze dispatches to std
            const T theta = ::std::sqrt(u[0UL] * u[0UL] + u[1UL] * u[1UL] +
                                        u[2UL] * u[2UL]);
            const T c_alpha = ::std::cos(theta);
            const T alpha = T(1.0) - c_alpha;
            const T beta = ::std::sin(theta);

            // 3. normalize the axis of rotation
            /*Clang-Tidy: Use range-based for loop instead*/
            UNROLL_LOOP(3UL)
            for (std::size_t dim = 0UL; dim < 3UL; ++dim) { /* NOLINT */
              // bad access pattern
              u[dim] /= (theta + 1e-14);  // TODO : refactor magic number
            }

            /*Clang-Tidy: Use range-based for loop instead*/
            UNROLL_LOOP(3UL)
            for (std::size_t j_dim = 0UL; j_dim < 3UL; ++j_dim) { /* NOLINT */
              const T com = q[j_dim] * u[0UL] + q[3UL + j_dim] * u[1UL] +
                            q[6UL + j_dim] * u[2UL];

              lhs_matrix_batch(0UL, j_dim, i_dof) =
                  c_alpha * q[j_dim] + alpha * u[0UL] * com +
                  beta * (q[3UL + j_dim] * u[2UL] - q[6UL + j_dim] * u[1UL]);

              lhs_matrix_batch(1UL, j_dim, i_dof) =
                  c_alpha * q[3UL + j_dim] + alpha * u[1UL] * com +
                  beta * (q[6UL + j_dim] * u[0UL] - q[j_dim] * u[2UL]);

              lhs_matrix_batch(2UL, j_dim, i_dof) =
                  c_alpha * q[6UL + j_dim] + alpha * u[2UL] * com +
                  beta * (q[j_dim] * u[1UL] - q[3UL + j_dim] * u[0UL]);
            }  // j_dim
          }    // i_dof
        }

        template <typename T, typename RHSVectorBatch>
        static inline auto apply(
            blaze::DynamicTensor<T>& lhs_matrix_batch,
            RHSVectorBatch const& rhs_vector_batch) noexcept -> void {
          using ::elastica::states::size_backend;
          internal_apply(lhs_matrix_batch, rhs_vector_batch, 0UL,
                         size_backend(lhs_matrix_batch));
        }
      };

    }  // namespace detail

  }  // namespace states

}  // namespace elastica
