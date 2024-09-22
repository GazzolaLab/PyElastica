#pragma once

#include <cmath>
#include <cstdint>

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/BaseTemplate.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Restrict.hpp"
#include "Utilities/Unroll.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <>
      struct InvRotateDivideOp<InvRotateDivideKind::scalar> {
        template <typename MT,  // blaze Matrix expression type
                  typename TT,  // blaze Tensor expression type
                  typename VT>  // blaze Matrix expression type
        static inline auto internal_apply(MT& rot_axis_vector,
                                          const TT& rot_matrix,
                                          const VT& span_vector,
                                          const std::size_t index_start,
                                          const std::size_t index_end) noexcept
            -> void {
          using T = typename TT::ElementType;

          // allocate rolling memory
          T q_first[9UL];
          T q_second[9UL];
          /*
           * structure is
           * q0 q3 q6
           * q1 q4 q7
           * q2 q5 q8
           */
          T* ELASTICA_RESTRICT Q_curr = q_first;
          T* ELASTICA_RESTRICT Q_next = q_second;

          {
            UNROLL_LOOP(3UL)  // needs to be timed
            for (std::size_t k_dim = 0UL; k_dim < 3UL; ++k_dim) {
              UNROLL_LOOP(3UL)  // needs to be timed
              for (std::size_t j_dim = 0UL; j_dim < 3UL; ++j_dim) {
                // bad access pattern
                // convention according to SO3PrimitiveAssign
                Q_curr[3UL * k_dim + j_dim] =
                    rot_matrix(k_dim, j_dim, index_start);
              }
            }
          }

          for (std::size_t i_dof = index_start; i_dof < index_end; ++i_dof) {
            // begin to loop
            // i_dof is the current index being processed

            // load data for the next iteration
            {
              UNROLL_LOOP(3UL)  // needs to be timed
              for (std::size_t k_dim = 0UL; k_dim < 3UL; ++k_dim) {
                UNROLL_LOOP(3UL)  // needs to be timed
                for (std::size_t j_dim = 0UL; j_dim < 3UL; ++j_dim) {
                  Q_next[3UL * k_dim + j_dim] =
                      rot_matrix(k_dim, j_dim, i_dof + 1UL);
                }  // j
              }    // k
            }      // load

            /*
             * R = Q_i Q^T_{i+1} (elementwise)
             * theta = acos((tr(R) - 1) / 2)
             */
            // can be reduced but whatever
            T const rot_matrix_trace =
                Q_curr[0] * Q_next[0] + Q_curr[1] * Q_next[1] +
                Q_curr[2] * Q_next[2] + Q_curr[3] * Q_next[3] +
                Q_curr[4] * Q_next[4] + Q_curr[5] * Q_next[5] +
                Q_curr[6] * Q_next[6] + Q_curr[7] * Q_next[7] +
                Q_curr[8] * Q_next[8];

            constexpr T rest(T(0.5) + T(1e-10));
            T const theta = std::acos(T(0.5) * rot_matrix_trace - rest);

            T const st = T(0.5) * theta / std::sin(theta + T(1e-14)) /
                         span_vector[i_dof];
            rot_axis_vector(0, i_dof) =
                st * (-Q_curr[3] * Q_next[6] - Q_curr[4] * Q_next[7] -
                      Q_curr[5] * Q_next[8] + Q_curr[6] * Q_next[3] +
                      Q_curr[7] * Q_next[4] + Q_curr[8] * Q_next[5]);

            rot_axis_vector(1, i_dof) =
                st * (Q_curr[0] * Q_next[6] + Q_curr[1] * Q_next[7] +
                      Q_curr[2] * Q_next[8] - Q_curr[6] * Q_next[0] -
                      Q_curr[7] * Q_next[1] - Q_curr[8] * Q_next[2]);

            rot_axis_vector(2, i_dof) =
                st * (-Q_curr[0] * Q_next[3] - Q_curr[1] * Q_next[4] -
                      Q_curr[2] * Q_next[5] + Q_curr[3] * Q_next[0] +
                      Q_curr[4] * Q_next[1] + Q_curr[5] * Q_next[2]);

            std::swap(Q_curr, Q_next);

          }  // i_dof
        }    // apply

        template <typename MT,  // blaze Matrix expression type
                  typename TT,  // blaze Tensor expression type
                  typename VT>  // blaze Matrix expression type
        static ELASTICA_ALWAYS_INLINE auto apply(MT& rot_axis_vector,
                                                 const TT& rot_matrix,
                                                 const VT& span_vector) noexcept
            -> void {
          // assert that rot_matrix.columns() == rot_axis_vector.columns() + 1
          // in the public interface
          internal_apply(rot_axis_vector, rot_matrix, span_vector, 0UL,
                         rot_axis_vector.columns());
        }
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
