#pragma once

#include <cmath>
#include <cstdint>
//
#include <blaze/config/Vectorization.h>
#include <blaze/math/Aliases.h>
#include <blaze/system/Optimizations.h>  // for padding

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/BaseTemplate.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/Scalar.hpp"
#include "Utilities/Restrict.hpp"
//
#include "ErrorHandling/Assert.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/Unroll.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <>
      struct InvRotateDivideOp<InvRotateDivideKind::simd> {
        template <typename MT,  // blaze Matrix expression type
                  typename TT,  // blaze Tensor expression type
                  typename VT>  // blaze Matrix expression type
        static inline auto apply(MT& rot_axis_vector, const TT& rot_matrix,
                                 const VT& span_vector) noexcept -> void {
          /*
           * 1. main loop
           * 2. peel the rest and do scalar operations
           */
          // n_elem - 1
          const std::size_t blocksize = rot_axis_vector.columns();

          using T = blaze::ElementType_t<MT>;
          using SIMDType = blaze::SIMDType_t<MT>;
          constexpr std::size_t SIMDSIZE(SIMDType::size);

          /*
           *  // we need to process the remainder only in case
           *  // 1. The n_elem is multiple of SIMDsize and curvature is padded
           *  // then we cannot load director at the index n_element required by
           *  // the curvature calculation at index n_element
           *  // or
           *  // 2. The elements of MT, TT or VT are not padded
           *
          * const bool first_condition((blocksize + 1) % SIMDSIZE);
          * constexpr bool second_condition(
          *     (!::blaze::usePadding) ||
          *     // use demorgan
          *     !cpp17::conjunction_v<blaze::IsPadded<MT>, blaze::IsPadded<TT>,
          *                           blaze::IsPadded<VT>>);
          * const bool remainder(first_condition || second_condition);

          * const std::size_t index_till_simd(
          *     remainder ? blaze::prevMultiple(blocksize, SIMDSIZE)
          *               : blaze::nextMultiple(blocksize, SIMDSIZE));
          */
          // Seems like you always need to do the last vector without SIMDing
          const std::size_t index_till_simd(
              blaze::prevMultiple(blocksize, SIMDSIZE));

          typename TT::ConstIterator q_it[9UL] = {
              rot_matrix.begin(0UL, 0UL), rot_matrix.begin(1UL, 0UL),
              rot_matrix.begin(2UL, 0UL), rot_matrix.begin(0UL, 1UL),
              rot_matrix.begin(1UL, 1UL), rot_matrix.begin(2UL, 1UL),
              rot_matrix.begin(0UL, 2UL), rot_matrix.begin(1UL, 2UL),
              rot_matrix.begin(2UL, 2UL)};

          typename MT::Iterator u_it[3UL] = {rot_axis_vector.begin(0UL),
                                             rot_axis_vector.begin(1UL),
                                             rot_axis_vector.begin(2UL)};

          auto span_vector_it = span_vector.cbegin();

          // allocate rolling memory
          /*
           * structure is
           * q0 q3 q6
           * q1 q4 q7
           * q2 q5 q8
           */
          SIMDType Q_curr[9UL];
          SIMDType Q_next[9UL];

          std::size_t i_dof = 0UL;
          for (; i_dof < index_till_simd; i_dof += SIMDSIZE) {
            // begin to loop
            // i_dof is the current index being processed

            // 1. load data into registers for the next iteration
            {
              // 1.1 load q0
              UNROLL_LOOP(9UL)
              for (std::size_t dim = 0UL; dim < 9UL; ++dim)
                Q_curr[dim] = q_it[dim].load();  // TODO : check aligned

                // 1.2 load q1
              UNROLL_LOOP(9UL)
              for (std::size_t dim = 0UL; dim < 9UL; ++dim)
                // need to do an unaligned load here given its off by one
                Q_next[dim] = (++q_it[dim]).loadu();

            }  // load

            /*
             * 2. perform
             * R = Q_i Q^T_{i+1} (elementwise)
             * 0.5 * tr(R)
             */
#define FORCE_EVALUATE 1
#if defined(FORCE_EVALUATE)
               // Force evaluation here, don't make it lazy
            SIMDType
#else
            auto
#endif
                const half_rot_matrix_trace =
                    ::blaze::set(T(0.5)) *
                    (Q_curr[0] * Q_next[0] + Q_curr[1] * Q_next[1] +
                     Q_curr[2] * Q_next[2] + Q_curr[3] * Q_next[3] +
                     Q_curr[4] * Q_next[4] + Q_curr[5] * Q_next[5] +
                     Q_curr[6] * Q_next[6] + Q_curr[7] * Q_next[7] +
                     Q_curr[8] * Q_next[8]);

            /*
             * 3. calculate theta
             * theta = acos((tr(R) - 1) / 2)
             *
             * and prefactor
             * 0.5 * sin(theta) / theta / d_i
             */
            constexpr T acos_constant(T(0.5) + T(1e-10));
#if BLAZE_SVML_MODE || BLAZE_SLEEF_MODE

            SIMDType const theta = ::blaze::acos(half_rot_matrix_trace -
                                                 ::blaze::set(acos_constant));

            SIMDType const st = ::blaze::set(0.5) * theta /
                                (::blaze::sin(theta + ::blaze::set(1e-14)) *
                                 span_vector_it.load());

#else
            // a temporary aligned register to pipe contents into
            alignas(::blaze::AlignmentOf_v<T>) T half_theta[SIMDSIZE];
            alignas(::blaze::AlignmentOf_v<T>) T s_theta[SIMDSIZE];

// no intrinsics, so we unpack the SIMD vector, compute sin and
// cos on each via regular math functions and then repack it
            UNROLL_LOOP(SIMDSIZE)
            for (std::size_t simd_idx = 0UL; simd_idx < SIMDSIZE; ++simd_idx) {
              // make clear the intent to compiler
              const T temp =
                  std::acos(half_rot_matrix_trace[simd_idx] - acos_constant);
              half_theta[simd_idx] = 0.5 * temp;
              s_theta[simd_idx] = std::sin(temp + 1e-14);
            }

            SIMDType const st = blaze::loada(half_theta) /
                                (blaze::loada(s_theta) * span_vector_it.load());
#endif

            /*
             * 4. Store kappa = prefactor * (Q[i] Q[i+1].T)
             */
            {
              u_it[0].store(st *
                            (Q_curr[6] * Q_next[3] + Q_curr[7] * Q_next[4] +
                             Q_curr[8] * Q_next[5] - Q_curr[3] * Q_next[6] -
                             Q_curr[4] * Q_next[7] - Q_curr[5] * Q_next[8]));

              u_it[1].store(st *
                            (Q_curr[0] * Q_next[6] + Q_curr[1] * Q_next[7] +
                             Q_curr[2] * Q_next[8] - Q_curr[6] * Q_next[0] -
                             Q_curr[7] * Q_next[1] - Q_curr[8] * Q_next[2]));

              u_it[2].store(st *
                            (Q_curr[3] * Q_next[0] + Q_curr[4] * Q_next[1] +
                             Q_curr[5] * Q_next[2] - Q_curr[0] * Q_next[3] -
                             Q_curr[1] * Q_next[4] - Q_curr[2] * Q_next[5]));
            }

            /*
             * 5. advance all the iterators to the next SIMD lane
             */
            {
              UNROLL_LOOP(3UL)
              for (std::size_t dim = 0UL; dim < 3UL; ++dim) /* NOLINT */
                u_it[dim] += SIMDSIZE;

              UNROLL_LOOP(9UL)
              for (std::size_t dim = 0UL; dim < 9UL; ++dim) /* NOLINT */
                q_it[dim] +=
                    (SIMDSIZE - 1UL);  // because I already dim ++q_it[dim]

              span_vector_it += SIMDSIZE;
            }  // 5

          }  // i_dof

          //          if (remainder)
          InvRotateDivideOp<InvRotateDivideKind::scalar>::internal_apply(
              rot_axis_vector, rot_matrix, span_vector, i_dof, blocksize);

        }  // apply
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
