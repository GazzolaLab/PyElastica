#pragma once
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecVec/Checks.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecVec/Scalar.hpp"
#include "Utilities/IgnoreUnused.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/Aliases.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/expressions/DenseVector.h>
#include <blaze/math/shims/PrevMultiple.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/system/Optimizations.h>  // for padding
#include <blaze/util/Misalignment.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename MT3,        // blaze Vector type
              bool SO>             // Storage order
    void lazy_vector_vector_kernel_simd_unchecked(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix1,
        blaze::DenseMatrix<MT3, SO> const& input_matrix2) {
      // Controls load to store ratio, should be multiple of cacheline
      constexpr std::size_t n_packs_in_flight = 2UL;
      // Switch between simd and scalar execution
      constexpr std::size_t scalar_simdpack_threshold = 2UL * n_packs_in_flight;

      //
      using SIMDType = blaze::SIMDType_t<MT1>;
      using ValueType = typename MT1::ElementType;

      constexpr bool remainder(!::blaze::usePadding || !blaze::IsPadded_v<MT1>);
      constexpr std::size_t dimension(3UL);
      constexpr std::size_t SIMDSIZE(SIMDType::size);

      auto const& in_matrix1(*input_matrix1);
      auto const& in_matrix2(*input_matrix2);
      auto& out_matrix(*output_matrix);

      // TODO : n_outputs will only be n_columns() in case we use unaligned
      // matrices, else we can assume an extra padding to the next SIMDSIZE
      const std::size_t n_outputs = out_matrix.columns();
      constexpr std::size_t index_start_simd(0UL);
      const std::size_t index_stop_simd(
          // See comment below as to why we dont use remainder
          remainder ? blaze::prevMultiple(n_outputs, SIMDSIZE) : n_outputs);
      Expects(index_stop_simd <= n_outputs);

      if (UNLIKELY(index_stop_simd < scalar_simdpack_threshold * SIMDSIZE)) {
        // Do scalar processing in this case
        lazy_vector_vector_kernel_scalar(std::move(op), out_matrix, in_matrix1,
                                         in_matrix2, 0UL, n_outputs);
        return;
      }

      // Temporary storage
      typename MT2::ConstIterator v_it1[dimension] = {
          in_matrix1.begin(0UL) + index_start_simd,
          in_matrix1.begin(1UL) + index_start_simd,
          in_matrix1.begin(2UL) + index_start_simd,
      };
      typename MT3::ConstIterator v_it2[dimension] = {
          in_matrix2.begin(0UL) + index_start_simd,
          in_matrix2.begin(1UL) + index_start_simd,
          in_matrix2.begin(2UL) + index_start_simd,
      };
      typename MT1::Iterator ov_it[dimension] = {
          out_matrix.begin(0UL) + index_start_simd,
          out_matrix.begin(1UL) + index_start_simd,
          out_matrix.begin(2UL) + index_start_simd,
      };
      // SIMDTypes are already aligned.
      // Store all packs in dim[0] first, then dim[1] and so on...
      SIMDType vector_cache[n_packs_in_flight][2UL][dimension];

      // When a block is sliced, we cannot guarantee alignment requirements in
      // the memory address.
      // So to be safe, assume non-aligned loads/stores except into temporary
      // memory.
      // Should get compiled away.
      /*
      Expects([&]() -> bool {
        bool okay = true;
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          // v_it[] need not be aligned
          // okay &= (!blaze::misalignment(v_it[dim].base()));
          okay &= !blaze::misalignment(&(*ov_it[dim]));
        }
        return okay;
      }());
      */

      auto index(index_start_simd);
      for (; index + (n_packs_in_flight - 1) * SIMDSIZE < index_stop_simd;
           index += (n_packs_in_flight * SIMDSIZE)) {
        // Load up sequential packs first
        UNROLL_LOOP(n_packs_in_flight)
        for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
          // Load all into temporaries
          UNROLL_LOOP(dimension)
          for (auto dim = 0UL; dim < dimension; ++dim) {
            vector_cache[pack_idx][0UL][dim] = v_it1[dim].load();
            vector_cache[pack_idx][1UL][dim] = v_it2[dim].load();
            v_it1[dim] += SIMDSIZE;
            v_it2[dim] += SIMDSIZE;
          }
        }

#define PERFORM_OP(DIMENSION)                                               \
  ov_it[DIMENSION].store(op.template operator()<DIMENSION>(                 \
      vector_cache[pack_idx][0UL][0UL], vector_cache[pack_idx][0UL][1UL],   \
      vector_cache[pack_idx][0UL][2UL], vector_cache[pack_idx][1UL][0UL],   \
      vector_cache[pack_idx][1UL][1UL], vector_cache[pack_idx][1UL][2UL])); \
  ov_it[DIMENSION] += SIMDSIZE

        UNROLL_LOOP(n_packs_in_flight)
        for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
          PERFORM_OP(0UL);
          PERFORM_OP(1UL);
          PERFORM_OP(2UL);
        }  // packs

      }  // SIMD Pack index

      // The index load here can go beyond the number of columns, upto the
      // capacity.
      // We only utilize the first pack here
      for (; index < index_stop_simd; index += SIMDSIZE) {
        constexpr auto pack_idx = 0UL;

        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          vector_cache[pack_idx][0UL][dim] = v_it1[dim].load();
          vector_cache[pack_idx][1UL][dim] = v_it2[dim].load();
          v_it1[dim] += SIMDSIZE;
          v_it2[dim] += SIMDSIZE;
        }

        PERFORM_OP(0UL);
        PERFORM_OP(1UL);
        PERFORM_OP(2UL);

      }  // SIMD index

#undef PERFORM_OP

      ValueType scalar_cache[2UL][dimension];
#define PERFORM_OP(DIMENSION)                                                  \
  *ov_it[DIMENSION] = op.template operator()<DIMENSION>(                       \
      scalar_cache[0UL][0UL], scalar_cache[0UL][1UL], scalar_cache[0UL][2UL],  \
      scalar_cache[1UL][0UL], scalar_cache[1UL][1UL], scalar_cache[1UL][2UL]); \
  ++ov_it[DIMENSION]

      for (; remainder && index < n_outputs; ++index) {
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          scalar_cache[0UL][dim] = (*v_it1[dim]);
          scalar_cache[1UL][dim] = (*v_it2[dim]);
          ++v_it1[dim];
          ++v_it2[dim];
        }

        PERFORM_OP(0UL);
        PERFORM_OP(1UL);
        PERFORM_OP(2UL);

#undef PERFORM_OP
      }  // scalar index
    }

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename MT3,        // blaze Vector type
              bool SO>             // Storage order
    void lazy_vector_vector_kernel_simd(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_vector,
        blaze::DenseMatrix<MT2, SO> const& in_vector1,
        blaze::DenseMatrix<MT3, SO> const& in_vector2) {
      detail::vector_vector_kernel_checks(*out_vector, *in_vector1,
                                          *in_vector2);
      lazy_vector_vector_kernel_simd_unchecked(std::move(op), *out_vector,
                                               *in_vector1, *in_vector2);
    }

  }  // namespace cosserat_rod

}  // namespace elastica
