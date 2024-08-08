#pragma once
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecScalar/Checks.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecScalar/Scalar.hpp"
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
              typename VT,         // blaze Vector type
              bool SO,             // Storage order
              bool TF>
    void lazy_vector_scalar_kernel_simd_unchecked(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix,
        blaze::DenseVector<VT, TF> const& input_vector) {
      // Controls load to store ratio, should be multiple of cacheline
      constexpr std::size_t n_packs_in_flight = 2UL;
      // Switch between simd and scalar execution
      constexpr std::size_t scalar_simdpack_threshold = 4UL;

      constexpr bool remainder(!::blaze::usePadding || !blaze::IsPadded_v<MT1>);
      elastica::ignore_unused(remainder);
      //
      constexpr std::size_t dimension(3UL);

      using SIMDType = blaze::SIMDType_t<MT1>;
      constexpr std::size_t SIMDSIZE(SIMDType::size);

      auto const& in_matrix(*input_matrix);
      auto const& in_vector(*input_vector);
      auto& out_matrix(*output_matrix);

      // TODO : n_outputs will only be n_columns() in case we use unaligned
      // matrices, else we can assume an extra padding to the next SIMDSIZE
      const std::size_t n_outputs = out_matrix.columns();
      constexpr std::size_t index_start_simd(0UL);
      const std::size_t index_stop_simd(
          // See comment below as to why we dont use remainder
          // remainder ? blaze::prevMultiple(n_outputs, SIMDSIZE) : n_outputs
          blaze::prevMultiple(n_outputs, SIMDSIZE));
      Expects(index_stop_simd <= n_outputs);

      if (UNLIKELY(index_stop_simd < scalar_simdpack_threshold * SIMDSIZE)) {
        // Do scalar processing in this case
        lazy_vector_scalar_kernel_scalar(std::move(op), out_matrix, in_matrix,
                                         in_vector, 0UL, n_outputs);
        return;
      }

      // Temporary storage
      typename MT2::ConstIterator v_it[dimension] = {
          in_matrix.begin(0UL) + index_start_simd,
          in_matrix.begin(1UL) + index_start_simd,
          in_matrix.begin(2UL) + index_start_simd,
      };
      typename VT::ConstIterator s_it(in_vector.begin() + index_start_simd);
      typename MT1::Iterator ov_it[dimension];

      UNROLL_LOOP(dimension)
      for (auto dim = 0UL; dim < dimension; ++dim)
        ov_it[dim] = out_matrix.begin(dim) + index_start_simd;

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
        // Loop over rows" and process elements
        // The page is the output dim in this regard
        UNROLL_LOOP(n_packs_in_flight)
        for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
          auto cached = s_it.load();

          UNROLL_LOOP(dimension)
          for (auto dim = 0UL; dim < dimension; ++dim) {
            ov_it[dim].store(op(v_it[dim].load(), cached));
            v_it[dim] += SIMDSIZE;
            ov_it[dim] += SIMDSIZE;
          }

          s_it += SIMDSIZE;
        }  // packs

      }  // SIMD Pack index

      // The index load here can go beyond the number of columns, upto the
      // capacity. However, when doing things like division, it may lead to a
      // 0 / 0 error in uninitialized capacity values. To prevent this from
      // happening, we skip processing the last SIMD elements in a lane and
      // instead do them scalar.
      /*
      for (; index < index_stop_simd; index += SIMDSIZE) {
        auto cached = s_it.loada();

        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          // All input expressions  may not have loada()
          ov_it[dim].stream(op(v_it[dim].load(), cached));
          v_it[dim] += SIMDSIZE;
          ov_it[dim] += SIMDSIZE;
        }

        s_it += SIMDSIZE;
      }  // SIMD index
      */

      for (; /*remainder &&*/ index < n_outputs; ++index) {
        auto cached = *s_it;

        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          *(ov_it[dim]) = op(*(v_it[dim]), cached);
          ++v_it[dim];
          ++ov_it[dim];
        }
        ++s_it;
      }  // scalar index
    }

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename VT,         // blaze Vector type
              bool SO,             // Storage order
              bool TF>
    void lazy_vector_scalar_kernel_simd(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_vector,
        blaze::DenseMatrix<MT2, SO> const& in_vector,
        blaze::DenseVector<VT, TF> const& in_scalar) {
      detail::vector_scalar_kernel_checks(*out_vector, *in_vector, *in_scalar);
      lazy_vector_scalar_kernel_simd_unchecked(std::move(op), *out_vector,
                                               *in_vector, *in_scalar);
    }

  }  // namespace cosserat_rod

}  // namespace elastica
