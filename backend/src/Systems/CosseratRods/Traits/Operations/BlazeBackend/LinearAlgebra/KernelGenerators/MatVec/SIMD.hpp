#pragma once
//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/MatVec/Checks.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/MatVec/Scalar.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/Aliases.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/shims/PrevMultiple.h>
#include <blaze/math/typetraits/IsPadded.h>
#include <blaze/system/Optimizations.h>  // for padding
#include <blaze/util/Misalignment.h>
//
#include <blaze_tensor/math/expressions/DenseTensor.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename TT,         // blaze Tensor type
              bool SO>             // Storage order
    void lazy_matrix_vector_kernel_simd_unchecked(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseTensor<TT> const& input_tensor,
        blaze::DenseMatrix<MT2, SO> const& input_matrix) {
      // Controls load to store ratio, should be multiple of cacheline
      constexpr std::size_t n_packs_in_flight = 2UL;
      // Switch between simd and scalar execution
      constexpr std::size_t scalar_simdpack_threshold = 2UL * n_packs_in_flight;

      //
      constexpr std::size_t dimension(3UL);

      using SIMDType = blaze::SIMDType_t<MT1>;
      constexpr std::size_t SIMDSIZE(SIMDType::size);

      auto const& in_tensor(*input_tensor);
      auto const& in_matrix(*input_matrix);
      auto& out_matrix(*output_matrix);

      // TODO : n_outputs will only be n_columns() in case we use unaligned
      // matrices, else we can assume an extra padding to the next SIMDSIZE
      const std::size_t n_outputs = out_matrix.columns();

      const std::size_t output_simd_idx(
          blaze::prevMultiple(n_outputs, SIMDSIZE));

      if (UNLIKELY(output_simd_idx < scalar_simdpack_threshold * SIMDSIZE)) {
        // Do scalar processing in this case
        lazy_matrix_vector_kernel_scalar(std::move(op), out_matrix, in_tensor,
                                         in_matrix, 0UL, n_outputs);
        return;
      }

      // We process n_packs_in_flight * SIMDSIZE elements at a time. We can use
      // this to figure out the number of trip counts
      constexpr std::size_t index_start_simd(0UL);

      constexpr std::size_t n_elements_in_flight(n_packs_in_flight * SIMDSIZE);

      Expects((output_simd_idx - index_start_simd) % SIMDSIZE == 0);

      // n_packs that can fit in
      const std::size_t n_trips =
          (output_simd_idx - index_start_simd) / n_elements_in_flight;
      const std::size_t index_stop_simd(index_start_simd +
                                        n_trips * n_elements_in_flight);

      // In case n_outputs is a perfect multiple of SIMDSIZE, we only do process
      // till the last SIMDSIZE elements, and evaluate the last pack as a scalar
      // to incorporate boundary conditions
      Expects(n_trips > 1UL);
      Expects(output_simd_idx >= index_stop_simd);

      // Edge cases are done via scalar processing
      // Right edge
      if (index_stop_simd != n_outputs)
        lazy_matrix_vector_kernel_scalar(op, out_matrix, in_tensor, in_matrix,
                                         index_stop_simd, n_outputs);

      // Temporary storage
      typename MT2::ConstIterator v_it[dimension] = {
          in_matrix.begin(0UL) + index_start_simd,
          in_matrix.begin(1UL) + index_start_simd,
          in_matrix.begin(2UL) + index_start_simd,
      };

      // begin has (row, page) as the syntax
      typename TT::ConstIterator q_it[dimension][dimension];

      UNROLL_LOOP(dimension)
      for (auto page_idx = 0UL; page_idx < dimension; ++page_idx) {
        UNROLL_LOOP(dimension)
        for (auto row_idx = 0UL; row_idx < dimension; ++row_idx) {
          q_it[page_idx][row_idx] =
              in_tensor.begin(row_idx, page_idx) + index_start_simd;
        }
      }

      typename MT1::Iterator ov_it[dimension];

      UNROLL_LOOP(dimension)
      for (auto dim = 0UL; dim < dimension; ++dim)
        ov_it[dim] = out_matrix.begin(dim) + index_start_simd;

      // SIMDTypes are already aligned.
      // Store all packs in dim[0] first, then dim[1] and so on...
      SIMDType input_cache[dimension][n_packs_in_flight];

      // Should get compiled away.
      // When a block is sliced, we cannot guarantee alignment requirements in
      // the memory address.
      // So to be safe, assume non-aligned loads/stores except into temporary
      // memory.
      /*
      Expects([&]() -> bool {
        bool okay = true;
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          // v_it[] need not be aligned
          // okay &= (!blaze::misalignment(v_it[dim].base()));
          okay &= !blaze::misalignment(&(*ov_it[dim]));
          UNROLL_LOOP(dimension)
          for (auto rowdim = 0UL; rowdim < dimension; ++rowdim) {
            okay &= !blaze::misalignment(&(*(q_it[dim][rowdim])));
          }
        }
        return okay;
      }());
       */

      for (auto i_trip = 0UL; i_trip < n_trips; ++i_trip) {
        // Load vector into cache
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          UNROLL_LOOP(n_packs_in_flight)
          for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
            input_cache[dim][pack_idx] = v_it[dim].load();
            v_it[dim] += SIMDSIZE;
          }  // packs
        }    // dim

        // Loop over "pages" and process elements
        // The page is the output dim in this regard
        UNROLL_LOOP(dimension)
        for (auto dim = 0UL; dim < dimension; ++dim) {
          UNROLL_LOOP(n_packs_in_flight)
          for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
            // stream() requires aligned addresses, which we cannot gurantee
            // for slices here.
            ov_it[dim].store(
                op(q_it[dim][0UL].load(), q_it[dim][1UL].load(),
                   q_it[dim][2UL].load(), input_cache[0UL][pack_idx],
                   input_cache[1UL][pack_idx], input_cache[2UL][pack_idx]));
            q_it[dim][0UL] += SIMDSIZE;
            q_it[dim][1UL] += SIMDSIZE;
            q_it[dim][2UL] += SIMDSIZE;
            ov_it[dim] += SIMDSIZE;
          }  // packs
        }    // output_dim

      }  // trip
    }

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              typename TT,         // blaze Tensor type
              bool SO>             // Storage order
    void lazy_matrix_vector_kernel_simd(
        Operation op, blaze::DenseMatrix<MT1, SO>& out_matrix,
        blaze::DenseTensor<TT> const& in_tensor,
        blaze::DenseMatrix<MT2, SO> const& in_matrix) {
      detail::matrix_vector_kernel_checks(*out_matrix, *in_tensor, *in_matrix);
      lazy_matrix_vector_kernel_simd_unchecked(std::move(op), *out_matrix,
                                               *in_tensor, *in_matrix);
    }

  }  // namespace cosserat_rod

}  // namespace elastica
