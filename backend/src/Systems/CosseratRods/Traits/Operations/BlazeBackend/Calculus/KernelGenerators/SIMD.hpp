#pragma once

//
#include "ErrorHandling/ExpectsAndEnsures.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/KernelGenerators/Scalar.hpp"
#include "Utilities/Unroll.hpp"
//
#include <cstddef>  // size_t
#include <utility>  // move
//
#include <blaze/math/Aliases.h>
#include <blaze/math/SIMD.h>
#include <blaze/math/expressions/DenseMatrix.h>
#include <blaze/math/shims/PrevMultiple.h>
#include <blaze/util/typetraits/AlignmentOf.h>

namespace elastica {

  namespace cosserat_rod {

    template <typename Operation,  // Operation type
              typename MT1,        // blaze Matrix expression type 1
              typename MT2,        // blaze Matrix expression type 2
              bool SO>             // Storage order
    void lazy_twopoint_kernel_simd(
        Operation op, blaze::DenseMatrix<MT1, SO>& output_matrix,
        blaze::DenseMatrix<MT2, SO> const& input_matrix) {
      // Kernel configuration parameters

      // Controls load to store ratio, should be multiple of cacheline
      constexpr std::size_t n_packs_in_flight = 2UL;
      // Switch between simd and scalar execution
      constexpr std::size_t scalar_simdpack_threshold = 2UL * n_packs_in_flight;
      // Where to begin processing the first simdpack
      constexpr std::size_t start_simdpack_idx = 1UL;

      //
      constexpr std::size_t dimension(3UL);

      using T = blaze::ElementType_t<MT1>;
      using SIMDType = blaze::SIMDType_t<MT1>;
      constexpr std::size_t SIMDSIZE(SIMDType::size);

      auto const& in_matrix(*input_matrix);
      auto& out_matrix(*output_matrix);

      const std::size_t n_outputs = out_matrix.columns();

      // Refactor out
      Expects(in_matrix.rows() == dimension);
      Expects(out_matrix.rows() == dimension);
      Expects(n_outputs == in_matrix.columns() + 1UL);

      const std::size_t output_simd_idx(
          blaze::prevMultiple(n_outputs, SIMDSIZE));

      // We process n_packs_in_flight * SIMDSIZE elements at a time. We can use
      // this to figure out the number of trip counts
      constexpr std::size_t index_start_simd(start_simdpack_idx * SIMDSIZE);
      constexpr std::size_t n_elements_in_flight(n_packs_in_flight * SIMDSIZE);

      if (UNLIKELY((output_simd_idx) < scalar_simdpack_threshold * SIMDSIZE) || UNLIKELY((output_simd_idx - index_start_simd) < scalar_simdpack_threshold * SIMDSIZE)) {
        // Do scalar processing in this case
        lazy_twopoint_kernel_scalar(std::move(op), out_matrix, in_matrix);
        return;
      }

      Expects((output_simd_idx - index_start_simd) % SIMDSIZE == 0);
      // n_packs that can fit in
      const std::size_t n_trips_hint =
          (output_simd_idx - index_start_simd) / n_elements_in_flight;
      const std::size_t index_stop_simd_hint(
          index_start_simd + n_trips_hint * n_elements_in_flight);

      // In case n_outputs is a perfect multiple of SIMDSIZE, we only do process
      // till the last SIMDSIZE elements, and evaluate the last pack as a scalar
      // to incorporate boundary conditions
      Expects(n_trips_hint > 1UL);

      const std::size_t n_trips =
          n_trips_hint - std::size_t(n_outputs == index_stop_simd_hint);
      const std::size_t index_stop_simd =
          index_stop_simd_hint -
          ((n_outputs == index_stop_simd_hint) ? n_elements_in_flight : 0UL);

      Expects(index_start_simd + n_trips * n_elements_in_flight ==
              index_stop_simd);

      // Edge cases are done via scalar processing
      // Left edge
      lazy_twopoint_kernel_scalar(op, out_matrix, in_matrix, 0UL,
                                  index_start_simd);
      // Right edge
      // -1UL here because for cases with index_stop == n_outputs - 1, we will have
      // an out of bound read from inputs
      lazy_twopoint_kernel_scalar(op, out_matrix, in_matrix, index_stop_simd - 1UL,
                                  n_outputs);

      // Need to align the element [1] to alignment and not zero.
      constexpr auto alignment(::blaze::AlignmentOf_v<T>);
      // Number of elements that can fit in one allocation
      constexpr auto padding(alignment / sizeof(T));
      // Instead of allocation +1 element, we allocate +padding number of
      // elements
      constexpr std::size_t input_cache_width(n_elements_in_flight + padding);
      alignas(alignment) T _input_cache[input_cache_width];
      // This is aligned at the first element.
      T* input_cache = _input_cache + (padding - 1UL);

      for (auto dim = 0UL; dim < dimension; ++dim) {
        // edge case first

        auto out_it = out_matrix.begin(dim) + index_start_simd;
        // Some expr do not have a cbegin, so use begin instead.
        // in_it is unaligned, so we can only use unaligned loads/stores
        auto in_it = in_matrix.begin(dim) + index_start_simd;

        // middle case
        // Can also compare iterators directly
        for (auto i_trip = 0UL; i_trip < n_trips; ++i_trip) {
          // Fill the first value since it is not aligned
          input_cache[0UL] = *(in_it - 1);

          UNROLL_LOOP(n_packs_in_flight)
          for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
            // Store the loaded packs
            // The load may have arbitrary computations, so dont interleave
            // load and op()
            blaze::storea(
                input_cache + pack_idx * SIMDSIZE + 1UL,
                in_it.load());  // All expressions may not have aligned loads
            in_it += SIMDSIZE;
          }

          UNROLL_LOOP(n_packs_in_flight)
          for (auto pack_idx = 0UL; pack_idx < n_packs_in_flight; ++pack_idx) {
            const std::size_t curr_idx = pack_idx * SIMDSIZE;
            out_it.store(op(blaze::loada(input_cache + curr_idx + 1UL),
                            blaze::loadu(input_cache + curr_idx)));
            out_it += SIMDSIZE;
          }
        }  // middle case
      }
    }
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica
