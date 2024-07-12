#pragma once

#include <cstddef>

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/BaseTemplate.hpp"
#include "blaze/Blaze.h"
#include "blaze_tensor/Blaze.h"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {
      template <>
      struct InvRotateDivideOp<InvRotateDivideKind::blaze> {
        template <typename MT,  // blaze Matrix expression type
                  typename TT,  // blaze Tensor expression type
                  typename VT>  // blaze Matrix expression type
        static inline auto apply(MT& rot_axis_vector, const TT& rot_matrix,
                                 const VT& span_vector) noexcept -> void {
          using ValueType = typename TT::ElementType;
          // TODO : replace with Frames Dimension
          constexpr std::size_t dimension(3UL);
          const std::size_t blocksize = rot_axis_vector.columns();

          /* hardcoded to avoid external memory passing */
          /* also now operates only on views */
          /*
           * R = Q_t Q_{t+1} (elementwise)
           * theta = acos((tr(R) - 1) / 2)
           */
          auto q_next = blaze::subtensor(rot_matrix, 0UL, 0UL, 1UL, dimension,
                                         dimension, blocksize);
          auto q = blaze::subtensor(rot_matrix, 0UL, 0UL, 0UL, dimension,
                                    dimension, blocksize);
          auto qqt = q % q_next;
          auto rot_matrix_trace =  // TODO: maybe use reduced_sum
              blaze::row(blaze::pageslice(qqt, 0UL), 0UL) +
              blaze::row(blaze::pageslice(qqt, 0UL), 1UL) +
              blaze::row(blaze::pageslice(qqt, 0UL), 2UL) +
              blaze::row(blaze::pageslice(qqt, 1UL), 0UL) +
              blaze::row(blaze::pageslice(qqt, 1UL), 1UL) +
              blaze::row(blaze::pageslice(qqt, 1UL), 2UL) +
              blaze::row(blaze::pageslice(qqt, 2UL), 0UL) +
              blaze::row(blaze::pageslice(qqt, 2UL), 1UL) +
              blaze::row(blaze::pageslice(qqt, 2UL), 2UL);
          auto theta = blaze::acos(ValueType(0.5) * rot_matrix_trace -
                                   ValueType(0.5) - ValueType(1e-10));

          auto trans_span_vector = blaze::trans(span_vector);
          /* theta (u) = -theta * inv_skew([R - RT]) / 2 sin(theta) */
          for (std::size_t i(0UL); i < dimension; ++i) {
            std::size_t idx1 = (i + 1UL) % dimension;
            std::size_t idx2 = (i + 2UL) % dimension;
            blaze::row(rot_axis_vector, i) =
                (ValueType(-0.5) * theta /
                 blaze::sin(theta + ValueType(1e-14)) / trans_span_vector) *
                blaze::sum<blaze::columnwise>((blaze::pageslice(q_next, idx2) %
                                               blaze::pageslice(q, idx1)) -
                                              (blaze::pageslice(q_next, idx1) %
                                               blaze::pageslice(q, idx2)));
          }
        }
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
