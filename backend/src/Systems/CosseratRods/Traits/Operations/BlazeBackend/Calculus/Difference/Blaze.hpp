#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/Difference/BaseTemplate.hpp"
#include "Utilities/Math/BlazeDetail/BlazeCalculus.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct DifferenceOp<DifferenceKind::blaze> {
      template <typename... Args>
      static inline auto apply(Args&&... args) -> void {
        elastica::two_point_difference_kernel(std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
