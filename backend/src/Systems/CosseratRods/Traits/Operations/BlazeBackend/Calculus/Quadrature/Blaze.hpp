#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/Quadrature/BaseTemplate.hpp"
#include "Utilities/Math/BlazeDetail/BlazeCalculus.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct QuadratureOp<QuadratureKind::blaze> {
      template <typename... Args>
      static inline auto apply(Args&&... args) noexcept -> void {
        elastica::quadrature_kernel(std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
