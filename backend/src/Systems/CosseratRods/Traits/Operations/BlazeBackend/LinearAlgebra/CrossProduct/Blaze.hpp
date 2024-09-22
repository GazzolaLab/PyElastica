#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/CrossProduct/BaseTemplate.hpp"
#include "Utilities/Math/BlazeDetail/BlazeLinearAlgebra.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct CrossProductOp<CrossProductKind::blaze> {
      template <typename... Args>  // blaze Matrix expression type
      static inline auto apply(Args&&... args) noexcept -> void {
        using ::elastica::batch_cross;
        elastica::batch_cross(std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
