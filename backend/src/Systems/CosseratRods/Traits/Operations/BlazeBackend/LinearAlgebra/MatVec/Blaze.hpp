#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/MatVec/BaseTemplate.hpp"
#include "Utilities/Math/BlazeDetail/BlazeLinearAlgebra.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct MatVecOp<MatVecKind::blaze> {
      template <typename... Args>  // blaze Matrix expression type
      static inline auto apply(Args&&... args) noexcept -> void {
        elastica::batch_matvec(std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
