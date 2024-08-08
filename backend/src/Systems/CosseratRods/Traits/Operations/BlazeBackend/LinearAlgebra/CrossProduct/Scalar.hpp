#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/VecVec/Scalar.hpp"
//
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/CrossProduct/BaseTemplate.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/CrossProduct/Operation.hpp"
//
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
#include "Utilities/ForceInline.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct CrossProductOp<CrossProductKind::scalar> {
      template <typename... Args>  // blaze Matrix expression type
      static ELASTICA_ALWAYS_INLINE auto apply(Args&&... args)
          COSSERATROD_LIB_NOEXCEPT->void {
        lazy_vector_vector_kernel_scalar(VectorCrossProductOperation{},
                                         std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
