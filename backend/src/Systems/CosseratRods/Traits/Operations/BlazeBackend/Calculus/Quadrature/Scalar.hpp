#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/KernelGenerators/Scalar.hpp"
//
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/Quadrature/BaseTemplate.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/Calculus/Quadrature/Operation.hpp"
//
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
#include "Utilities/ForceInline.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct QuadratureOp<QuadratureKind::scalar> {
      template <typename... Args>  // blaze Matrix expression type
      static ELASTICA_ALWAYS_INLINE auto apply(Args&&... args)
          COSSERATROD_LIB_NOEXCEPT->void {
        lazy_twopoint_kernel_scalar(QuadratureOperation{},
                                    std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
