#pragma once

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/KernelGenerators/MatVec/SIMD.hpp"
//
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/MatVec/BaseTemplate.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/LinearAlgebra/MatVec/Operation.hpp"
//
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
#include "Utilities/ForceInline.hpp"
//
#include <utility>  // forward

namespace elastica {

  namespace cosserat_rod {

    template <>
    struct MatVecOp<MatVecKind::simd> {
      template <typename... Args>  // blaze Matrix expression type
      static ELASTICA_ALWAYS_INLINE auto apply(Args&&... args)
          COSSERATROD_LIB_NOEXCEPT->void {
        lazy_matrix_vector_kernel_simd(MatrixVectorProductOperation{},
                                       std::forward<Args>(args)...);
      };
    };

  }  // namespace cosserat_rod

}  // namespace elastica
