#pragma once

#include "Utilities/ForceInline.hpp"

namespace elastica {

  namespace cosserat_rod {

    struct VectorScalarMultOperation {
      explicit inline VectorScalarMultOperation() = default;

      template <typename V, typename S>
      ELASTICA_ALWAYS_INLINE decltype(auto) operator()(
          V const& v, S const& s) const noexcept {
        return v * s;
      }
    };

  }  // namespace cosserat_rod

}  // namespace elastica
