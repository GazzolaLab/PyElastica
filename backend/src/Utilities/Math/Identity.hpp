#pragma once

#include <blaze/math/Matrix.h>

#include "Utilities/Math/Rot3.hpp"

namespace elastica {

  template <typename... Args>
  inline auto is_identity(Args&&... args) -> bool {
    return blaze::isIdentity(std::forward<Args>(args)...);
  }

  inline auto identity3() noexcept -> Rot3 {
    return Rot3{{static_cast<real_t>(1.0), static_cast<real_t>(0.0),
                 static_cast<real_t>(0.0)},
                {static_cast<real_t>(0.0), static_cast<real_t>(1.0),
                 static_cast<real_t>(0.0)},
                {static_cast<real_t>(0.0), static_cast<real_t>(0.0),
                 static_cast<real_t>(1.0)}};
  }

}  // namespace elastica
