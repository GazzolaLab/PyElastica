#pragma once

#include "Length.hpp"
#include "Utilities/ForceInline.hpp"
#include "Vec3.hpp"

namespace elastica {

  // computes distance between p1 and p2
  ELASTICA_ALWAYS_INLINE real_t dist(const Vec3& a, const Vec3& b) {
    return length(b - a);
  }

}  // namespace elastica
