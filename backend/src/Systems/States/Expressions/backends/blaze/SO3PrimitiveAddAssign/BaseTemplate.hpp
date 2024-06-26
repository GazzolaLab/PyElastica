#pragma once

#include "Configuration/Kernels.hpp"
#include "ModuleSettings/Vectorization.hpp"

namespace elastica {

  namespace states {

    namespace detail {

      enum class SO3AddAssignKind { scalar, simd };

      template <SO3AddAssignKind Op>
      struct SO3AddAssign;

      template <typename BackendEnumKind>
      constexpr auto backend_choice() -> BackendEnumKind;

      template <>
      constexpr auto backend_choice<SO3AddAssignKind>() -> SO3AddAssignKind {
        return ELASTICA_USE_VECTORIZATION
                   ? SO3AddAssignKind::
                         ELASTICA_COSSERATROD_LIB_SO3_ADDITION_BACKEND
                   : SO3AddAssignKind::scalar;
      }

    }  // namespace detail

  }  // namespace states

}  // namespace elastica
