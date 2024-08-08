#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class QuadratureKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    inline std::string to_string(QuadratureKind k) {
      return "QuadratureKind" + to_string(BackendKind(k));
    }

    template <QuadratureKind K>
    struct QuadratureOp;

    template <>
    constexpr auto backend_user_choice<QuadratureKind>() noexcept
        -> QuadratureKind {
      return QuadratureKind::ELASTICA_COSSERATROD_LIB_QUADRATURE_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
