#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class VecScalarMultKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    inline std::string to_string(VecScalarMultKind k) {
      return "VecScalarMultKind" + to_string(BackendKind(k));
    }

    template <VecScalarMultKind K>
    struct VecScalarMultOp;

    template <>
    constexpr auto backend_user_choice<VecScalarMultKind>() noexcept
        -> VecScalarMultKind {
      return VecScalarMultKind::ELASTICA_COSSERATROD_LIB_VECSCALARMULT_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
