#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class VecScalarDivKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    inline std::string to_string(VecScalarDivKind k) {
      return "VecScalarDivKind" + to_string(BackendKind(k));
    }

    template <VecScalarDivKind K>
    struct VecScalarDivOp;

    template <>
    constexpr auto backend_user_choice<VecScalarDivKind>() noexcept
        -> VecScalarDivKind {
      return VecScalarDivKind::ELASTICA_COSSERATROD_LIB_VECSCALARDIV_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
