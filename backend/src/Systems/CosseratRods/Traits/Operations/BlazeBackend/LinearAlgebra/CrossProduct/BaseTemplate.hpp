#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class CrossProductKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    inline std::string to_string(CrossProductKind k) {
      return "CrossProductKind" + to_string(BackendKind(k));
    }

    template <CrossProductKind K>
    struct CrossProductOp;

    template <>
    constexpr auto backend_user_choice<CrossProductKind>() noexcept
        -> CrossProductKind {
      return CrossProductKind::ELASTICA_COSSERATROD_LIB_CROSSPRODUCT_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
