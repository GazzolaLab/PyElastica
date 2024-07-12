
#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class DifferenceKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    template <DifferenceKind K>
    struct DifferenceOp;

    inline std::string to_string(DifferenceKind k) {
      return "DifferenceKind" + to_string(BackendKind(k));
    }

    template <>
    constexpr auto backend_user_choice<DifferenceKind>() noexcept
        -> DifferenceKind {
      return DifferenceKind::ELASTICA_COSSERATROD_LIB_DIFFERENCE_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
