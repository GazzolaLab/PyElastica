#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    enum class MatVecKind : std::uint8_t {
      scalar = std::uint8_t(BackendKind::scalar),
      simd = std::uint8_t(BackendKind::simd),
      blaze = std::uint8_t(BackendKind::blaze)
    };

    inline std::string to_string(MatVecKind k) {
      return "MatVecKind" + to_string(BackendKind(k));
    }

    template <MatVecKind K>
    struct MatVecOp;

    template <>
    constexpr auto backend_user_choice<MatVecKind>() noexcept -> MatVecKind {
      return MatVecKind::ELASTICA_COSSERATROD_LIB_MATVEC_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
