#pragma once

#include "ModuleSettings/Vectorization.hpp"
//
#include <string>

namespace elastica {

  namespace cosserat_rod {

    enum class BackendKind : std::uint8_t { scalar, simd, blaze };

    inline std::string to_string(BackendKind k) {
      switch (k) {
        case BackendKind::scalar:
          return "Scalar";
        case BackendKind::simd:
          return "SIMD";
        case BackendKind::blaze:
          return "Blaze";
      }
      // https://abseil.io/tips/147
      throw std::runtime_error("Unexpected enumeration value");
      return "";
    }

    template <typename EnumKind>
    constexpr auto backend_user_choice() noexcept -> EnumKind;

    template <>
    constexpr auto backend_user_choice<BackendKind>() noexcept -> BackendKind {
      return BackendKind::scalar;
    }

    template <typename BackendEnumKind>
    constexpr auto backend_choice() -> BackendEnumKind {
      using ::elastica::cosserat_rod::backend_user_choice;
      // This is a simplification of our backend choosing logic, but for now
      // it suffices.
      return ELASTICA_USE_VECTORIZATION ? backend_user_choice<BackendEnumKind>()
                                        : BackendEnumKind::scalar;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
