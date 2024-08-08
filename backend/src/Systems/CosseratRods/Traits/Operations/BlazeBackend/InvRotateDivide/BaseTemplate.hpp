#pragma once

#include "Configuration/Kernels.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/BackendKind.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      enum class InvRotateDivideKind : std::uint8_t {
        scalar = std::uint8_t(BackendKind::scalar),
        simd = std::uint8_t(BackendKind::simd),
        blaze = std::uint8_t(BackendKind::blaze)
      };

      template <InvRotateDivideKind OpKind>
      struct InvRotateDivideOp;

    }  // namespace detail

    template <>
    constexpr auto backend_user_choice<detail::InvRotateDivideKind>() noexcept
        -> detail::InvRotateDivideKind {
      return detail::InvRotateDivideKind::
          ELASTICA_COSSERATROD_LIB_INV_ROTATE_DIVIDE_BACKEND;
    }

  }  // namespace cosserat_rod

}  // namespace elastica
