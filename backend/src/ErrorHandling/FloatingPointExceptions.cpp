// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ErrorHandling/FloatingPointExceptions.hpp"

#include "ErrorHandling/Abort.hpp"

#include <csignal>

#ifdef __APPLE__
#ifdef __arm64__
#include <fenv.h>
#else
#include <xmmintrin.h>
#endif
#else
#include <cfenv>
#endif

namespace {

#ifdef __APPLE__
#ifndef __arm64__
  auto old_mask = _mm_getcsr();
#endif
#endif

  [[noreturn]] void fpe_signal_handler(int /*signal*/) {
    Parallel::abort("Floating point exception!");  // LCOV_EXCL_LINE
  }
}  // namespace

void enable_floating_point_exceptions() {
#ifdef __APPLE__
#ifndef __arm64__
  _mm_setcsr(_MM_MASK_MASK &
             ~(_MM_MASK_OVERFLOW | _MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
#else
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
  std::signal(SIGFPE, fpe_signal_handler);
}

void disable_floating_point_exceptions() {
#ifdef __APPLE__
#ifndef __arm64__
  _mm_setcsr(old_mask);
#endif
#else
  fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
#endif
}
