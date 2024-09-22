/// \file
/// Defines function ErrorHandling::abort.

#pragma once

#include <exception>

namespace Parallel {
  /// Abort the program with an error message.
  [[noreturn]] inline void exit() {
    std::exit(EXIT_SUCCESS);
    // the following call is never reached, but suppresses the warning that
    // a 'noreturn' functions does return
    std::terminate();  // LCOV_EXCL_LINE
  }
}  // namespace Parallel
