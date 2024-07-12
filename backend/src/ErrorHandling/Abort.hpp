/// \file
/// Defines function ErrorHandling::abort.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace Parallel {
  /// Abort the program with an error message.
  [[noreturn]] inline void abort(const std::string& message) {
    printf("%s", message.c_str());
    std::exit(EXIT_SUCCESS);

    // the following call is never reached, but suppresses the warning that
    // a 'noreturn' functions does return
    std::terminate();  // LCOV_EXCL_LINE
  }
}  // namespace Parallel
