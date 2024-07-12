// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ErrorHandling/NestedExceptions.hpp"

#include <iostream>

#include "ErrorHandling/Abort.hpp"

void abort_with_backtrace(const std::exception& e, std::size_t level) {
  std::cerr << std::string(level, ' ') << "exception: " << e.what() << '\n';
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& ne) {
    abort_with_backtrace(ne, level + 1);
  } catch (...) {
  }

  // issues with Catch here, instead of aborting, simply throw an
  throw std::runtime_error("Cannot proceed with program, aborting...");
  // Parallel::abort("Cannot proceed with program, aborting...");
}
