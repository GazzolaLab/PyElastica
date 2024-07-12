// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ErrorHandling/Breakpoint.hpp"

#include <csignal>

void breakpoint() {
  // We send ourselves a SIGTRAP and ignore it.  If we're not being
  // traced (e.g. being run in a debugger), that doesn't do much, but if we are
  // then the tracer (debugger) can see the signal delivery.  We don't reset the
  // signal handler afterwards in case this is called on multiple threads; we
  // don't want one thread reenabling the default handler just before another
  // calls raise().
  struct sigaction handler {};
  handler.sa_handler = SIG_IGN;  // NOLINT
  handler.sa_flags = 0;
  sigemptyset(&handler.sa_mask);
  sigaction(SIGTRAP, &handler, nullptr);
  raise(SIGTRAP);
}
