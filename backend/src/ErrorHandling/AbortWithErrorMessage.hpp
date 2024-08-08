// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function abort_with_error_message

#pragma once

#include <string>

/// \ingroup ErrorHandlingGroup
/// Compose an error message with an expression and abort the program.
[[noreturn]] void abort_with_error_message(const char* expression,
                                           const char* file, int line,
                                           const char* pretty_function,
                                           const std::string& message);

/// \ingroup ErrorHandlingGroup
/// Compose an error message and abort the program.
[[noreturn]] void abort_with_error_message(const char* file, int line,
                                           const char* pretty_function,
                                           const std::string& message);
