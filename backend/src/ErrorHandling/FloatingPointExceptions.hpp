// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Functions to enable/disable termination on floating point exceptions

#pragma once

/// \ingroup ErrorHandlingGroup
/// After a call to this function, the code will terminate with a floating
/// point exception on overflow, divide-by-zero, and invalid operations.
void enable_floating_point_exceptions();

/// \ingroup ErrorHandlingGroup
/// After a call to this function, the code will NOT terminate with a floating
/// point exception on overflow, divide-by-zero, and invalid operations.
void disable_floating_point_exceptions();
