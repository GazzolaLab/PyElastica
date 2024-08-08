/// \file
/// Defines function ErrorHandling::breakpoint.

#pragma once

/*!
 * \brief Raise `SIGTRAP` so that debuggers will trap.
 */
void breakpoint();
