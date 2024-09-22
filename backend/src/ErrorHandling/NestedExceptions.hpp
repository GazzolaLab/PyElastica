#pragma once

#include <exception>

//******************************************************************************
/*!\brief Print backtrace of nested exceptions
 * \ingroup ErrorHandlingGroup
 *
 * \details
 * Prints the explanatory string of an exception. If the exception is
 * nested, ecurses to print the explanatory of the exception it holds.
 */
void abort_with_backtrace(const std::exception& e, std::size_t level = 0UL);
//******************************************************************************
