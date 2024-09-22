// Distributed under the MIT License.
// See LICENSE.txt for details.

// Source :
// https://raw.githubusercontent.com/sxs-collaboration/spectre/develop/src/Utilities/Requires.hpp

#pragma once

/// \file
/// Defines the type alias Requires

#include <cstddef>

namespace Requires_detail {
  template <bool B>
  struct requires_impl {
    using template_error_type_failed_to_meet_requirements_on_template_parameters =
        std::nullptr_t;
  };

  template <>
  struct requires_impl<false> {};
}  // namespace Requires_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Express requirements on the template parameters of a function or
 * class, replaces `std::enable_if_t`
 *
 * Replacement for `std::enable_if_t` and Concepts for expressing requirements
 * on template parameters. This does not require merging of the Concepts
 * TS (whose merit is debatable) and provides an "error message" if substitution
 * of a template parameter failed. Specifically, the compiler error will contain
 * "template_error_type_failed_to_meet_requirements_on_template_parameters",
 * aiding the user of a function or class in tracking down the list of
 * requirements on the deduced type.
 *
 * For example, if a function `foo` is defined as:
 * \snippet Utilities/Test_Requires.cpp foo_definition
 * then calling the function with a list, `foo(std::list<double>{});` results in
 * the following compilation error from clang:
 * \code
 * ./tests/Unit/Utilities/Test_Requires.cpp:29:3: error: no matching function
 *    for call to 'foo'
 *   foo(std::list<double>{});
 *   ^~~
 * ./tests/Unit/Utilities/Test_Requires.cpp:15:13: note: candidate
 *     template ignored: substitution failure [with T = std::__1::list<double,
 *     std::__1::allocator<double> >]: no type named
 *     'template_error_type_failed_to_meet_requirements_on_template_parameters'
 *     in 'Requires_detail::requires_impl<false>'
 * std::string foo(const T&) {
 *             ^
 * 1 error generated.
 * \endcode
 *
 * Here is an example of how write function overloads using `Requires` or to
 * express constraints on the template parameters:
 * \snippet Utilities/Test_Requires.cpp function_definitions
 *
 * \note
 * Using `Requires` is safer than using `std::enable_if_t` because the
 * nested type alias is of type `std::nullptr_t` and so usage is always:
 * \code
 * template <typename T, Requires<(bool depending on T)> = nullptr>
 * \endcode
 */
template <bool B>
using Requires = typename Requires_detail::requires_impl<
    B>::template_error_type_failed_to_meet_requirements_on_template_parameters;
