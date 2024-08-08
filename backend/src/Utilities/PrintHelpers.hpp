// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Utilities/StlStreamDeclarations.hpp"

template <typename ForwardIt, typename Func>
void sequence_print_helper(std::ostream& out, ForwardIt&& begin,
                           ForwardIt&& end, Func f) noexcept {
  out << "(";
  if (begin != end) {
    while (true) {
      f(out, begin++);
      if (begin == end) {
        break;
      }
      out << ",";
    }
  }
  out << ")";
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Prints all the items as a comma separated list surrounded by parens.
 */
template <typename ForwardIt>
void sequence_print_helper(std::ostream& out, ForwardIt&& begin,
                           ForwardIt&& end) noexcept {
  sequence_print_helper(out, std::forward<ForwardIt>(begin),
                        std::forward<ForwardIt>(end),
                        [](std::ostream& os, const ForwardIt& it) noexcept {
                          using ::operator<<;
                          os << *it;
                        });
}

//@{
/*!
 * \ingroup UtilitiesGroup
 * Like sequence_print_helper, but sorts the string representations.
 */
template <typename ForwardIt, typename Func>
void unordered_print_helper(std::ostream& out, ForwardIt&& begin,
                            ForwardIt&& end, Func f) noexcept {
  std::vector<std::string> entries;
  while (begin != end) {
    std::ostringstream ss;
    f(ss, begin++);
    entries.push_back(ss.str());
  }
  std::sort(entries.begin(), entries.end());
  sequence_print_helper(out, entries.begin(), entries.end());
}

template <typename ForwardIt>
void unordered_print_helper(std::ostream& out, ForwardIt&& begin,
                            ForwardIt&& end) noexcept {
  unordered_print_helper(out, std::forward<ForwardIt>(begin),
                         std::forward<ForwardIt>(end),
                         [](std::ostream& os, const ForwardIt& it) noexcept {
                           using ::operator<<;
                           os << *it;
                         });
}
