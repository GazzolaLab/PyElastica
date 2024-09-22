#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Utilities/PrintHelpers.hpp"
#include "Utilities/StlStreamDeclarations.hpp"
#include "Utilities/TypeTraits/IsStreamable.hpp"

namespace StdHelpers_detail {
  // Helper classes for operator<< for tuples
  template <size_t N>
  struct TuplePrinter {
    template <typename... Args>
    static std::ostream& print(std::ostream& os, const std::tuple<Args...>& t) {
      TuplePrinter<N - 1>::print(os, t);
      os << "," << std::get<N - 1>(t);
      return os;
    }
  };

  template <>
  struct TuplePrinter<1> {
    template <typename... Args>
    static std::ostream& print(std::ostream& os, const std::tuple<Args...>& t) {
      os << std::get<0>(t);
      return os;
    }
  };

  template <>
  struct TuplePrinter<0> {
    template <typename... Args>
    static std::ostream& print(std::ostream& os,
                               const std::tuple<Args...>& /*t*/) {
      return os;
    }
  };
}  // namespace StdHelpers_detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::list
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const std::list<T>& v) noexcept {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::vector
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<T>& v) noexcept {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::deque
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const std::deque<T>& v) noexcept {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::array
 */
template <typename T, size_t N>
inline std::ostream& operator<<(std::ostream& os,
                                const std::array<T, N>& a) noexcept {
  sequence_print_helper(os, std::begin(a), std::end(a));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for tuples
 */
template <typename... Args>
inline std::ostream& operator<<(std::ostream& os,
                                const std::tuple<Args...>& t) noexcept {
  os << "(";
  StdHelpers_detail::TuplePrinter<sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output all the key, value pairs of a std::unordered_map
 */
template <typename K, typename V, typename H>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_map<K, V, H>& m) noexcept {
  unordered_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out,
         typename std::unordered_map<K, V, H>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output all the key, value pairs of a std::map
 */
template <typename K, typename V, typename C>
inline std::ostream& operator<<(std::ostream& os,
                                const std::map<K, V, C>& m) noexcept {
  sequence_print_helper(
      os, begin(m), end(m),
      [](std::ostream& out, typename std::map<K, V, C>::const_iterator it) {
        out << "[" << it->first << "," << it->second << "]";
      });
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::unordered_set
 */
template <typename T, typename H>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_set<T, H>& v) noexcept {
  unordered_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Output the items of a std::set
 */
template <typename T, typename C>
inline std::ostream& operator<<(std::ostream& os,
                                const std::set<T, C>& v) noexcept {
  sequence_print_helper(os, std::begin(v), std::end(v));
  return os;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::unique_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unique_ptr<T>& t) noexcept {
  return os << *t;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::shared_ptr
 */
template <typename T, Requires<tt::is_streamable<std::ostream, T>::value>>
inline std::ostream& operator<<(std::ostream& os,
                                const std::shared_ptr<T>& t) noexcept {
  return os << *t;
}

/*!
 * \ingroup UtilitiesGroup
 * \brief Stream operator for std::pair
 */
template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os,
                                const std::pair<T, U>& t) noexcept {
  return os << "(" << t.first << ", " << t.second << ")";
}

/*
template <tuples::TagType TT, class MetaData, typename... Tags>
inline std::ostream& operator<<(
    std::ostream& os, tuples::TaggedTuple<TT, MetaData, Tags...>& tt) noexcept {
  os << "(\n";
  using expander = int[];
  expander{0, ((void)(os << Tags::desc() << '\n'
                         << std::get<Tags>(tt) << '\n'),
               0)...};
  return os << ")\n";
}
*/

// template <typename... Args>
// inline std::ostream& operator<<(std::ostream& os,
//                                TaggedTuple<Args...>& tt) noexcept {
//  os << "(\n";
//  using expander = int[];
//  expander { 0, ( (void) (os << Args::desc() << '\n' << std::get<Args>(tt) <<
//  '\n'), 0) ... };
////  expander {0, ((void) (os << '2'), 0)};
//  os << ")\n";
//  return os;
//}
