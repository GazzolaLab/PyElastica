// Reused from SpECTRE : https://spectre-code.org/
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// Suppress warnings from this header since they can occur from not including
// StdHelpers later. This header is used only in TypeTraits.hpp so that
// is_streamable works correctly for STL types that have stream operators
// defined.
#ifdef __GNUC__
#pragma GCC system_header
#endif

#include <array>
#include <deque>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Requires.hpp"

namespace tt {
  template <typename S, typename T, typename>
  struct is_streamable;
}  // namespace tt

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::list<T>& v) noexcept;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) noexcept;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::deque<T>& v) noexcept;

template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& a) noexcept;

template <typename... Args>
std::ostream& operator<<(std::ostream& os,
                         const std::tuple<Args...>& t) noexcept;

template <typename K, typename V, typename H>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_map<K, V, H>& m) noexcept;

template <typename K, typename V, typename C>
inline std::ostream& operator<<(std::ostream& os,
                                const std::map<K, V, C>& m) noexcept;

template <typename T, typename H>
std::ostream& operator<<(std::ostream& os,
                         const std::unordered_set<T, H>& v) noexcept;

template <typename T, typename C>
inline std::ostream& operator<<(std::ostream& os,
                                const std::set<T, C>& v) noexcept;

template <typename T,
          Requires<tt::is_streamable<std::ostream, T, void>::value> = nullptr>
std::ostream& operator<<(std::ostream& os,
                         const std::unique_ptr<T>& t) noexcept;

template <typename T,
          Requires<tt::is_streamable<std::ostream, T, void>::value> = nullptr>
std::ostream& operator<<(std::ostream& os,
                         const std::shared_ptr<T>& t) noexcept;

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& t) noexcept;
