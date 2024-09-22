#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <iterator>  // make_move_iterator
#include <numeric>
#include <vector>

//******************************************************************************
/*!\brief Appends the source vector into the destination
 * \ingroup utils
 * \example
 * \snippet Test_StdVectorHelpers.cpp append_eg
 * \note
 * Returns by reference for use in flatten(). Reference outlives function scope
 * so no UB is invoked.
 */
template <typename T, typename Allocator>
inline void append(std::vector<T, Allocator>& destination,
                   std::vector<T, Allocator> source) {
  if (destination.empty())
    destination = std::move(source);
  else
    destination.insert(std::end(destination),
                       std::make_move_iterator(std::begin(source)),
                       std::make_move_iterator(std::end(source)));
}
//******************************************************************************

//******************************************************************************
/*!\brief Flattens a vector of vectors into a vector.
 * \ingroup utils
 * \example
 * \snippet Test_StdVectorHelpers.cpp flatten_eg
 */
template <typename T, typename InnerAlloc, typename OuterAlloc,
          typename Result = std::vector<T, InnerAlloc>>
auto flatten(std::vector<std::vector<T, InnerAlloc>, OuterAlloc> v) -> Result {
  Result result;
  using Iterated = Result;

  auto const total_size = std::accumulate(
      std::cbegin(v), std::cend(v), typename Result::size_type{0},
      [](auto sz, Iterated const& iterated) { return sz + iterated.size(); });
  result.reserve(total_size);

  // accumulate only moves from C++20 onwards, so pass a ref instead.
  std::accumulate(std::make_move_iterator(std::begin(v)),
                  std::make_move_iterator(std::end(v)), std::ref(result),
                  [](Result& accum, Iterated iterated) -> Result& {
                    append(accum, std::move(iterated));
                    return accum;
                  });

  return result;
}
//******************************************************************************
