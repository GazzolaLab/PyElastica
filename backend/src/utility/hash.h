#pragma once

#include <cstddef>
#include <utility>  // For std::pair
#include <functional>  // For std::hash

namespace elasticapp {
namespace utility {

/**
 * Hash function for std::pair<std::size_t, std::size_t> (needed for unordered_map)
 * C++11 doesn't provide std::hash for pairs by default
 */
struct PairHash {
    std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
        // Combine hashes of both elements
        // Using a simple hash combination (can be improved with boost::hash_combine)
        return std::hash<std::size_t>()(p.first) ^ (std::hash<std::size_t>()(p.second) << 1);
    }
};

} // namespace utility
} // namespace elasticapp
