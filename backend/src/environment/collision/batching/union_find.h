#pragma once

#include "../types.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace elasticapp::environment {
namespace collision {

/**
 * UnionFind batching policy.
 *
 * Groups contacts into batches using a Union-Find data structure.
 * Based on the Elastica UnionFind batching implementation.
 *
 * Algorithm:
 * 1. Build Union-Find structure on nodes: For each contact, union the two nodes it connects
 * 2. Identify root nodes: Each connected component has a unique root
 * 3. Group contacts by root: Contacts whose nodes share the same root are in the same batch
 *
 * This creates independent batches of contacts that can be processed in parallel,
 * as contacts in different batches don't share any nodes and thus don't interfere.
 *
 * This is a CRTP policy class that will be mixed into CollisionSystem.
 */
class UnionFind {
public:
    /**
     * Default constructor.
     */
    UnionFind() = default;

    /**
     * Destructor.
     */
    ~UnionFind() = default;

    /**
     * Group contacts into batches.
     *
     * Uses Union-Find to identify connected components of contacts.
     * Contacts are connected if they share a common node (directly or transitively).
     *
     * Algorithm:
     * 1. Initialize Union-Find: Each node is its own root initially
     * 2. Union nodes: For each contact, union node1 and node2
     * 3. Group contacts: Contacts whose nodes have the same root are grouped together
     * 4. Set contact indices: Each contact's index is set to its position within its batch
     *
     * @param contacts Vector of all contacts (non-const to allow setting index)
     * @return Vector of batches, where each batch is a vector of contact indices
     */
    std::vector<std::vector<std::size_t>> batch(
        std::vector<Contact>& contacts
    ) const;

private:
    /**
     * Union-Find data structure for finding connected components.
     */
    class UnionFindDS {
    public:
        /**
         * Constructor with maximum node index.
         */
        explicit UnionFindDS(std::size_t max_node);

        /**
         * Find root of a node with path compression.
         */
        std::size_t find_root(std::size_t x);

        /**
         * Union two nodes.
         */
        void union_nodes(std::size_t x, std::size_t y);

    private:
        std::vector<std::size_t> parent_;  // parent[i] = parent of node i
        std::vector<std::size_t> rank_;    // rank[i] = approximate tree height (for union by rank)
    };
};

} // namespace collision
} // namespace elasticapp::environment
