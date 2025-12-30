#include "union_find.h"
#include <algorithm>
#include <unordered_map>

namespace elasticapp::environment {
namespace collision {

//==============================================================================
// UnionFind::UnionFindDS Implementation
//==============================================================================

UnionFind::UnionFindDS::UnionFindDS(std::size_t max_node)
    : parent_(max_node + 1), rank_(max_node + 1, 0) {
    // Initialize: each node is its own parent (root)
    for (std::size_t i = 0; i <= max_node; ++i) {
        parent_[i] = i;
    }
}

std::size_t UnionFind::UnionFindDS::find_root(std::size_t x) {
    // Path compression: make parent point directly to root
    if (parent_[x] != x) {
        parent_[x] = find_root(parent_[x]);
    }
    return parent_[x];
}

void UnionFind::UnionFindDS::union_nodes(std::size_t x, std::size_t y) {
    std::size_t root_x = find_root(x);
    std::size_t root_y = find_root(y);

    if (root_x == root_y) {
        return;  // Already in same component
    }

    // Union by rank: attach smaller tree under larger tree
    if (rank_[root_x] < rank_[root_y]) {
        parent_[root_x] = root_y;
    } else if (rank_[root_x] > rank_[root_y]) {
        parent_[root_y] = root_x;
    } else {
        // Same rank: attach one to the other and increment rank
        parent_[root_y] = root_x;
        rank_[root_x]++;
    }
}

//==============================================================================
// UnionFind Implementation
//==============================================================================

std::vector<std::vector<std::size_t>> UnionFind::batch(
    std::vector<Contact>& contacts
) const {
    const std::size_t n_contacts = contacts.size();

    if (n_contacts == 0) {
        return {};
    }

    // Step 1: Find maximum node index to size the Union-Find structure
    std::size_t max_node = 0;
    for (const auto& contact : contacts) {
        max_node = std::max(max_node, std::max(contact.node1_idx, contact.node2_idx));
    }

    // Step 2: Initialize Union-Find data structure
    UnionFindDS uf(max_node);

    // Step 3: Union all nodes connected by contacts
    // This builds the connected components: nodes connected by contacts are in the same component
    for (const auto& contact : contacts) {
        uf.union_nodes(contact.node1_idx, contact.node2_idx);
    }

    // Step 4: Group contacts by their root component and set their indices
    // Contacts whose nodes share the same root are in the same batch
    std::unordered_map<std::size_t, std::vector<std::size_t>> batches_map;
    batches_map.reserve(n_contacts);  // Reserve space (worst case: one batch per contact)

    for (std::size_t i = 0; i < n_contacts; ++i) {
        auto& contact = contacts[i];
        // Use root of node1 (both nodes in a contact have the same root after union)
        std::size_t root = uf.find_root(contact.node1_idx);
        auto& batch = batches_map[root];

        // Set contact index to its position within the batch (matches reference implementation)
        contact.set_index(batch.size());

        // Add contact index to batch
        batch.push_back(i);
    }

    // Step 5: Convert map to vector of batches
    std::vector<std::vector<std::size_t>> batches;
    batches.reserve(batches_map.size());
    for (auto& [root, batch] : batches_map) {
        batches.push_back(std::move(batch));
    }

    return batches;
}

} // namespace collision
} // namespace elasticapp::environment
