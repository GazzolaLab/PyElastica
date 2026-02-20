#pragma once

#include <cstddef>
#include <vector>

namespace elasticapp {
namespace math {

/**
 * Map a global node index to the corresponding element index in a Block.
 *
 * For a rod with n_elements, there are n_elements+1 nodes.
 * The mapping strategy:
 * - For interior nodes (1 to n_elements-1): use the element before the node (element = node - 1)
 * - For the first node (0): use element 0
 * - For the last node (n_elements): use element n_elements-1
 *
 * This function handles multiple rods in a Block by using rod_start_indices
 * to determine which rod a node belongs to.
 *
 * @param node_idx Global node index in the Block
 * @param rod_start_indices Vector of starting node indices for each rod
 * @param rod_n_elems Vector of number of elements for each rod
 * @return Global element index corresponding to the node
 * @throws std::out_of_range if node_idx is invalid
 */
std::size_t node_to_element_index(
    std::size_t node_idx,
    const std::vector<std::size_t>& rod_start_indices,
    const std::vector<std::size_t>& rod_n_elems
);

/**
 * Map a global node index to the corresponding element index within its rod.
 *
 * This is a helper function that first finds which rod the node belongs to,
 * then maps it to the element index within that rod.
 *
 * @param node_idx Global node index in the Block
 * @param rod_start_indices Vector of starting node indices for each rod
 * @param rod_n_elems Vector of number of elements for each rod
 * @return Pair of (rod_index, local_element_index)
 * @throws std::out_of_range if node_idx is invalid
 */
std::pair<std::size_t, std::size_t> node_to_rod_and_element(
    std::size_t node_idx,
    const std::vector<std::size_t>& rod_start_indices,
    const std::vector<std::size_t>& rod_n_elems
);

} // namespace math
} // namespace elasticapp
