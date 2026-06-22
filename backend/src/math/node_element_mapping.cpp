#include "node_element_mapping.h"
#include <stdexcept>
#include <algorithm>

namespace elasticapp {
namespace math {

std::size_t node_to_element_index(
    std::size_t node_idx,
    const std::vector<std::size_t>& rod_start_indices,
    const std::vector<std::size_t>& rod_n_elems
) {
    if (rod_start_indices.empty() || rod_n_elems.empty()) {
        throw std::invalid_argument("rod_start_indices and rod_n_elems cannot be empty");
    }

    // Find which rod this node belongs to
    std::size_t rod_index = 0;
    for (std::size_t i = 0; i < rod_start_indices.size(); ++i) {
        if (i + 1 < rod_start_indices.size()) {
            // Not the last rod
            if (node_idx >= rod_start_indices[i] && node_idx < rod_start_indices[i + 1]) {
                rod_index = i;
                break;
            }
        } else {
            // Last rod
            if (node_idx >= rod_start_indices[i]) {
                rod_index = i;
                break;
            }
        }
    }

    // Check if node_idx is valid for the found rod
    std::size_t rod_start = rod_start_indices[rod_index];
    std::size_t rod_n_nodes = rod_n_elems[rod_index] + 1;  // n_elements + 1 nodes
    std::size_t local_node_idx = node_idx - rod_start;

    if (local_node_idx >= rod_n_nodes) {
        throw std::out_of_range("node_idx is beyond the last node of its rod");
    }

    // Map local node index to local element index
    // Strategy:
    // - Node 0 -> Element 0
    // - Node i (1 to n_elements-1) -> Element i-1
    // - Node n_elements (last) -> Element n_elements-1
    std::size_t local_elem_idx;
    if (local_node_idx == 0) {
        local_elem_idx = 0;
    } else if (local_node_idx < rod_n_elems[rod_index]) {
        local_elem_idx = local_node_idx - 1;
    } else {
        // Last node
        local_elem_idx = rod_n_elems[rod_index] - 1;
    }

    // Convert to global element index
    // Need to compute element start indices
    std::size_t elem_start = 0;
    for (std::size_t i = 0; i < rod_index; ++i) {
        elem_start += rod_n_elems[i];
    }

    return elem_start + local_elem_idx;
}

std::pair<std::size_t, std::size_t> node_to_rod_and_element(
    std::size_t node_idx,
    const std::vector<std::size_t>& rod_start_indices,
    const std::vector<std::size_t>& rod_n_elems
) {
    if (rod_start_indices.empty() || rod_n_elems.empty()) {
        throw std::invalid_argument("rod_start_indices and rod_n_elems cannot be empty");
    }

    // Find which rod this node belongs to
    std::size_t rod_index = 0;
    for (std::size_t i = 0; i < rod_start_indices.size(); ++i) {
        if (i + 1 < rod_start_indices.size()) {
            // Not the last rod
            if (node_idx >= rod_start_indices[i] && node_idx < rod_start_indices[i + 1]) {
                rod_index = i;
                break;
            }
        } else {
            // Last rod
            if (node_idx >= rod_start_indices[i]) {
                rod_index = i;
                break;
            }
        }
    }

    // Check if node_idx is valid for the found rod
    std::size_t rod_start = rod_start_indices[rod_index];
    std::size_t rod_n_nodes = rod_n_elems[rod_index] + 1;  // n_elements + 1 nodes
    std::size_t local_node_idx = node_idx - rod_start;

    if (local_node_idx >= rod_n_nodes) {
        throw std::out_of_range("node_idx is beyond the last node of its rod");
    }

    // Map local node index to local element index
    std::size_t local_elem_idx;
    if (local_node_idx == 0) {
        local_elem_idx = 0;
    } else if (local_node_idx < rod_n_elems[rod_index]) {
        local_elem_idx = local_node_idx - 1;
    } else {
        // Last node
        local_elem_idx = rod_n_elems[rod_index] - 1;
    }

    return std::make_pair(rod_index, local_elem_idx);
}

} // namespace math
} // namespace elasticapp
