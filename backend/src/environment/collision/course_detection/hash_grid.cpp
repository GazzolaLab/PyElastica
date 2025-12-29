#include "hash_grid.h"
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <limits>

namespace elasticapp {
namespace collision {

//==============================================================================
// HashGrid::SingleHashGrid Implementation
//==============================================================================

HashGrid::SingleHashGrid::SingleHashGrid(double cell_span, const Parameters& params)
    : params_(params),
      cell_span_(cell_span),
      inverse_cell_span_(1.0 / cell_span),
      x_cell_count_(params.initial_x_cell_count),
      y_cell_count_(params.initial_y_cell_count),
      z_cell_count_(params.initial_z_cell_count) {

    // Ensure grid dimensions are powers of 2 for bitwise operations
    x_cell_count_ = round_up_to_power_of_2(x_cell_count_);
    y_cell_count_ = round_up_to_power_of_2(y_cell_count_);
    z_cell_count_ = round_up_to_power_of_2(z_cell_count_);

    // Compute hash masks (for bitwise modulo: x & mask instead of x % size)
    x_hash_mask_ = x_cell_count_ - 1;
    y_hash_mask_ = y_cell_count_ - 1;
    z_hash_mask_ = z_cell_count_ - 1;

    xy_cell_count_ = x_cell_count_ * y_cell_count_;
    xyz_cell_count_ = xy_cell_count_ * z_cell_count_;

    // Allocate cells
    cells_.resize(xyz_cell_count_);

    // Initialize neighbor offsets
    initialize_neighbor_offsets();
}

HashGrid::SingleHashGrid::~SingleHashGrid() {
    clear_neighbor_offsets();
}

void HashGrid::SingleHashGrid::clear_neighbor_offsets() {
    for (auto& cell : cells_) {
        if (cell.neighbor_offsets != nullptr && cell.neighbor_offsets != std_neighbor_offsets_) {
            delete[] cell.neighbor_offsets;
            cell.neighbor_offsets = nullptr;
        }
    }
}

void HashGrid::SingleHashGrid::initialize_neighbor_offsets() {
    // Initialize standard neighbor offsets (for inner cells)
    // 27 neighbors in 3D: (-1,-1,-1) to (1,1,1)
    long xc = static_cast<long>(x_cell_count_);
    long yc = static_cast<long>(y_cell_count_);
    long zc = static_cast<long>(z_cell_count_);
    long xyc = static_cast<long>(xy_cell_count_);

    std::size_t i = 0;
    for (long zz = -xyc; zz <= xyc; zz += xyc) {
        for (long yy = -xc; yy <= xc; yy += xc) {
            for (long xx = -1; xx <= 1; ++xx, ++i) {
                std_neighbor_offsets_[i] = xx + yy + zz;
            }
        }
    }

    // Set up neighbor offsets for each cell
    for (std::size_t z = 0; z < z_cell_count_; ++z) {
        for (std::size_t y = 0; y < y_cell_count_; ++y) {
            for (std::size_t x = 0; x < x_cell_count_; ++x) {
                std::size_t cell_idx = x + y * x_cell_count_ + z * xy_cell_count_;
                Cell& cell = cells_[cell_idx];

                // Check if border cell
                bool is_border = (x == 0 || x == x_cell_count_ - 1 ||
                                 y == 0 || y == y_cell_count_ - 1 ||
                                 z == 0 || z == z_cell_count_ - 1);
                cell.is_border_cell = is_border;

                if (is_border) {
                    // Border cells need custom offsets (wrapping or clamping)
                    cell.neighbor_offsets = new long[27];

                    i = 0;
                    for (long zz = -xyc; zz <= xyc; zz += xyc) {
                        long zo = zz;
                        // Handle z wrapping
                        if (z == 0 && zz == -xyc) {
                            zo = static_cast<long>(xyz_cell_count_) - xyc;
                        } else if (z == z_cell_count_ - 1 && zz == xyc) {
                            zo = xyc - static_cast<long>(xyz_cell_count_);
                        }

                        for (long yy = -xc; yy <= xc; yy += xc) {
                            long yo = yy;
                            // Handle y wrapping
                            if (y == 0 && yy == -xc) {
                                yo = xyc - xc;
                            } else if (y == y_cell_count_ - 1 && yy == xc) {
                                yo = xc - xyc;
                            }

                            for (long xx = -1; xx <= 1; ++xx, ++i) {
                                long xo = xx;
                                // Handle x wrapping
                                if (x == 0 && xx == -1) {
                                    xo = xc - 1;
                                } else if (x == x_cell_count_ - 1 && xx == 1) {
                                    xo = 1 - xc;
                                }

                                cell.neighbor_offsets[i] = xo + yo + zo;
                            }
                        }
                    }
                } else {
                    // Inner cells use standard offsets
                    cell.neighbor_offsets = std_neighbor_offsets_;
                }
            }
        }
    }
}

std::size_t HashGrid::SingleHashGrid::hash(const Eigen::Vector3d& position) const {
    const double x = position.x();
    const double y = position.y();
    const double z = position.z();

    std::size_t x_hash, y_hash, z_hash;

    // Use inverse multiplication and bitwise AND for efficiency
    if (x < 0) {
        double i = (-x) * inverse_cell_span_;
        x_hash = x_cell_count_ - 1 - (static_cast<std::size_t>(i) & x_hash_mask_);
    } else {
        double i = x * inverse_cell_span_;
        x_hash = static_cast<std::size_t>(i) & x_hash_mask_;
    }

    if (y < 0) {
        double i = (-y) * inverse_cell_span_;
        y_hash = y_cell_count_ - 1 - (static_cast<std::size_t>(i) & y_hash_mask_);
    } else {
        double i = y * inverse_cell_span_;
        y_hash = static_cast<std::size_t>(i) & y_hash_mask_;
    }

    if (z < 0) {
        double i = (-z) * inverse_cell_span_;
        z_hash = z_cell_count_ - 1 - (static_cast<std::size_t>(i) & z_hash_mask_);
    } else {
        double i = z * inverse_cell_span_;
        z_hash = static_cast<std::size_t>(i) & z_hash_mask_;
    }

    return x_hash + y_hash * x_cell_count_ + z_hash * xy_cell_count_;
}

void HashGrid::SingleHashGrid::add_node(const NodeInfo& node) {
    nodes_.push_back(node);
    std::size_t node_idx = nodes_.size() - 1;

    // Compute cell index
    std::size_t cell_idx = hash(node.position);

    // Add to cell
    Cell& cell = cells_[cell_idx];
    if (cell.node_indices.empty()) {
        cell.node_indices.reserve(params_.initial_cell_vector_size);
    }
    cell.node_indices.push_back(node_idx);

    // Store mapping
    node_to_cell_[node.index] = cell_idx;
}

void HashGrid::SingleHashGrid::detect_collisions(
    std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
    const std::vector<NodeInfo>& finer_grid_nodes
) const {
    // Use set to avoid duplicates
    std::unordered_set<std::size_t> processed_pairs;

    auto make_pair_key = [](std::size_t i, std::size_t j) -> std::size_t {
        if (i > j) std::swap(i, j);
        return i * 4294967291ULL + j;  // Large prime for hashing
    };

    // Process collisions within this grid
    for (std::size_t cell_idx = 0; cell_idx < cells_.size(); ++cell_idx) {
        const Cell& cell = cells_[cell_idx];
        if (cell.node_indices.empty()) continue;

        const auto& node_indices = cell.node_indices;

        // Check pairs within the same cell
        for (std::size_t i = 0; i < node_indices.size(); ++i) {
            for (std::size_t j = i + 1; j < node_indices.size(); ++j) {
                std::size_t idx1 = nodes_[node_indices[i]].index;
                std::size_t idx2 = nodes_[node_indices[j]].index;

                std::size_t pair_key = make_pair_key(idx1, idx2);
                if (processed_pairs.find(pair_key) == processed_pairs.end()) {
                    candidate_pairs.push_back({idx1, idx2});
                    processed_pairs.insert(pair_key);
                }
            }
        }

        // Check pairs with neighboring cells
        constexpr std::size_t hnn = get_half_number_of_neighbors();
        for (std::size_t i = 0; i < hnn; ++i) {
            long offset = cell.neighbor_offsets[i];
            std::size_t neighbor_idx = cell_idx + offset;

            // Bounds check
            if (neighbor_idx >= cells_.size()) {
                continue;  // Out of bounds
            }

            const Cell& neighbor_cell = cells_[neighbor_idx];
            if (neighbor_cell.node_indices.empty()) continue;

            // Check all pairs between this cell and neighbor
            for (std::size_t idx1 : node_indices) {
                for (std::size_t idx2 : neighbor_cell.node_indices) {
                    std::size_t n1 = nodes_[idx1].index;
                    std::size_t n2 = nodes_[idx2].index;

                    if (n1 < n2) {  // Only check once per pair
                        std::size_t pair_key = make_pair_key(n1, n2);
                        if (processed_pairs.find(pair_key) == processed_pairs.end()) {
                            candidate_pairs.push_back({n1, n2});
                            processed_pairs.insert(pair_key);
                        }
                    }
                }
            }
        }
    }

    // Check collisions with nodes from finer grids
    if (!finer_grid_nodes.empty()) {
        for (const auto& fine_node : finer_grid_nodes) {
            std::size_t cell_idx = hash(fine_node.position);
            const Cell& cell = cells_[cell_idx];

            // Check all neighbors
            for (std::size_t i = 0; i < get_number_of_neighbors(); ++i) {
                long offset = cell.neighbor_offsets[i];
                std::size_t neighbor_idx = cell_idx + offset;

                // Bounds check
                if (neighbor_idx >= cells_.size()) {
                    continue;
                }

                const Cell& neighbor_cell = cells_[neighbor_idx];
                for (std::size_t idx : neighbor_cell.node_indices) {
                    std::size_t n1 = fine_node.index;
                    std::size_t n2 = nodes_[idx].index;

                    if (n1 < n2) {
                        std::size_t pair_key = make_pair_key(n1, n2);
                        if (processed_pairs.find(pair_key) == processed_pairs.end()) {
                            candidate_pairs.push_back({n1, n2});
                            processed_pairs.insert(pair_key);
                        }
                    }
                }
            }
        }
    }
}

void HashGrid::SingleHashGrid::clear() {
    nodes_.clear();
    node_to_cell_.clear();
    for (auto& cell : cells_) {
        cell.node_indices.clear();
    }
}

std::vector<HashGrid::NodeInfo> HashGrid::SingleHashGrid::get_all_nodes() const {
    return nodes_;
}

//==============================================================================
// HashGrid Implementation
//==============================================================================

HashGrid::HashGrid() : HashGrid(Parameters()) {}

HashGrid::HashGrid(const Parameters& params) : params_(params) {}

HashGrid::~HashGrid() {
    clear();
}

std::size_t HashGrid::round_up_to_power_of_2(std::size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if (sizeof(std::size_t) > 4) {
        n |= n >> 32;
    }
    return n + 1;
}

double HashGrid::compute_cell_span_for_node(double node_diameter, const Parameters& params) {
    // Cell span should be slightly larger than node diameter
    // Use hierarchy factor to determine appropriate level
    double base_span = node_diameter * std::sqrt(params.hierarchy_factor);

    // Round to reasonable value
    return base_span;
}

void HashGrid::clear() {
    grids_.clear();
}

std::vector<std::pair<std::size_t, std::size_t>> HashGrid::detect(
    const Eigen::MatrixXd& positions,
    const Eigen::MatrixXd& radii
) const {
    const Eigen::Index n_nodes = positions.cols();

    if (n_nodes == 0) {
        return std::vector<std::pair<std::size_t, std::size_t>>();
    }

    // Clear previous state
    clear();

    // Build node info
    std::vector<NodeInfo> nodes;
    nodes.reserve(n_nodes);
    double max_diameter = 0.0;

    for (Eigen::Index i = 0; i < n_nodes; ++i) {
        Eigen::Vector3d pos = positions.col(i);
        double radius = std::abs(radii(i));
        double diameter = 2.0 * radius;
        max_diameter = std::max(max_diameter, diameter);

        nodes.emplace_back(static_cast<std::size_t>(i), pos, radius);
    }

    // If too few nodes, use simple approach
    if (n_nodes < params_.hashgrid_activation_threshold) {
        // Simple all-pairs (for very small systems)
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t j = i + 1; j < nodes.size(); ++j) {
                pairs.push_back({nodes[i].index, nodes[j].index});
            }
        }
        return pairs;
    }

    // Create initial grid with appropriate cell span
    double initial_cell_span = compute_cell_span_for_node(max_diameter, params_);
    auto initial_grid = std::make_unique<SingleHashGrid>(initial_cell_span, params_);
    grids_.push_back(std::move(initial_grid));

    // Assign nodes to appropriate grids
    for (const auto& node : nodes) {
        double required_cell_span = compute_cell_span_for_node(node.diameter, params_);

        // Find or create appropriate grid
        SingleHashGrid* target_grid = nullptr;
        auto it = grids_.begin();

        while (it != grids_.end()) {
            double grid_span = (*it)->get_cell_span();

            if (node.diameter < grid_span) {
                // Check if next grid is too small
                auto next_it = std::next(it);
                if (next_it == grids_.end() || (*next_it)->get_cell_span() > required_cell_span) {
                    // This grid is appropriate
                    target_grid = it->get();
                    break;
                }
            }
            ++it;
        }

        // If no appropriate grid found, create a new one
        if (target_grid == nullptr) {
            // Create grid with larger cell span
            double new_span = required_cell_span;
            if (!grids_.empty()) {
                double largest_span = grids_.back()->get_cell_span();
                while (new_span <= largest_span) {
                    new_span *= params_.hierarchy_factor;
                }
            }

            auto new_grid = std::make_unique<SingleHashGrid>(new_span, params_);
            target_grid = new_grid.get();
            grids_.push_back(std::move(new_grid));
        }

        target_grid->add_node(node);
    }

    // Detect collisions hierarchically
    std::vector<std::pair<std::size_t, std::size_t>> candidate_pairs;

    // Process grids from finest to coarsest
    for (auto it = grids_.begin(); it != grids_.end(); ++it) {
        // Collect nodes from finer grids
        std::vector<NodeInfo> finer_nodes;
        for (auto prev_it = grids_.begin(); prev_it != it; ++prev_it) {
            auto grid_nodes = (*prev_it)->get_all_nodes();
            finer_nodes.insert(finer_nodes.end(), grid_nodes.begin(), grid_nodes.end());
        }

        // Detect collisions in this grid
        (*it)->detect_collisions(candidate_pairs, finer_nodes);
    }

    return candidate_pairs;
}

} // namespace collision
} // namespace elasticapp
