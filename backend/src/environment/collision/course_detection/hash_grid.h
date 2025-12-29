#pragma once

#include "../types.h"
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <memory>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace elasticapp {
namespace collision {

/**
 * HashGrid coarse collision detection policy.
 *
 * Implements hierarchical spatial hashing for O(N) broad-phase collision detection.
 * Based on the Elastica HashGrids implementation, adapted for node-based systems.
 *
 * Key features:
 * - Hierarchical grids: Multiple grids with different cell sizes for optimal performance
 * - Precomputed neighbor offsets: Fast access to adjacent cells
 * - Bitwise hash operations: Efficient cell coordinate computation (requires power-of-2 dimensions)
 * - Automatic grid sizing: Adapts to node sizes at runtime
 *
 * Algorithm:
 * 1. Creates a hierarchy of grids with different cell sizes
 * 2. Assigns each node to the grid with smallest cells larger than the node's diameter
 * 3. Uses spatial hashing to partition nodes into grid cells
 * 4. For collision detection, checks nodes in same cell and 26 neighboring cells (3D)
 *
 * This is a CRTP policy class that will be mixed into CollisionSystem.
 */
class HashGrid {
public:
    /**
     * Configuration parameters for the hash grid.
     */
    struct Parameters {
        std::size_t initial_x_cell_count = 64;      // Initial grid size in x (must be power of 2)
        std::size_t initial_y_cell_count = 64;      // Initial grid size in y (must be power of 2)
        std::size_t initial_z_cell_count = 64;      // Initial grid size in z (must be power of 2)
        std::size_t initial_cell_vector_size = 16;   // Initial capacity for node vectors in cells
        std::size_t minimal_grid_density = 4;        // Minimum cells per node
        std::size_t hashgrid_activation_threshold = 10; // Min nodes to use hash grid
        double hierarchy_factor = 2.0;              // Factor between grid levels

        Parameters() = default;
    };

    /**
     * Default constructor.
     */
    HashGrid();

    /**
     * Constructor with parameters.
     */
    explicit HashGrid(const Parameters& params);

    /**
     * Destructor.
     */
    ~HashGrid();

    /**
     * Perform coarse collision detection.
     *
     * Identifies pairs of nodes that are potentially colliding using
     * hierarchical spatial hashing. Returns candidate pairs for fine detection.
     *
     * @param positions Node positions (3 x n_nodes)
     * @param radii Node radii (1 x n_nodes)
     * @return Vector of pairs (node1_idx, node2_idx) that are potentially colliding
     */
    std::vector<std::pair<std::size_t, std::size_t>> detect(
        const Eigen::MatrixXd& positions,
        const Eigen::MatrixXd& radii
    ) const;

    /**
     * Clear all grids (for reuse).
     */
    void clear();

private:
    // Forward declaration
    class SingleHashGrid;

    /**
     * Grid cell coordinates (3D integer coordinates).
     */
    struct CellCoord {
        int x, y, z;

        bool operator==(const CellCoord& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    /**
     * Hash function for CellCoord to use in unordered_map.
     */
    struct CellCoordHash {
        std::size_t operator()(const CellCoord& coord) const {
            const std::size_t h1 = std::hash<int>{}(coord.x);
            const std::size_t h2 = std::hash<int>{}(coord.y);
            const std::size_t h3 = std::hash<int>{}(coord.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    /**
     * Node information for grid assignment.
     */
    struct NodeInfo {
        std::size_t index;
        Eigen::Vector3d position;
        double radius;
        double diameter;  // 2 * radius

        NodeInfo(std::size_t idx, const Eigen::Vector3d& pos, double r)
            : index(idx), position(pos), radius(r), diameter(2.0 * r) {}
    };

    /**
     * Single hash grid in the hierarchy.
     * Implements one level of the hierarchical grid system.
     */
    class SingleHashGrid {
    public:
        /**
         * Constructor.
         *
         * @param cell_span Size of each grid cell (edge length)
         * @param params Configuration parameters
         */
        SingleHashGrid(double cell_span, const Parameters& params);

        /**
         * Destructor.
         */
        ~SingleHashGrid();

        /**
         * Get cell span (cell size).
         */
        double get_cell_span() const noexcept { return cell_span_; }

        /**
         * Add a node to this grid.
         */
        void add_node(const NodeInfo& node);

        /**
         * Detect collisions within this grid and with nodes from finer grids.
         *
         * @param candidate_pairs Output vector for candidate pairs
         * @param finer_grid_nodes Nodes from finer grids (smaller cell sizes) to check against
         */
        void detect_collisions(
            std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
            const std::vector<NodeInfo>& finer_grid_nodes = {}
        ) const;

        /**
         * Clear all nodes from this grid.
         */
        void clear();

        /**
         * Get all nodes in this grid.
         */
        std::vector<NodeInfo> get_all_nodes() const;

    private:
        /**
         * Cell structure for storing nodes.
         */
        struct Cell {
            std::vector<std::size_t> node_indices;  // Indices into nodes_ vector
            long* neighbor_offsets;                 // Offsets to neighboring cells (27 in 3D)
            bool is_border_cell;                    // Whether this is a border cell

            Cell() : neighbor_offsets(nullptr), is_border_cell(false) {}
        };

        /**
         * Compute hash (cell index) from position.
         * Uses bitwise operations for efficiency (requires power-of-2 grid dimensions).
         */
        std::size_t hash(const Eigen::Vector3d& position) const;

        /**
         * Initialize neighbor offsets for all cells.
         */
        void initialize_neighbor_offsets();

        /**
         * Clear neighbor offsets.
         */
        void clear_neighbor_offsets();

        /**
         * Get number of neighbors (27 in 3D).
         */
        static constexpr std::size_t get_number_of_neighbors() { return 27; }

        /**
         * Get half number of neighbors (13, excluding center).
         */
        static constexpr std::size_t get_half_number_of_neighbors() { return 13; }

        // Member variables
        Parameters params_;
        double cell_span_;
        double inverse_cell_span_;  // Precomputed 1.0 / cell_span_ for efficiency

        // Grid dimensions (must be powers of 2 for bitwise operations)
        std::size_t x_cell_count_;
        std::size_t y_cell_count_;
        std::size_t z_cell_count_;
        std::size_t xy_cell_count_;   // x_cell_count_ * y_cell_count_
        std::size_t xyz_cell_count_;  // Total number of cells

        // Hash masks for bitwise modulo operations
        std::size_t x_hash_mask_;
        std::size_t y_hash_mask_;
        std::size_t z_hash_mask_;

        // Grid cells (linear array)
        std::vector<Cell> cells_;

        // Standard neighbor offsets (for inner cells)
        long std_neighbor_offsets_[27];

        // Node storage
        std::vector<NodeInfo> nodes_;
        std::unordered_map<std::size_t, std::size_t> node_to_cell_;  // node index -> cell index
    };

    /**
     * Compute appropriate cell span for a node based on its diameter.
     */
    static double compute_cell_span_for_node(double node_diameter, const Parameters& params);

    /**
     * Round up to next power of 2.
     */
    static std::size_t round_up_to_power_of_2(std::size_t n);

    // Member variables
    Parameters params_;
    mutable std::list<std::unique_ptr<SingleHashGrid>> grids_;  // Grids sorted by cell span (ascending)
};

} // namespace collision
} // namespace elasticapp
