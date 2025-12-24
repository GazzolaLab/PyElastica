#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>
#include "traits.h"
#include "system.h"
#include "block_get_impl.h"
#include "block_view.h"
#include "operations.h"

namespace elasticapp {

// Block class using CRTP with SystemModel and Operations
// Block inherits from SystemModel, giving it access to all system variables and methods
// Block also inherits from OperationsType using CRTP, allowing for extensible operations
// SystemModel must be a System<ValidVariable...> type
// OperationsType must be a template class that takes the Block type as a template parameter
template<SystemModel SystemType, template<typename> class OperationsType = DefaultOperations>
class Block : public SystemType, public OperationsType<Block<SystemType, OperationsType>> {
public:

    using System = SystemType;
    using Variables = typename SystemType::Variables;
    using View = BlockView<SystemType>;

    constexpr static std::size_t GHOST_NODE_WIDTH = 1; // Size of ghost for node variables.

    // Constructor: takes list of element counts per rod
    explicit Block(const std::vector<std::size_t>& n_elems_per_rod) {
        // Validate input: reject empty list or less than 6 total elements
        if (n_elems_per_rod.empty()) {
            throw std::invalid_argument("n_elems_per_rod cannot be empty");
        }

        std::size_t total_elements = 0;
        for (std::size_t n_elems : n_elems_per_rod) {
            total_elements += n_elems;
        }
        if (total_elements < 6) {
            throw std::invalid_argument("Total number of elements must be at least 6, got " +
                                       std::to_string(total_elements));
        }

        compute_width_and_indices(n_elems_per_rod);
        depth_ = SystemType::get_depth();
        data_ = MatrixType(static_cast<IndexType>(depth_), static_cast<IndexType>(width_));
        initialize_ghost_indices();
        reset_ghost();  // Initialize all ghost values
    }

    std::pair<std::size_t, std::size_t> shape() const {
        return std::make_pair(
            static_cast<std::size_t>(data_.rows()),
            static_cast<std::size_t>(data_.cols())
        );
    }

    // Accessors
    std::size_t width() const { return width_; }
    std::size_t depth() const { return depth_; }
    std::size_t n_systems() const { return rod_start_indices_.size(); }

    std::size_t system_start_index(std::size_t rod_index) const {
        if (rod_index >= rod_start_indices_.size()) {
            throw std::out_of_range("System index out of range");
        }
        return rod_start_indices_[rod_index];
    }

    MatrixType& data() { return data_; }
    const MatrixType& data() const { return data_; }

    // Get number of elements for a specific rod
    std::size_t rod_n_elems(std::size_t rod_index) const {
        if (rod_index >= rod_n_elems_.size()) {
            throw std::out_of_range("System index out of range");
        }
        return rod_n_elems_[rod_index];
    }

    std::size_t rod_n_nodes(std::size_t rod_index) const {
        return rod_n_elems(rod_index) + 1;
    }

    std::size_t rod_n_voronoi(std::size_t rod_index) const {
        return rod_n_elems(rod_index) - 1;
    }

    // Get the n_elems_per_rod vector (for BlockView construction)
    const std::vector<std::size_t>& get_n_elems_per_rod() const { return rod_n_elems_; }

    // Get ghost indices (return const references for efficiency)
    inline const std::vector<std::size_t>& ghost_nodes_idx() const {
        return ghost_nodes_idx_;
    }
    inline const std::vector<std::size_t>& ghost_elems_idx() const {
        return ghost_elems_idx_;
    }
    inline const std::vector<std::size_t>& ghost_voronoi_idx() const {
        return ghost_voronoi_idx_;
    }

    // Reset ghost values for a specific variable
    // Uses VariableTag::ghost_value and appropriate ghost indices based on placement
    template<typename VariableTag>
    void reset_ghost_for_variable() {
        static_assert(tuple_contains_v<VariableTag, system_variables_t<SystemType>>,
            "VariableTag is not a valid member of tuple SystemType::Variables");

        // Compute row offset for this variable
        constexpr std::size_t row_offset = compute_variable_offset<VariableTag, SystemType>();
        constexpr std::size_t var_dimension = get_dimension_v<VariableTag>;

        // Get appropriate ghost indices based on placement
        std::vector<std::size_t> ghost_indices;
        if constexpr (std::is_base_of_v<Placement::OnNode, VariableTag>) {
            ghost_indices = ghost_nodes_idx();
        } else if constexpr (std::is_base_of_v<Placement::OnElement, VariableTag>) {
            ghost_indices = ghost_elems_idx();
        } else if constexpr (std::is_base_of_v<Placement::OnVoronoi, VariableTag>) {
            ghost_indices = ghost_voronoi_idx();
        }

        // Set ghost values at each ghost index
        // Note: ghost indices are in the full width coordinate system
        // For OnElement and OnVoronoi variables, we need to ensure ghost indices
        // are within the adjusted width (since get() returns adjusted width views)
        const auto& ghost_val = VariableTag::ghost_value;

        // Optimized: Add simd hint for inner loop (var_dimension is compile-time constant)
        for (std::size_t ghost_col : ghost_indices) {
            // Only reset ghost values that are within the adjusted width
            // (ghost indices beyond adjusted width are not accessible via get())
            IndexType data_col = static_cast<IndexType>(ghost_col);
            #ifdef ELASTICAPP_USE_THREADING
            #pragma omp simd
            #endif
            for (std::size_t row = 0; row < var_dimension; ++row) {
                IndexType data_row = static_cast<IndexType>(row_offset + row);
                data_(data_row, data_col) = ghost_val(static_cast<IndexType>(row), 0);
            }
        }
    }

    // Reset ghost values for all variables
    // Iterates over all variables and calls reset_ghost_for_variable for each
    void reset_ghost() {
        reset_ghost_impl<system_variables_t<SystemType>, 0>();
    }

    // Get a view for a specific variable across all rods
    // Returns a view into the variable's data (rows) and adjusted columns based on placement
    // - OnNode: full width
    // - OnElement: width - 1
    // - OnVoronoi: width - 2
    // No data is copied - returns a reference to the same matrix
    template<typename VariableTag>
    auto get() {
        // Assert VariableTag is a valid member of tuple SystemType::Variables
        static_assert(tuple_contains_v<VariableTag, system_variables_t<SystemType>>,
            "VariableTag is not a valid member of tuple SystemType::Variables");
        // Compute row offset for this variable
        constexpr std::size_t row_offset = compute_variable_offset<VariableTag, SystemType>();
        constexpr std::size_t var_dimension = get_dimension_v<VariableTag>;

        // Adjust width based on placement type
        std::size_t adjusted_width = width_;
        if constexpr (std::is_base_of_v<Placement::OnElement, VariableTag>) {
            adjusted_width = width_ > 0 ? width_ - 1 : 0;
        } else if constexpr (std::is_base_of_v<Placement::OnVoronoi, VariableTag>) {
            adjusted_width = width_ > 1 ? width_ - 2 : 0;
        }
        // OnNode: no adjustment needed (full width)

        // Return a view into the specific rows and adjusted columns
        return get_block_slice(data_, row_offset, var_dimension, 0, adjusted_width);
    }

    template<typename VariableTag>
    auto get() const {
        // Assert VariableTag is a valid member of tuple SystemType::Variables
        static_assert(tuple_contains_v<VariableTag, system_variables_t<SystemType>>,
            "VariableTag is not a valid member of tuple SystemType::Variables");
        // Compute row offset for this variable
        constexpr std::size_t row_offset = compute_variable_offset<VariableTag, SystemType>();
        constexpr std::size_t var_dimension = get_dimension_v<VariableTag>;

        // Adjust width based on placement type
        std::size_t adjusted_width = width_;
        if constexpr (std::is_base_of_v<Placement::OnElement, VariableTag>) {
            adjusted_width = width_ > 0 ? width_ - 1 : 0;
        } else if constexpr (std::is_base_of_v<Placement::OnVoronoi, VariableTag>) {
            adjusted_width = width_ > 1 ? width_ - 2 : 0;
        }
        // OnNode: no adjustment needed (full width)

        // Return a view into the specific rows and adjusted columns
        return get_block_slice(data_, row_offset, var_dimension, 0, adjusted_width);
    }

    // Create a view for a specific rod
    View at(std::size_t rod_index);

private:
    MatrixType data_;
    std::size_t width_;
    std::size_t depth_;
    std::vector<std::size_t> rod_start_indices_;
    std::vector<std::size_t> rod_n_elems_;  // Store n_elems for each rod

    std::vector<std::size_t> ghost_nodes_idx_; // Indices of ghost nodes between rods.
    std::vector<std::size_t> ghost_elems_idx_; // Indices of ghost elements between rods.
    std::vector<std::size_t> ghost_voronoi_idx_; // Indices of ghost voronoi nodes between rods.

    void compute_width_and_indices(const std::vector<std::size_t>& n_elems_per_rod) {
        width_ = 0;
        rod_start_indices_.clear();
        rod_n_elems_.clear();
        rod_start_indices_.reserve(n_elems_per_rod.size());
        rod_n_elems_.reserve(n_elems_per_rod.size());

        for (std::size_t n_elems : n_elems_per_rod) {
            rod_start_indices_.push_back(width_);
            rod_n_elems_.push_back(n_elems);
            // Each rod has n_elems + 1 nodes
            width_ += n_elems + 1 + GHOST_NODE_WIDTH;
        }
        width_ -= GHOST_NODE_WIDTH; // Remove the last ghost node width.
    }

    void initialize_ghost_indices() {
        ghost_nodes_idx_.reserve(rod_n_elems_.size() * GHOST_NODE_WIDTH);
        ghost_elems_idx_.reserve(rod_n_elems_.size() * (1 + GHOST_NODE_WIDTH));
        ghost_voronoi_idx_.reserve(rod_n_elems_.size() * (2 + GHOST_NODE_WIDTH));

        std::size_t cumulative_nodes = 0;
        for (std::size_t i = 0; i < rod_n_elems_.size(); ++i) {
            cumulative_nodes += rod_n_elems_[i] + 1;  // n_elems + 1 = n_nodes
            ghost_elems_idx_.push_back(cumulative_nodes - 1);
            ghost_voronoi_idx_.push_back(cumulative_nodes - 2);
            ghost_voronoi_idx_.push_back(cumulative_nodes - 1);
            for (std::size_t j = 0; j < GHOST_NODE_WIDTH; ++j) {
                ghost_nodes_idx_.push_back(cumulative_nodes + j);
                ghost_elems_idx_.push_back(cumulative_nodes + j);
                ghost_voronoi_idx_.push_back(cumulative_nodes + j);
            }
            cumulative_nodes += GHOST_NODE_WIDTH;
        }

        ghost_nodes_idx_.pop_back();
        ghost_elems_idx_.pop_back();
        ghost_voronoi_idx_.pop_back();
    }

    // Helper to iterate over all variables and reset ghost values
    template<typename VariablesTuple, std::size_t Index>
    void reset_ghost_impl() {
        using CurrentVar = std::tuple_element_t<Index, VariablesTuple>;
        reset_ghost_for_variable<CurrentVar>();

        // Recurse to next variable if not last
        if constexpr (Index + 1 < std::tuple_size_v<VariablesTuple>) {
            reset_ghost_impl<VariablesTuple, Index + 1>();
        }
    }

};


// Implement at() after BlockView is fully defined
template<SystemModel SystemType, template<typename> class OperationsType>
typename Block<SystemType, OperationsType>::View Block<SystemType, OperationsType>::at(std::size_t rod_index) {
    std::size_t rod_start_col = system_start_index(rod_index);
    std::size_t rod_n_elems = rod_n_elems_[rod_index];
    return BlockView<SystemType>(data_, rod_index, rod_start_col, rod_n_elems);
}


} // namespace elasticapp
