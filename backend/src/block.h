#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
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

    // Constructor: takes list of element counts per rod
    explicit Block(const std::vector<std::size_t>& n_elems_per_rod) {
        compute_width_and_indices(n_elems_per_rod);
        depth_ = SystemType::get_depth();
        data_ = MatrixType(static_cast<Eigen::Index>(depth_), static_cast<Eigen::Index>(width_));
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
            width_ += n_elems + 1;
        }

        // Add ghost nodes between rods (only if there are at least 2 rods)
        // For n rods, there are (n-1) ghost nodes between them
        if (n_elems_per_rod.size() > 1) {
            width_ += n_elems_per_rod.size() - 1;
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
