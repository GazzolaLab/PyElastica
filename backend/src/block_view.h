#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include "system.h"
#include "block_get_impl.h"
#include "traits.h"

namespace elasticapp {

// BlockView provides access to variables for a specific rod
// This is templated on SystemType and uses template metaprogramming to expose variables
template<SystemModel SystemType>
class BlockView {
public:
    // using Variables = typename SystemType::Variables; // tuple of variables
    constexpr static std::size_t depth = SystemType::get_depth();

    BlockView(MatrixType& data, std::size_t rod_index,
              std::size_t rod_start_col, std::size_t rod_n_elems);

    std::pair<std::size_t, std::size_t> shape() const {
        return std::make_pair(depth, rod_n_nodes_);
    }

    // Return a view into the submatrix (columns for this rod)
    // Uses traits helper function to encapsulate Eigen-specific operations
    auto data() {
        return get_column_slice(data_, rod_start_col_, rod_n_nodes_);
    }

    auto data() const {
        return get_column_slice(data_, rod_start_col_, rod_n_nodes_);
    }

    // Get a view for a specific variable
    // Returns a view into the variable's data (rows) and this rod's columns
    // No data is copied - returns a reference to the same matrix
    template<typename VariableTag>
    auto get() {
        return get_impl<VariableTag, SystemType>(
            data_, rod_n_nodes_, rod_n_elems_, rod_n_voronoi_, rod_start_col_);
    }

    template<typename VariableTag>
    auto get() const {
        return get_impl<VariableTag, SystemType>(
            data_, rod_n_nodes_, rod_n_elems_, rod_n_voronoi_, rod_start_col_);
    }

protected:
    MatrixType& data_;
    std::size_t rod_index_;
    std::size_t rod_n_elems_;
    std::size_t rod_n_nodes_;
    std::size_t rod_n_voronoi_;
    std::size_t rod_start_col_;
};

// Note: block.h is included after BlockView definition to break circular dependency
// The BlockView constructor implementation is in block.h after Block is fully defined
template<SystemModel SystemType>
BlockView<SystemType>::BlockView(MatrixType& data, std::size_t rod_index,
                                 std::size_t rod_start_col, std::size_t rod_n_elems)
    : data_(data), rod_index_(rod_index),
      rod_n_elems_(rod_n_elems),
      rod_n_nodes_(rod_n_elems + 1),
      rod_n_voronoi_((rod_n_elems > 0) ? rod_n_elems - 1 : 0),
      rod_start_col_(rod_start_col)
{
}

} // namespace elasticapp
