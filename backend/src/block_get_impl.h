#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include "system.h"
#include "variable_offsets.h"
#include "traits.h"

namespace elasticapp {

// Uses C++17 fold expressions for a concise implementation
// Helper to check if a type is in a tuple
template<typename T, typename Tuple>
struct tuple_contains_impl;

template<typename T, typename... Ts>
struct tuple_contains_impl<T, std::tuple<Ts...>> {
    static constexpr bool value = (std::is_same_v<T, Ts> || ...);
};

template<typename T, typename Tuple>
constexpr bool tuple_contains_v = tuple_contains_impl<T, Tuple>::value;


// Helper function to get number of columns for a variable based on placement
template<typename VariableTag, SystemModel SystemType>
inline constexpr std::size_t get_variable_num_cols(std::size_t rod_n_nodes,
                                            std::size_t rod_n_elems,
                                            std::size_t rod_n_voronoi) {
    if constexpr (std::is_base_of_v<Placement::OnNode, VariableTag>) {
        return rod_n_nodes;
    } else if constexpr (std::is_base_of_v<Placement::OnElement, VariableTag>) {
        return rod_n_elems;
    } else if constexpr (std::is_base_of_v<Placement::OnVoronoi, VariableTag>) {
        return rod_n_voronoi;
    } else {
        static_assert(std::is_base_of_v<Placement::OnNode, VariableTag> ||
                     std::is_base_of_v<Placement::OnElement, VariableTag> ||
                     std::is_base_of_v<Placement::OnVoronoi, VariableTag>,
                     "VariableTag must have a Placement tag");
        return 0;  // Should never reach here
    }
}

// Generic implementation helper for Block::get() and BlockRodSystemView::get()
// Extracted to reduce code duplication between const and non-const versions
// This can be used by both Block and BlockRodSystemView classes
template<typename VariableTag, SystemModel SystemType, typename MatrixRef>
inline auto get_impl(MatrixRef&& matrix,
              std::size_t rod_n_nodes,
              std::size_t rod_n_elems,
              std::size_t rod_n_voronoi,
              std::size_t rod_start_col) {
    // Assert VariableTag is a valid member of tuple SystemType::Variables
    static_assert(tuple_contains_v<VariableTag, system_variables_t<SystemType>>,
        "VariableTag is not a valid member of tuple SystemType::Variables");
    // Compute row offset for this variable
    constexpr std::size_t row_offset = compute_variable_offset<VariableTag, SystemType>();
    constexpr std::size_t var_dimension = get_dimension_v<VariableTag>;

    // Determine number of columns based on placement
    std::size_t num_cols = get_variable_num_cols<VariableTag, SystemType>(
        rod_n_nodes, rod_n_elems, rod_n_voronoi);

    // Return a view into the specific rows and columns
    return get_block_slice(matrix, row_offset, var_dimension, rod_start_col, num_cols);
}

} // namespace elasticapp
