#pragma once

#include "cosserat_rod_system.h"
#include "system.h"
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <tuple>

namespace elasticapp {

// Generic helpers for computing variable offsets in any System

// Helper to find the index of a type in a parameter pack
template<typename Target, std::size_t CurrentIndex, typename First, typename... Rest>
struct find_index_impl {
    using type = std::conditional_t<
        std::is_same_v<Target, First>,
        std::integral_constant<std::size_t, CurrentIndex>,
        find_index_impl<Target, CurrentIndex + 1, Rest...>
    >;
    static constexpr std::size_t value = type::value;
};

// Base case: when Target matches First (Rest may be empty)
template<typename Target, std::size_t CurrentIndex>
struct find_index_impl<Target, CurrentIndex, Target> {
    static constexpr std::size_t value = CurrentIndex;
};

// Helper to sum dimensions of variables up to (but not including) a given index
template<std::size_t Index, typename... Variables>
struct sum_dimensions_up_to_index_impl;

template<std::size_t Index, typename First, typename... Rest>
struct sum_dimensions_up_to_index_impl<Index, First, Rest...> {
    static constexpr std::size_t value =
        (Index == 0) ? 0 :
        (get_dimension_v<First> + sum_dimensions_up_to_index_impl<Index - 1, Rest...>::value);
};

template<std::size_t Index>
struct sum_dimensions_up_to_index_impl<Index> {
    static constexpr std::size_t value = 0;
};

// Helper to compute offset for a variable in a System
// This finds the variable's position and sums dimensions of all variables before it
template<typename Variable, typename... SystemVariables>
struct compute_variable_offset_impl {
    static constexpr std::size_t index = find_index_impl<Variable, 0, SystemVariables...>::value;
    static constexpr std::size_t value = sum_dimensions_up_to_index_impl<index, SystemVariables...>::value;
};

// Helper to expand a tuple into a parameter pack for compute_variable_offset_impl
template<typename Variable, typename VariablesTuple>
struct compute_variable_offset_from_system_impl;

template<typename Variable, typename... SystemVariables>
struct compute_variable_offset_from_system_impl<Variable, std::tuple<SystemVariables...>> {
    static constexpr std::size_t value = compute_variable_offset_impl<Variable, SystemVariables...>::value;
};

// Generic function to compute variable offset for any System type
// This automatically extracts variables from SystemType::Variables and computes the offset
// SystemType must be a System<...> type
template<typename Variable, SystemModel SystemType>
constexpr std::size_t compute_variable_offset() {
    // Extract variables directly from SystemType's Variables type alias
    using VariablesTuple = typename SystemType::Variables;
    return compute_variable_offset_from_system_impl<Variable, VariablesTuple>::value;
}

} // namespace elasticapp
