#pragma once

#include <cstddef>
#include <type_traits>
#include <concepts>
#include <tuple>
#include "traits.h"

namespace elasticapp {

//
// System<Ts...> is a minimal header-only class representing a collection of physical variables
// and their properties for a mechanical or simulation system. It is used solely for compile-
// time type information and does not allocate memory or hold state. Its purpose is to define,
// via template arguments, which variables are present in a system and to specify their placement
// (e.g., on nodes, elements, voronoi) and data type (e.g., scalar, vector, matrix) using tags.
// System enables static reflection and type-driven logic for Block, BlockView, and other components.
//
// Example:
//   using MySystem = System<Displacement, Force, BendingMoment>;
//
//   - Displacement, Force, BendingMoment are structs inheriting correct placement and data type tags.
//   - MySystem::Variables is a type list of the variables, usable in metaprogramming.
//   - System does not itself allocate memory or store real data.
//

// ********************************************************
// Placement tags
// ********************************************************

// Placement tags - indicate where variables are stored
namespace Placement {
    struct OnNode {};
    struct OnElement {};
    struct OnVoronoi {};
} // namespace Placement

template<typename T>
concept HasPlacementTag =
    std::is_base_of_v<Placement::OnNode, T> ||
    std::is_base_of_v<Placement::OnElement, T> ||
    std::is_base_of_v<Placement::OnVoronoi, T>;

// ********************************************************
// DataType tags
// ********************************************************

namespace DataType {
    // Dimension tags - indicate the size of variables
    struct Scalar {
        static constexpr std::size_t dimension = 1;
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // Using inline static (C++17) to define in header
        inline static MatrixType ghost_value = MatrixType::Zero(1, 1);
    };
    struct Vector {
        static constexpr std::size_t dimension = 3;
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // Using inline static (C++17) to define in header
        inline static MatrixType ghost_value = MatrixType::Zero(3, 1);
    };
    struct Matrix {
        static constexpr std::size_t dimension = 9;
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // Using inline static (C++17) to define in header
        inline static MatrixType ghost_value = MatrixType::Zero(9, 1);
    };
} // namespace DataType

template<typename T>
concept HasDataTypeTag =
    std::is_base_of_v<DataType::Scalar, T> ||
    std::is_base_of_v<DataType::Vector, T> ||
    std::is_base_of_v<DataType::Matrix, T>;

template<typename T>
concept ValidVariable =
    HasPlacementTag<T> &&
    HasDataTypeTag<T> &&
    requires {
        { T::name } -> std::convertible_to<std::string_view>;
        { T::ghost_value };
    };

// Helper to extract dimension from a variable type
template<typename T>
constexpr std::size_t get_dimension_v =
    std::is_base_of_v<DataType::Scalar, T> ? DataType::Scalar::dimension :
    std::is_base_of_v<DataType::Vector, T> ? DataType::Vector::dimension :
    std::is_base_of_v<DataType::Matrix, T> ? DataType::Matrix::dimension : 0;

// Helper to sum dimensions across a parameter pack
template<typename... Variables>
constexpr std::size_t sum_dimensions_v = (0 + ... + get_dimension_v<Variables>);

// ********************************************************
// System class
// ********************************************************

// System class - collection of Variables
// Template parameter pack for variable types
// Each Variable must derive from one Placement tag and one DataType tag
template<ValidVariable... Vars>
class System {
public:
    // Expose the variable types for template metaprogramming
    using Variables = std::tuple<Vars...>;

    static constexpr std::size_t get_depth() {
        return sum_dimensions_v<Vars...>;
    }

};

// Helper to extract variable types from a System type
template<typename SystemType>
struct system_variables;

template<ValidVariable... Vars>
struct system_variables<System<Vars...>> {
    using type = std::tuple<Vars...>;
};

template<typename SystemType>
using system_variables_t = typename system_variables<SystemType>::type;

// Concept to check if a type is a System<ValidVariable...>
// A type is a SystemModel if:
// 1. It is not instantiable on its own.
// 2. It has a Variables type alias (std::tuple<...>)
// 3. It has a static get_depth() method
// 4. system_variables<T> is specialized (i.e., T is System<...>)
template<typename SystemType>
concept SystemModel = requires {
    typename SystemType::Variables;  // Must have Variables type alias
    { SystemType::get_depth() } -> std::convertible_to<std::size_t>;  // Must have static get_depth()
    typename system_variables<SystemType>::type;  // system_variables<T> must be specialized
    requires std::is_same_v<typename system_variables<SystemType>::type, typename SystemType::Variables>;
};

} // namespace elasticapp
