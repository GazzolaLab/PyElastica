#pragma once

#include "system.h"
#include "block.h"
#include "operations.h"
#include "mock_block_system.h"  // For MockSystem and variable definitions

namespace elasticapp::mock {

// Mock operations class that implements addition of two variables
// Adds MockVar1 (Vector) and MockVar2 (Scalar, broadcasted) and stores in MockVar1
template<typename Derived>
class AdditionOperations {
public:
    // Operation: result = var1 + var2 (where var2 is broadcasted if scalar)
    // This adds MockVar1 (Vector) and MockVar2 (Scalar broadcasted to Vector)
    // and stores the result back in MockVar1
    void add_variables() {
        auto& block = static_cast<Derived&>(*this);

        // Get views of the variables
        auto var1 = block.template get<MockVar1>();  // Vector on Node
        auto var2 = block.template get<MockVar2>();  // Scalar on Node

        // var1 is a matrix view: rows = 3 (vector dimension), cols = width
        // var2 is a matrix view: rows = 1 (scalar dimension), cols = width

        // Add var2 (scalar) to each component of var1 (vector)
        // var1 has 3 rows, var2 has 1 row
        for (Eigen::Index col = 0; col < var1.cols(); ++col) {
            double scalar_value = var2(0, col);
            for (Eigen::Index row = 0; row < var1.rows(); ++row) {
                var1(row, col) = var1(row, col) + scalar_value;
            }
        }
    }

    // Alternative operation: add two vectors (MockVar1 + MockVar3)
    // Note: MockVar1 is on Node, MockVar3 is on Element, so we need to handle different sizes
    // For simplicity, we'll add MockVar1 to a scaled version of itself
    void add_vector_to_itself() {
        auto& block = static_cast<Derived&>(*this);
        auto var1 = block.template get<MockVar1>();  // Vector on Node

        // Add var1 to itself (multiply by 2)
        var1 = var1 + var1;
    }

protected:
    AdditionOperations() = default;
    ~AdditionOperations() = default;

    AdditionOperations(const AdditionOperations&) = default;
    AdditionOperations(AdditionOperations&&) = default;
    AdditionOperations& operator=(const AdditionOperations&) = default;
    AdditionOperations& operator=(AdditionOperations&&) = default;
};

// MockBlockSystem with AdditionOperations
using MockBlockSystemWithOperations = ::elasticapp::Block<MockSystem, AdditionOperations>;

} // namespace elasticapp::mock
