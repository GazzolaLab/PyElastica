#pragma once

#include "system.h"
#include "block.h"

namespace elasticapp::mock {

// Mock system for testing BlockView with arbitrary variables
// This demonstrates that BlockView can work with any System type

// Simple test variables
struct MockVar1 :
    ::elasticapp::Placement::OnNode,
    ::elasticapp::DataType::Vector {
        static constexpr std::string_view name = "mock_var1";
    };
struct MockVar2 :
    ::elasticapp::Placement::OnNode,
    ::elasticapp::DataType::Scalar {
        static constexpr std::string_view name = "mock_var2";
    };
struct MockVar3 :
    ::elasticapp::Placement::OnElement,
    ::elasticapp::DataType::Vector {
        static constexpr std::string_view name = "mock_var3";
    };
struct MockVar4 :
    ::elasticapp::Placement::OnElement,
    ::elasticapp::DataType::Matrix {
        static constexpr std::string_view name = "mock_var4";
    };
struct MockVar5 :
    ::elasticapp::Placement::OnVoronoi,
    ::elasticapp::DataType::Scalar {
        static constexpr std::string_view name = "mock_var5";
    };

// MockSystem with a small set of variables for testing
using MockSystem = ::elasticapp::System<
    MockVar1,  // Node, Vector (3)
    MockVar2,  // Node, Scalar (1)
    MockVar3,  // Element, Vector (3)
    MockVar4,  // Element, Matrix (9)
    MockVar5   // Voronoi, Scalar (1)
>;
// Total depth: 3 + 1 + 3 + 9 + 1 = 17

using MockBlockSystem = ::elasticapp::Block<MockSystem>;
using MockBlockSystemView = ::elasticapp::BlockView<MockSystem>;

} // namespace elasticapp::mock
