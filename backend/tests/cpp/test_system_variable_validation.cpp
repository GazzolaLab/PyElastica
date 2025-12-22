#include <catch2/catch_test_macros.hpp>
#include "../../src/system.h"
#include <type_traits>
#include <string_view>

namespace elasticapp {

// Valid variable examples
struct ValidVar1 : Placement::OnNode, DataType::Vector {
    static constexpr std::string_view name = "valid_var1";
};
struct ValidVar2 : Placement::OnElement, DataType::Scalar {
    static constexpr std::string_view name = "valid_var2";
};
struct ValidVar3 : Placement::OnVoronoi, DataType::Matrix {
    static constexpr std::string_view name = "valid_var3";
};

// Invalid variable examples (for testing compile-time checks)
struct InvalidVarNoPlacement : DataType::Vector {};  // Missing Placement
struct InvalidVarNoDataType : Placement::OnNode {};  // Missing DataType
struct InvalidVarNeither {};  // Missing both

// Dummy classes that use System as a mixin
template<ValidVariable... Vars>
class DummySystem : public System<Vars...> {
public:
    // Expose block_depth() method that uses System's static get_depth()
    std::size_t block_depth() const {
        return System<Vars...>::get_depth();
    }
};

} // namespace elasticapp

TEST_CASE("System - Variable Validation", "[system]") {
    SECTION("Valid variables compile successfully") {
        // These should compile without errors - using System as a mixin
        elasticapp::DummySystem<elasticapp::ValidVar1> system1;
        elasticapp::DummySystem<elasticapp::ValidVar1, elasticapp::ValidVar2> system2;
        elasticapp::DummySystem<elasticapp::ValidVar1, elasticapp::ValidVar2, elasticapp::ValidVar3> system3;

        REQUIRE(system1.block_depth() == 3);  // Vector = 3
        REQUIRE(system2.block_depth() == 4);  // Vector(3) + Scalar(1) = 4
        REQUIRE(system3.block_depth() == 13);  // Vector(3) + Scalar(1) + Matrix(9) = 13
    }

    SECTION("Variable validation concepts work correctly") {
        // Test HasPlacementTag concept
        static_assert(elasticapp::HasPlacementTag<elasticapp::ValidVar1>);
        static_assert(elasticapp::HasPlacementTag<elasticapp::ValidVar2>);
        static_assert(elasticapp::HasPlacementTag<elasticapp::ValidVar3>);
        static_assert(!elasticapp::HasPlacementTag<elasticapp::InvalidVarNoPlacement>);

        // Test HasDataTypeTag concept
        static_assert(elasticapp::HasDataTypeTag<elasticapp::ValidVar1>);
        static_assert(elasticapp::HasDataTypeTag<elasticapp::ValidVar2>);
        static_assert(elasticapp::HasDataTypeTag<elasticapp::ValidVar3>);
        static_assert(!elasticapp::HasDataTypeTag<elasticapp::InvalidVarNoDataType>);

        // Test ValidVariable concept
        static_assert(elasticapp::ValidVariable<elasticapp::ValidVar1>);
        static_assert(elasticapp::ValidVariable<elasticapp::ValidVar2>);
        static_assert(elasticapp::ValidVariable<elasticapp::ValidVar3>);
        static_assert(!elasticapp::ValidVariable<elasticapp::InvalidVarNoPlacement>);
        static_assert(!elasticapp::ValidVariable<elasticapp::InvalidVarNoDataType>);
        static_assert(!elasticapp::ValidVariable<elasticapp::InvalidVarNeither>);
    }
}

// Uncomment these to verify compile-time errors are caught:
// TEST_CASE("System - Invalid Variables (should not compile)", "[system]") {
//     // These should cause compile-time errors
//     // elasticapp::System<elasticapp::InvalidVarNoPlacement> invalid1(10);
//     // elasticapp::System<elasticapp::InvalidVarNoDataType> invalid2(10);
//     // elasticapp::System<elasticapp::InvalidVarNeither> invalid3(10);
//     // elasticapp::System<elasticapp::ValidVar1, elasticapp::InvalidVarNoPlacement> invalid4(10);
// }
