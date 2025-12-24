#include <catch2/catch_test_macros.hpp>
#include "../../src/block.h"
#include "mock/mock_block_system.h"
#include <vector>
#include <cmath>

using MockBlockSystem = elasticapp::mock::MockBlockSystem;
using MockBlockSystemView = typename MockBlockSystem::View;

TEST_CASE("BlockView - Variable Retrieval", "[block_view]") {
    std::vector<std::size_t> n_elems_per_rod = {3, 5, 2};
    MockBlockSystem block(n_elems_per_rod);

    SECTION("BlockView has correct shape") {
        auto&& view1 = block.at(0);
        auto&& view2 = block.at(1);
        REQUIRE(view1.shape() == std::pair(17UL, 4UL));
        REQUIRE(view2.shape() == std::pair(17UL, 6UL));
        STATIC_REQUIRE(std::is_same_v<decltype(view1), MockBlockSystemView&&>);
        STATIC_REQUIRE(std::is_same_v<decltype(view2), MockBlockSystemView&&>);
    }

    SECTION("BlockView data read/write access") {
        for(size_t rod_index = 0; rod_index < block.n_systems(); ++rod_index) {
            auto& matrix = block.data();
            auto&& view = block.at(rod_index);
            auto&& view_matrix = view.data();
            size_t start_index = block.system_start_index(rod_index);

            // Read access test
            // start_index is a column index, so we access matrix(row, start_index + col)
            matrix(0, start_index) = 1.2;
            matrix(1, start_index + 1) = 2.3;
            matrix(2, start_index + 2) = 3.4;
            REQUIRE(matrix(0, start_index) == view_matrix(0, 0));
            REQUIRE(matrix(1, start_index + 1) == view_matrix(1, 1));
            REQUIRE(matrix(2, start_index + 2) == view_matrix(2, 2));

            // Write access test
            view_matrix(0, 0) = 4.2;
            view_matrix(1, 1) = 5.3;
            view_matrix(2, 2) = 6.4;
            REQUIRE(matrix(0, start_index) == 4.2);
            REQUIRE(matrix(1, start_index + 1) == 5.3);
            REQUIRE(matrix(2, start_index + 2) == 6.4);
        }
    }

    SECTION("BlockView get() method for specific variables") {
        using namespace elasticapp::mock;
        auto&& view = block.at(0);
        auto& matrix = block.data();

        // Test MockVar1 (Node, Vector, offset=0, dimension=3)
        auto var1_view = view.get<MockVar1>();
        REQUIRE(var1_view.rows() == 3);
        REQUIRE(var1_view.cols() == 4);  // rod_n_nodes_ = 3 + 1 = 4

        // Verify it's a view (no copy) - modify through view and check original
        var1_view(0, 0) = 10.0;
        var1_view(1, 1) = 20.0;
        var1_view(2, 2) = 30.0;
        size_t start_index = block.system_start_index(0);
        REQUIRE(matrix(0, start_index) == 10.0);
        REQUIRE(matrix(1, start_index + 1) == 20.0);
        REQUIRE(matrix(2, start_index + 2) == 30.0);

        // Test MockVar2 (Node, Scalar, offset=3, dimension=1)
        auto var2_view = view.get<MockVar2>();
        REQUIRE(var2_view.rows() == 1);
        REQUIRE(var2_view.cols() == 4);  // rod_n_nodes_ = 4
        var2_view(0, 0) = 100.0;
        REQUIRE(matrix(3, start_index) == 100.0);

        // Test MockVar3 (Element, Vector, offset=4, dimension=3)
        auto var3_view = view.get<MockVar3>();
        REQUIRE(var3_view.rows() == 3);
        REQUIRE(var3_view.cols() == 3);  // rod_n_elems_ = 3
        var3_view(0, 0) = 200.0;
        REQUIRE(matrix(4, start_index) == 200.0);

        // Test MockVar4 (Element, Matrix, offset=7, dimension=9)
        auto var4_view = view.get<MockVar4>();
        REQUIRE(var4_view.rows() == 9);
        REQUIRE(var4_view.cols() == 3);  // rod_n_elems_ = 3
        var4_view(0, 0) = 300.0;
        REQUIRE(matrix(7, start_index) == 300.0);

        // Test MockVar5 (Voronoi, Scalar, offset=16, dimension=1)
        auto var5_view = view.get<MockVar5>();
        REQUIRE(var5_view.rows() == 1);
        REQUIRE(var5_view.cols() == 2);  // rod_n_voronoi_ = 3 - 1 = 2
        var5_view(0, 0) = 400.0;
        REQUIRE(matrix(16, start_index) == 400.0);
    }

    SECTION("BlockView get() validates VariableTag is in System") {
        using namespace elasticapp::mock;
        auto&& view = block.at(0);


        // All valid variables from MockSystem should compile and work
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar1, typename MockSystem::Variables>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar2, typename MockSystem::Variables>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar3, typename MockSystem::Variables>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar4, typename MockSystem::Variables>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar5, typename MockSystem::Variables>);

        // Test that non-MockVar is not in MockSystem::Variables
        class NonMockVar : elasticapp::Placement::OnNode, elasticapp::DataType::Vector {};
        STATIC_REQUIRE(!elasticapp::tuple_contains_v<NonMockVar, typename MockSystem::Variables>);

        // Verify get() works for all valid variables (compile-time check)
        auto v1 = view.get<MockVar1>();
        auto v2 = view.get<MockVar2>();
        auto v3 = view.get<MockVar3>();
        auto v4 = view.get<MockVar4>();
        auto v5 = view.get<MockVar5>();

        // Use the variables to ensure they compile
        (void)v1;
        (void)v2;
        (void)v3;
        (void)v4;
        (void)v5;
    }

    SECTION("tuple_contains_v trait works correctly") {
        using namespace elasticapp::mock;
        using MockVars = typename MockSystem::Variables;

        // Test that tuple_contains_v correctly identifies members
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar1, MockVars>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar2, MockVars>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar3, MockVars>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar4, MockVars>);
        STATIC_REQUIRE(elasticapp::tuple_contains_v<MockVar5, MockVars>);

        // Test with invalid types (should be false)
        struct InvalidVar1 : elasticapp::Placement::OnNode, elasticapp::DataType::Vector {};
        struct InvalidVar2 : elasticapp::Placement::OnElement, elasticapp::DataType::Scalar {};

        STATIC_REQUIRE(!elasticapp::tuple_contains_v<InvalidVar1, MockVars>);
        STATIC_REQUIRE(!elasticapp::tuple_contains_v<InvalidVar2, MockVars>);
    }
}
