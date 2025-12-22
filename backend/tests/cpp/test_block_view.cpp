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

    // SECTION("Can create BlockView for valid rod index") {
    //     auto& view = block.at(0);
    //     REQUIRE(view.var1.depth_offset() == 0);
    //     REQUIRE(view.var2.depth_offset() == 3);
    //     REQUIRE(view.var3.depth_offset() == 4);
    //     REQUIRE(view.var4.depth_offset() == 7);
    //     REQUIRE(view.var5.depth_offset() == 16);
    // }

    // SECTION("Variables have correct dimensions") {
    //     auto& view = block.at(0);
    //     REQUIRE(view.var1.var_dimension() == 3);  // Vector
    //     REQUIRE(view.var2.var_dimension() == 1);  // Scalar
    //     REQUIRE(view.var3.var_dimension() == 3);  // Vector
    //     REQUIRE(view.var4.var_dimension() == 9);  // Matrix
    //     REQUIRE(view.var5.var_dimension() == 1);  // Scalar
    // }

    // SECTION("Variables have correct rod sizes based on placement") {
    //     auto& view = block.at(0);
    //     // Rod 0 has 3 elements -> 4 nodes, 3 elements, 2 voronoi
    //     REQUIRE(view.var1.rod_size() == 4);  // Node variable
    //     REQUIRE(view.var2.rod_size() == 4);  // Node variable
    //     REQUIRE(view.var3.rod_size() == 3);  // Element variable
    //     REQUIRE(view.var4.rod_size() == 3);  // Element variable
    //     REQUIRE(view.var5.rod_size() == 2);  // Voronoi variable
    // }

    // SECTION("Variables have correct rod sizes for different rod") {
    //     auto& view = block.at(1);
    //     // Rod 1 has 5 elements -> 6 nodes, 5 elements, 4 voronoi
    //     REQUIRE(view.var1.rod_size() == 6);  // Node variable
    //     REQUIRE(view.var2.rod_size() == 6);  // Node variable
    //     REQUIRE(view.var3.rod_size() == 5);  // Element variable
    //     REQUIRE(view.var4.rod_size() == 5);  // Element variable
    //     REQUIRE(view.var5.rod_size() == 4);  // Voronoi variable
    // }

    // SECTION("Cannot call at() with invalid rod index") {
    //     REQUIRE_THROWS_AS(block.at(3), std::out_of_range);
    //     REQUIRE_THROWS_WITH(block.at(3), "Rod index out of range");
    // }
}

// TEST_CASE("BlockView - Read/Write Access", "[block_view]") {
//     std::vector<std::size_t> n_elems_per_rod = {3};
//     MockBlockSystem block(n_elems_per_rod);

//     // Initialize block data with known values
//     auto& matrix = block.data();
//     for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
//         for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
//             matrix(i, j) = static_cast<double>(i * 100 + j);
//         }
//     }

//     SECTION("Can read values from var1 (Node, Vector)") {
//         auto& view = block.at(0);

//         // var1 is at depth offset 0, rod starts at column 0
//         // var1 is Vector (3D), so we have 3 rows starting at depth 0
//         // Rod 0 has 4 nodes (columns 0-3)
//         // For column-major: matrix(depth, col)
//         // var1[0, 0] should be matrix(0, 0) = 0
//         // var1[1, 0] should be matrix(1, 0) = 100
//         // var1[2, 0] should be matrix(2, 0) = 200

//         // Access through block data directly to verify
//         REQUIRE(std::abs(matrix(0, 0) - 0.0) < 1e-10);
//         REQUIRE(std::abs(matrix(1, 0) - 100.0) < 1e-10);
//         REQUIRE(std::abs(matrix(2, 0) - 200.0) < 1e-10);

//         // Verify depth offset and rod start column
//         REQUIRE(view.var1.depth_offset() == 0);
//         REQUIRE(view.var1.rod_start_col() == 0);
//         REQUIRE(view.var1.rod_size() == 4);
//     }

//     SECTION("Can write values to var1 (Node, Vector)") {
//         auto& view = block.at(0);

//         // Write through block data
//         matrix(0, 0) = 42.0;
//         matrix(1, 0) = 43.0;
//         matrix(2, 0) = 44.0;

//         // Verify values were written
//         REQUIRE(std::abs(matrix(0, 0) - 42.0) < 1e-10);
//         REQUIRE(std::abs(matrix(1, 0) - 43.0) < 1e-10);
//         REQUIRE(std::abs(matrix(2, 0) - 44.0) < 1e-10);
//     }

//     SECTION("Can read values from var2 (Node, Scalar)") {
//         auto& view = block.at(0);

//         // var2 is at depth offset 3, rod starts at column 0
//         // var2 is Scalar (1D), so we have 1 row at depth 3
//         // For column-major: matrix(3, 0) should be 300
//         REQUIRE(std::abs(matrix(3, 0) - 300.0) < 1e-10);

//         REQUIRE(view.var2.depth_offset() == 3);
//         REQUIRE(view.var2.rod_start_col() == 0);
//         REQUIRE(view.var2.rod_size() == 4);
//     }

//     SECTION("Can write values to var2 (Node, Scalar)") {
//         auto& view = block.at(0);

//         matrix(3, 0) = 99.0;
//         REQUIRE(std::abs(matrix(3, 0) - 99.0) < 1e-10);
//     }

//     SECTION("Can read values from var3 (Element, Vector)") {
//         auto& view = block.at(0);

//         // var3 is at depth offset 4, rod starts at column 0
//         // var3 is Vector (3D), so we have 3 rows starting at depth 4
//         // Rod 0 has 3 elements (columns 0-2)
//         REQUIRE(std::abs(matrix(4, 0) - 400.0) < 1e-10);
//         REQUIRE(std::abs(matrix(5, 0) - 500.0) < 1e-10);
//         REQUIRE(std::abs(matrix(6, 0) - 600.0) < 1e-10);

//         REQUIRE(view.var3.depth_offset() == 4);
//         REQUIRE(view.var3.rod_start_col() == 0);
//         REQUIRE(view.var3.rod_size() == 3);
//     }

//     SECTION("Can write values to var3 (Element, Vector)") {
//         auto& view = block.at(0);

//         matrix(4, 0) = 100.0;
//         matrix(5, 0) = 101.0;
//         matrix(6, 0) = 102.0;

//         REQUIRE(std::abs(matrix(4, 0) - 100.0) < 1e-10);
//         REQUIRE(std::abs(matrix(5, 0) - 101.0) < 1e-10);
//         REQUIRE(std::abs(matrix(6, 0) - 102.0) < 1e-10);
//     }

//     SECTION("Can read values from var4 (Element, Matrix)") {
//         auto& view = block.at(0);

//         // var4 is at depth offset 7, rod starts at column 0
//         // var4 is Matrix (9D), so we have 9 rows starting at depth 7
//         REQUIRE(std::abs(matrix(7, 0) - 700.0) < 1e-10);
//         REQUIRE(std::abs(matrix(15, 0) - 1500.0) < 1e-10);

//         REQUIRE(view.var4.depth_offset() == 7);
//         REQUIRE(view.var4.rod_start_col() == 0);
//         REQUIRE(view.var4.rod_size() == 3);
//     }

//     SECTION("Can write values to var4 (Element, Matrix)") {
//         auto& view = block.at(0);

//         matrix(7, 0) = 200.0;
//         matrix(15, 0) = 201.0;

//         REQUIRE(std::abs(matrix(7, 0) - 200.0) < 1e-10);
//         REQUIRE(std::abs(matrix(15, 0) - 201.0) < 1e-10);
//     }

//     SECTION("Can read values from var5 (Voronoi, Scalar)") {
//         auto& view = block.at(0);

//         // var5 is at depth offset 16, rod starts at column 0
//         // var5 is Scalar (1D), so we have 1 row at depth 16
//         // Rod 0 has 2 voronoi (columns 0-1)
//         REQUIRE(std::abs(matrix(16, 0) - 1600.0) < 1e-10);

//         REQUIRE(view.var5.depth_offset() == 16);
//         REQUIRE(view.var5.rod_start_col() == 0);
//         REQUIRE(view.var5.rod_size() == 2);
//     }

//     SECTION("Can write values to var5 (Voronoi, Scalar)") {
//         MockBlockSystemView view = MockBlockSystemView(block, 0, n_elems_per_rod);

//         matrix(16, 0) = 300.0;
//         REQUIRE(std::abs(matrix(16, 0) - 300.0) < 1e-10);
//     }

//     SECTION("Multiple rods - correct column offsets") {
//         std::vector<std::size_t> n_elems_per_rod_multi = {2, 3};
//         MockBlockSystem block_multi(n_elems_per_rod_multi);
//         auto& matrix_multi = block_multi.data();

//         // Initialize with known pattern
//         for (Eigen::Index i = 0; i < matrix_multi.rows(); ++i) {
//             for (Eigen::Index j = 0; j < matrix_multi.cols(); ++j) {
//                 matrix_multi(i, j) = static_cast<double>(i * 1000 + j);
//             }
//         }

//         // Rod 0: 2 elements -> 3 nodes, starts at column 0
//         MockBlockSystemView view0 = MockBlockSystemView(block_multi, 0, n_elems_per_rod_multi);
//         REQUIRE(view0.var1.rod_start_col() == 0);
//         REQUIRE(view0.var1.rod_size() == 3);
//         REQUIRE(std::abs(matrix_multi(0, 0) - 0.0) < 1e-10);

//         // Rod 1: 3 elements -> 4 nodes, starts at column 3
//         MockBlockSystemView view1 = MockBlockSystemView(block_multi, 1, n_elems_per_rod_multi);
//         REQUIRE(view1.var1.rod_start_col() == 3);
//         REQUIRE(view1.var1.rod_size() == 4);
//         REQUIRE(std::abs(matrix_multi(0, 3) - 3.0) < 1e-10);

//         // Write to rod 1, verify it doesn't affect rod 0
//         matrix_multi(0, 3) = 999.0;
//         REQUIRE(std::abs(matrix_multi(0, 3) - 999.0) < 1e-10);
//         REQUIRE(std::abs(matrix_multi(0, 0) - 0.0) < 1e-10);  // Rod 0 unchanged
//     }
// }
