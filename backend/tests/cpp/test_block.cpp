#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "block.h"
#include "mock/mock_block_system.h"

using MockBlockSystem = elasticapp::mock::MockBlockSystem;

// Type aliases for convenience in tests
TEST_CASE("Block construction", "[block]") {
    SECTION("Can be constructed with list of element counts") {
        std::vector<std::size_t> n_elems_per_rod = {3};  // 3 elements -> 4 nodes (width)
        MockBlockSystem block(n_elems_per_rod);
        auto shape = block.shape();
        REQUIRE(shape.first == 17);  // MockSystem depth
        REQUIRE(shape.second == 4);   // 4 nodes + 0 ghost = 4 nodes
    }

    SECTION("Can be constructed with list of element counts (auto depth)") {
        std::vector<std::size_t> n_elems_per_rod = {3, 5, 2};
        MockBlockSystem block(n_elems_per_rod);

        // Width = sum of (n_elems + 1) for each rod + 2 ghost = (4 + 6 + 3) + 2 = 15
        // Depth = MockSystem block_depth = 17
        REQUIRE(block.width() == 15);
        REQUIRE(block.depth() == 17);
    }

    SECTION("Block stores correct starting indices for each rod") {
        std::vector<std::size_t> n_elems_per_rod = {3, 5, 2};
        MockBlockSystem block(n_elems_per_rod);

        // Rod 0: starts at 0, has 3+1=4 nodes
        REQUIRE(block.system_start_index(0) == 0);
        // Rod 1: starts at 4, has 5+1=6 nodes
        REQUIRE(block.system_start_index(1) == 4);
        // Rod 2: starts at 4+6=10, has 2+1=3 nodes
        REQUIRE(block.system_start_index(2) == 10);
    }

    SECTION("Block calculates width correctly for single rod") {
        std::vector<std::size_t> n_elems_per_rod = {7};
        MockBlockSystem block(n_elems_per_rod);

        // Single rod: 8 nodes + 0 ghost = 8 nodes
        REQUIRE(block.width() == 8);
        REQUIRE(block.system_start_index(0) == 0);
    }

    SECTION("Block calculates width correctly for empty list") {
        std::vector<std::size_t> n_elems_per_rod = {};
        MockBlockSystem block(n_elems_per_rod);

        REQUIRE(block.width() == 0);
        REQUIRE(block.n_systems() == 0);
    }

    SECTION("Block automatically computes depth from System") {
        std::vector<std::size_t> n_elems_per_rod = {3};
        MockBlockSystem mock_block(n_elems_per_rod);
        REQUIRE(mock_block.depth() == 17);  // MockSystem depth
    }
}

TEST_CASE("Block shape", "[block]") {
    SECTION("Returns correct shape") {
        std::vector<std::size_t> n_elems_per_rod = {6};  // 6 elements -> 7 nodes (width)
        MockBlockSystem block(n_elems_per_rod);
        auto shape = block.shape();
        REQUIRE(shape.first == 17);  // MockSystem depth
        REQUIRE(shape.second == 7);   // 7 nodes + 0 ghost = 7 nodes
    }
}

TEST_CASE("Block data access", "[block]") {
    SECTION("Data can be accessed and modified") {
        std::vector<std::size_t> n_elems_per_rod = {2};  // 2 elements -> 3 nodes (width)
        MockBlockSystem block(n_elems_per_rod);
        auto& data = block.data();

        // Modify data
        data(0, 0) = 1.5;
        data(0, 1) = 2.5;
        data(1, 0) = 3.5;

        // Verify modifications
        REQUIRE(data(0, 0) == 1.5);
        REQUIRE(data(0, 1) == 2.5);
        REQUIRE(data(1, 0) == 3.5);
    }
}

TEST_CASE("Block CRTP - access to System methods", "[block]") {
    SECTION("Block inherits System methods") {
        std::vector<std::size_t> n_elems_per_rod = {5};
        MockBlockSystem block(n_elems_per_rod);

        // Block inherits from MockSystem, so it has System methods
        REQUIRE(block.depth() == 17);  // MockSystem depth
    }
}

TEST_CASE("Block get() method", "[block]") {
    std::vector<std::size_t> n_elems_per_rod = {3, 5, 2};
    MockBlockSystem block(n_elems_per_rod);
    // Block has 3 rods: rod 0 has 3 elems (4 nodes), rod 1 has 5 elems (6 nodes), rod 2 has 2 elems (3 nodes)
    // Total width = 4 + 6 + 3 + (2 ghost) = 15

    SECTION("Block get() returns correct shapes for different variable types") {
        using namespace elasticapp::mock;
        auto& matrix = block.data();
        // Block width = 4 + 6 + 3 + 2 (ghost) = 15

        // Test MockVar1 (Node, Vector, offset=0, dimension=3)
        auto var1_view = block.get<MockVar1>();
        REQUIRE(var1_view.rows() == 3);
        REQUIRE(var1_view.cols() == 15);  // OnNode: full width

        // Test MockVar2 (Node, Scalar, offset=3, dimension=1)
        auto var2_view = block.get<MockVar2>();
        REQUIRE(var2_view.rows() == 1);
        REQUIRE(var2_view.cols() == 15);  // OnNode: full width

        // Test MockVar3 (Element, Vector, offset=4, dimension=3)
        auto var3_view = block.get<MockVar3>();
        REQUIRE(var3_view.rows() == 3);
        REQUIRE(var3_view.cols() == 14);  // OnElement: width - 1

        // Test MockVar4 (Element, Matrix, offset=7, dimension=9)
        auto var4_view = block.get<MockVar4>();
        REQUIRE(var4_view.rows() == 9);
        REQUIRE(var4_view.cols() == 14);  // OnElement: width - 1

        // Test MockVar5 (Voronoi, Scalar, offset=16, dimension=1)
        auto var5_view = block.get<MockVar5>();
        REQUIRE(var5_view.rows() == 1);
        REQUIRE(var5_view.cols() == 13);  // OnVoronoi: width - 2
    }

    SECTION("Block get() returns writable views that modify underlying data") {
        using namespace elasticapp::mock;
        auto& matrix = block.data();

        // Test MockVar1 (Node, Vector)
        auto var1_view = block.get<MockVar1>();

        // Modify through the view
        var1_view(0, 0) = 10.0;
        var1_view(1, 1) = 20.0;
        var1_view(2, 2) = 30.0;

        // Verify modifications are reflected in underlying matrix
        REQUIRE(matrix(0, 0) == 10.0);
        REQUIRE(matrix(1, 1) == 20.0);
        REQUIRE(matrix(2, 2) == 30.0);

        // Modify through matrix and verify view sees changes
        matrix(0, 3) = 40.0;
        REQUIRE(var1_view(0, 3) == 40.0);

        // Get another view - should see the same data
        auto var1_view2 = block.get<MockVar1>();
        REQUIRE(var1_view2(0, 0) == 10.0);
        REQUIRE(var1_view2(1, 1) == 20.0);
        REQUIRE(var1_view2(2, 2) == 30.0);

        // Modify through second view
        var1_view2(0, 4) = 50.0;
        REQUIRE(var1_view(0, 4) == 50.0);  // Should be reflected in first view
        REQUIRE(matrix(0, 4) == 50.0);     // And in underlying matrix
    }

    SECTION("Block get() works for different variable types") {
        using namespace elasticapp::mock;
        auto& matrix = block.data();

        // Test MockVar1 (OnNode, Vector)
        auto var1_view = block.get<MockVar1>();
        var1_view(0, 0) = 100.0;
        REQUIRE(matrix(0, 0) == 100.0);

        // Test MockVar2 (OnNode, Scalar)
        auto var2_view = block.get<MockVar2>();
        var2_view(0, 0) = 200.0;
        REQUIRE(matrix(3, 0) == 200.0);  // offset = 3

        // Test MockVar3 (OnElement, Vector)
        auto var3_view = block.get<MockVar3>();
        var3_view(0, 0) = 300.0;
        REQUIRE(matrix(4, 0) == 300.0);  // offset = 4

        // Test MockVar4 (OnElement, Matrix)
        auto var4_view = block.get<MockVar4>();
        var4_view(0, 0) = 400.0;
        REQUIRE(matrix(7, 0) == 400.0);  // offset = 7

        // Test MockVar5 (OnVoronoi, Scalar)
        auto var5_view = block.get<MockVar5>();
        var5_view(0, 0) = 500.0;
        REQUIRE(matrix(16, 0) == 500.0);  // offset = 16
    }

    SECTION("Block get() works across multiple rods") {
        using namespace elasticapp::mock;
        auto& matrix = block.data();

        // Rod 0: starts at column 0, has 4 nodes
        // Rod 1: starts at column 4, has 6 nodes
        // Rod 2: starts at column 10, has 3 nodes

        auto var1_view = block.get<MockVar1>();

        // Modify data for different rods
        var1_view(0, 0) = 1.0;   // Rod 0, first node
        var1_view(0, 3) = 2.0;   // Rod 0, last node
        var1_view(0, 4) = 3.0;   // Rod 1, first node
        var1_view(0, 9) = 4.0;   // Rod 1, last node
        var1_view(0, 10) = 5.0;  // Rod 2, first node
        var1_view(0, 12) = 6.0;  // Rod 2, last node

        // Verify all modifications
        REQUIRE(matrix(0, 0) == 1.0);
        REQUIRE(matrix(0, 3) == 2.0);
        REQUIRE(matrix(0, 4) == 3.0);
        REQUIRE(matrix(0, 9) == 4.0);
        REQUIRE(matrix(0, 10) == 5.0);
        REQUIRE(matrix(0, 12) == 6.0);
    }

    SECTION("Block get() validates VariableTag is in System") {
        using namespace elasticapp::mock;

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
        auto v1 = block.get<MockVar1>();
        auto v2 = block.get<MockVar2>();
        auto v3 = block.get<MockVar3>();
        auto v4 = block.get<MockVar4>();
        auto v5 = block.get<MockVar5>();

        // Use the variables to ensure they compile
        (void)v1;
        (void)v2;
        (void)v3;
        (void)v4;
        (void)v5;
    }

    SECTION("Block get() const version works") {
        using namespace elasticapp::mock;
        const MockBlockSystem& const_block = block;
        auto& matrix = block.data();

        // Set some values
        matrix(0, 0) = 42.0;
        matrix(3, 1) = 43.0;
        matrix(4, 2) = 44.0;

        // Get const views
        auto var1_view = const_block.get<MockVar1>();
        auto var2_view = const_block.get<MockVar2>();
        auto var3_view = const_block.get<MockVar3>();

        // Verify we can read values
        REQUIRE(var1_view(0, 0) == 42.0);
        REQUIRE(var2_view(0, 1) == 43.0);
        REQUIRE(var3_view(0, 2) == 44.0);

        // Verify shapes (with width adjustment)
        REQUIRE(var1_view.rows() == 3);
        REQUIRE(var1_view.cols() == 15);  // OnNode: full width
        REQUIRE(var2_view.rows() == 1);
        REQUIRE(var2_view.cols() == 15);  // OnNode: full width
        REQUIRE(var3_view.rows() == 3);
        REQUIRE(var3_view.cols() == 14);  // OnElement: width - 1
    }

    SECTION("Block get() width adjustment based on placement type") {
        using namespace elasticapp::mock;
        // Block width = 4 + 6 + 3 + 2 (ghost) = 15

        // Test OnNode variables get full width
        auto var1_view = block.get<MockVar1>();  // OnNode
        auto var2_view = block.get<MockVar2>();  // OnNode
        REQUIRE(var1_view.cols() == 15);
        REQUIRE(var2_view.cols() == 15);
        REQUIRE(var1_view.cols() == block.width());
        REQUIRE(var2_view.cols() == block.width());

        // Test OnElement variables get width - 1
        auto var3_view = block.get<MockVar3>();  // OnElement
        auto var4_view = block.get<MockVar4>();  // OnElement
        REQUIRE(var3_view.cols() == 14);  // 15 - 1
        REQUIRE(var4_view.cols() == 14);  // 15 - 1
        REQUIRE(var3_view.cols() == block.width() - 1);
        REQUIRE(var4_view.cols() == block.width() - 1);

        // Test OnVoronoi variables get width - 2
        auto var5_view = block.get<MockVar5>();  // OnVoronoi
        REQUIRE(var5_view.cols() == 13);  // 15 - 2
        REQUIRE(var5_view.cols() == block.width() - 2);
    }

    SECTION("Block get() width adjustment with single rod") {
        using namespace elasticapp::mock;
        std::vector<std::size_t> single_rod = {5};  // 5 elems -> 6 nodes, width = 6
        MockBlockSystem single_block(single_rod);

        // OnNode: full width
        auto var1_view = single_block.get<MockVar1>();
        REQUIRE(var1_view.cols() == 6);

        // OnElement: width - 1
        auto var3_view = single_block.get<MockVar3>();
        REQUIRE(var3_view.cols() == 5);  // 6 - 1

        // OnVoronoi: width - 2
        auto var5_view = single_block.get<MockVar5>();
        REQUIRE(var5_view.cols() == 4);  // 6 - 2
    }

    SECTION("Block get() width adjustment edge cases") {
        using namespace elasticapp::mock;
        // Test with very small width
        std::vector<std::size_t> small_rod = {1};  // 1 elem -> 2 nodes, width = 2
        MockBlockSystem small_block(small_rod);

        // OnNode: full width
        auto var1_view = small_block.get<MockVar1>();
        REQUIRE(var1_view.cols() == 2);

        // OnElement: width - 1 (should be 1, not negative)
        auto var3_view = small_block.get<MockVar3>();
        REQUIRE(var3_view.cols() == 1);  // 2 - 1

        // OnVoronoi: width - 2 (should be 0, not negative)
        auto var5_view = small_block.get<MockVar5>();
        REQUIRE(var5_view.cols() == 0);  // 2 - 2 = 0

        // Test with empty block
        std::vector<std::size_t> empty_rod = {};
        MockBlockSystem empty_block(empty_rod);

        auto var1_empty = empty_block.get<MockVar1>();
        REQUIRE(var1_empty.cols() == 0);

        auto var3_empty = empty_block.get<MockVar3>();
        REQUIRE(var3_empty.cols() == 0);  // 0 - 1 clamped to 0

        auto var5_empty = empty_block.get<MockVar5>();
        REQUIRE(var5_empty.cols() == 0);  // 0 - 2 clamped to 0
    }
}
