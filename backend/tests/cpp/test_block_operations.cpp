#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <cmath>
#include "block.h"
#include "mock/mock_block_system_with_operation.h"

using MockBlockSystemWithOps = elasticapp::mock::MockBlockSystemWithOperations;
using namespace elasticapp::mock;
using Catch::Approx;

TEST_CASE("Block with Operations - Addition", "[block][operations]") {
    std::vector<std::size_t> n_elems_per_rod = {3, 5};
    MockBlockSystemWithOps block(n_elems_per_rod);

    SECTION("Operations can be called on block") {
        // Verify the block has the operations method
        REQUIRE_NOTHROW(block.add_variables());
    }

    SECTION("Add variables operation - Vector + Scalar") {
        // Get variable views
        auto var1 = block.template get<MockVar1>();  // Vector (3 rows)
        auto var2 = block.template get<MockVar2>();  // Scalar (1 row)

        // Initialize test data
        // Set var1 to some values
        var1(0, 0) = 1.0;  // First component, first column
        var1(1, 0) = 2.0;  // Second component, first column
        var1(2, 0) = 3.0;  // Third component, first column

        var1(0, 1) = 4.0;
        var1(1, 1) = 5.0;
        var1(2, 1) = 6.0;

        // Set var2 (scalar) values
        var2(0, 0) = 10.0;  // First column
        var2(0, 1) = 20.0;  // Second column

        // Perform addition operation
        block.add_variables();

        // Verify results: var1 should now be var1 + var2 (broadcasted)
        REQUIRE(var1(0, 0) == Approx(11.0));  // 1.0 + 10.0
        REQUIRE(var1(1, 0) == Approx(12.0));  // 2.0 + 10.0
        REQUIRE(var1(2, 0) == Approx(13.0));  // 3.0 + 10.0

        REQUIRE(var1(0, 1) == Approx(24.0));  // 4.0 + 20.0
        REQUIRE(var1(1, 1) == Approx(25.0));  // 5.0 + 20.0
        REQUIRE(var1(2, 1) == Approx(26.0));  // 6.0 + 20.0
    }

    SECTION("Add vector to itself operation") {
        // Get variable view
        auto var1 = block.template get<MockVar1>();  // Vector (3 rows)

        // Initialize test data
        var1(0, 0) = 1.0;
        var1(1, 0) = 2.0;
        var1(2, 0) = 3.0;

        var1(0, 1) = 4.0;
        var1(1, 1) = 5.0;
        var1(2, 1) = 6.0;

        // Perform addition operation (adds var1 to itself)
        block.add_vector_to_itself();

        // Verify results: var1 should now be 2 * original
        REQUIRE(var1(0, 0) == Approx(2.0));  // 1.0 * 2
        REQUIRE(var1(1, 0) == Approx(4.0));  // 2.0 * 2
        REQUIRE(var1(2, 0) == Approx(6.0));  // 3.0 * 2

        REQUIRE(var1(0, 1) == Approx(8.0));  // 4.0 * 2
        REQUIRE(var1(1, 1) == Approx(10.0));  // 5.0 * 2
        REQUIRE(var1(2, 1) == Approx(12.0));  // 6.0 * 2
    }

    SECTION("Operations work across multiple rods") {
        // Block has 2 rods: rod 0 with 3 elems (4 nodes), rod 1 with 5 elems (6 nodes)
        // Total width = 4 + 6 = 10

        auto var1 = block.template get<MockVar1>();
        auto var2 = block.template get<MockVar2>();

        // Set values for rod 0 (columns 0-3)
        var1(0, 0) = 1.0;
        var1(1, 0) = 2.0;
        var1(2, 0) = 3.0;
        var2(0, 0) = 5.0;

        // Set values for rod 1 (columns 4-9)
        var1(0, 4) = 10.0;
        var1(1, 4) = 20.0;
        var1(2, 4) = 30.0;
        var2(0, 4) = 50.0;

        // Perform operation
        block.add_variables();

        // Verify rod 0 results
        REQUIRE(var1(0, 0) == Approx(6.0));  // 1.0 + 5.0
        REQUIRE(var1(1, 0) == Approx(7.0));  // 2.0 + 5.0
        REQUIRE(var1(2, 0) == Approx(8.0));  // 3.0 + 5.0

        // Verify rod 1 results
        REQUIRE(var1(0, 4) == Approx(60.0));  // 10.0 + 50.0
        REQUIRE(var1(1, 4) == Approx(70.0));  // 20.0 + 50.0
        REQUIRE(var1(2, 4) == Approx(80.0));  // 30.0 + 50.0
    }

    SECTION("Operations can access block data and methods") {
        // Verify that operations can access block's public interface
        REQUIRE(block.width() > 0);
        REQUIRE(block.depth() == 17);  // MockSystem depth
        REQUIRE(block.n_systems() == 2);

        // Operations should be able to call block methods
        auto var1 = block.template get<MockVar1>();
        REQUIRE(var1.rows() == 3);  // Vector dimension
        REQUIRE(var1.cols() == block.width());
    }
}
