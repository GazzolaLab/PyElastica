#include <catch2/catch_test_macros.hpp>
#include <type_traits>
#include "system.h"
#include "cosserat_rod_system.h"
#include "mock/mock_block_system.h"

TEST_CASE("Variable tags - compile-time only", "[system]") {
    // Variables should be tag types, not instantiable
    // We can use them as template parameters

    SECTION("Variable placement types exist") {
        static_assert(std::is_same_v<elasticapp::Placement::OnNode, elasticapp::Placement::OnNode>);
        static_assert(std::is_same_v<elasticapp::Placement::OnElement, elasticapp::Placement::OnElement>);
        static_assert(std::is_same_v<elasticapp::Placement::OnVoronoi, elasticapp::Placement::OnVoronoi>);
    }
}

TEST_CASE("Dimension types", "[system]") {
    SECTION("Dimension types exist") {
        static_assert(std::is_empty_v<elasticapp::DataType::Scalar>);
        static_assert(std::is_empty_v<elasticapp::DataType::Vector>);
        static_assert(std::is_empty_v<elasticapp::DataType::Matrix>);
    }

    SECTION("Dimension types have correct sizes") {
        STATIC_REQUIRE(std::is_integral_v<decltype(elasticapp::DataType::Scalar::dimension)>);
        STATIC_REQUIRE(std::is_integral_v<decltype(elasticapp::DataType::Vector::dimension)>);
        STATIC_REQUIRE(std::is_integral_v<decltype(elasticapp::DataType::Matrix::dimension)>);
    }
}

TEST_CASE("System block depth computation", "[system]") {

    using MockSystem = elasticapp::mock::MockSystem;

    SECTION("MockSystem has correct depth") {
        REQUIRE(MockSystem::get_depth() == 17);
    }

}
