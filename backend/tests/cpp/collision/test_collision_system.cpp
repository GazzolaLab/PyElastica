#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <vector>
#include <utility>
#include <cstddef>
#include <Eigen/Dense>
#include "../../../src/environment/collision/collision_system.h"
#include "../../../src/environment/collision/types.h"
#include "../../../src/api.h"

using Catch::Approx;

namespace elasticapp::environment::collision {
namespace testing {

/**
 * Null coarse detection policy for unit testing.
 * Returns empty vector (no candidate pairs detected).
 */
struct NullCoarseDetection {
    std::vector<std::pair<std::size_t, std::size_t>> detect(
        const Eigen::MatrixXd& positions,
        const Eigen::MatrixXd& radii
    ) const {
        (void)positions;
        (void)radii;
        return {};
    }
};

/**
 * Null fine detection policy for unit testing.
 * Returns empty vector (no contacts detected).
 */
struct NullFineDetection {
    template<typename PhysicsModel>
    std::vector<Contact> detect(
        const std::vector<std::pair<std::size_t, std::size_t>>& candidate_pairs,
        const Eigen::MatrixXd& positions,
        const Eigen::MatrixXd& velocities,
        const Eigen::MatrixXd& radii,
        const PhysicsModel& physics_model
    ) const {
        (void)candidate_pairs;
        (void)positions;
        (void)velocities;
        (void)radii;
        (void)physics_model;
        return {};
    }
};

/**
 * Null batching policy for unit testing.
 * Returns empty batches.
 */
struct NullBatching {
    std::vector<std::vector<std::size_t>> batch(
        std::vector<Contact>& contacts
    ) const {
        (void)contacts;
        return {};
    }
};

} // namespace testing
} // namespace elasticapp::environment::collision

// Type alias for test CollisionSystem with null policies
using TestCollisionSystem = elasticapp::environment::collision::CollisionSystem<
    elasticapp::environment::collision::testing::NullCoarseDetection,
    elasticapp::environment::collision::testing::NullFineDetection,
    elasticapp::environment::collision::testing::NullBatching
>;

TEST_CASE("CollisionSystem construction", "[collision]") {
    SECTION("Can be constructed with nullptr_t physics model") {
        std::nullptr_t null_model = nullptr;
        TestCollisionSystem system(null_model);

        // Should construct successfully
        REQUIRE(true);
    }

    SECTION("Can be constructed with nullptr_t physics model and detect_every") {
        std::nullptr_t null_model = nullptr;
        TestCollisionSystem system(null_model, 5);

        REQUIRE(system.detect_every() == 5);
    }

    SECTION("Default detect_every is 1") {
        std::nullptr_t null_model = nullptr;
        TestCollisionSystem system(null_model);

        REQUIRE(system.detect_every() == 1);
    }
}

TEST_CASE("CollisionSystem contact_cache", "[collision]") {
    std::nullptr_t null_model = nullptr;
    TestCollisionSystem system(null_model);

    SECTION("contact_cache() returns non-const reference") {
        auto& cache = system.contact_cache();
        REQUIRE(cache.empty());

        // Can modify cache
        cache.push_back({0, 1});
        cache.push_back({2, 3});

        REQUIRE(cache.size() == 2);
        REQUIRE(cache[0] == std::make_pair<std::size_t, std::size_t>(0, 1));
        REQUIRE(cache[1] == std::make_pair<std::size_t, std::size_t>(2, 3));
    }

    SECTION("contact_cache() const returns const reference") {
        const auto& system_const = system;
        const auto& cache = system_const.contact_cache();

        // Should be empty initially
        REQUIRE(cache.empty());

        // Modify via non-const reference
        system.contact_cache().push_back({4, 5});

        // Const reference should see the change
        REQUIRE(cache.size() == 1);
        REQUIRE(cache[0] == std::make_pair<std::size_t, std::size_t>(4, 5));
    }

    SECTION("contact_cache persists across calls") {
        auto& cache1 = system.contact_cache();
        cache1.push_back({10, 20});

        auto& cache2 = system.contact_cache();
        REQUIRE(cache2.size() == 1);
        REQUIRE(cache2[0] == std::make_pair<std::size_t, std::size_t>(10, 20));
    }
}

TEST_CASE("CollisionSystem detect_every", "[collision]") {
    std::nullptr_t null_model = nullptr;
    TestCollisionSystem system(null_model);

    SECTION("get detect_every returns current value") {
        REQUIRE(system.detect_every() == 1);
    }

    SECTION("set_detect_every updates the value") {
        system.set_detect_every(3);
        REQUIRE(system.detect_every() == 3);

        system.set_detect_every(10);
        REQUIRE(system.detect_every() == 10);
    }

    SECTION("set_detect_every can set to 0") {
        system.set_detect_every(0);
        REQUIRE(system.detect_every() == 0);
    }

    SECTION("set_detect_every can set to large values") {
        system.set_detect_every(1000);
        REQUIRE(system.detect_every() == 1000);
    }
}

TEST_CASE("CollisionSystem with null policies", "[collision]") {
    SECTION("System can be instantiated with all null policies") {
        std::nullptr_t null_model = nullptr;
        TestCollisionSystem system(null_model);

        // Should construct successfully
        REQUIRE(true);
    }

    SECTION("System methods work with null policies") {
        std::nullptr_t null_model = nullptr;
        TestCollisionSystem system(null_model);

        // All public methods should work
        auto& cache = system.contact_cache();
        REQUIRE(cache.empty());

        REQUIRE(system.detect_every() == 1);
        system.set_detect_every(2);
        REQUIRE(system.detect_every() == 2);
    }
}
