#include <catch2/catch_test_macros.hpp>
#include "version.h"
#include <string>

// Simple test to verify Catch2 is working
TEST_CASE("Catch2 is working", "[basic]") {
    REQUIRE(1 + 1 == 2);
    REQUIRE(std::string("hello") == "hello");
}

// Test version string from version.cpp
TEST_CASE("Version string from version.cpp", "[version]") {
    std::string version = elasticapp::version();
    REQUIRE(version.length() > 0);
    REQUIRE(version.find('.') != std::string::npos);
}
