#include <catch2/catch_test_macros.hpp>
#include "traits.h"
#include "block.h"
#include "mock/mock_block_system.h"

using MockBlockSystem = elasticapp::mock::MockBlockSystem;

TEST_CASE("Traits - Matrix storage order", "[traits]") {
    SECTION("Matrix type is defined") {
        std::vector<std::size_t> n_elems_per_rod = {3};  // 3 elements -> 4 nodes (width)
        MockBlockSystem block(n_elems_per_rod);
        auto& matrix = block.data();

        // Verify matrix has correct dimensions
        REQUIRE(matrix.rows() == 17);  // MockSystem depth
        REQUIRE(matrix.cols() == 4);    // (3+1) nodes + 0 ghost = 4 nodes

        // Verify we can access and modify data
        matrix(0, 0) = 1.0;
        REQUIRE(matrix(0, 0) == 1.0);
    }

    SECTION("Stride computation") {
        auto strides = elasticapp::compute_strides(3, 4);

        if constexpr (elasticapp::IsRowMajor) {
            // Row-major: row stride = cols * sizeof(double), col stride = sizeof(double)
            REQUIRE(strides.first == static_cast<ptrdiff_t>(4 * sizeof(double)));
            REQUIRE(strides.second == static_cast<ptrdiff_t>(sizeof(double)));
        } else {
            // Column-major: row stride = sizeof(double), col stride = rows * sizeof(double)
            REQUIRE(strides.first == static_cast<ptrdiff_t>(sizeof(double)));
            REQUIRE(strides.second == static_cast<ptrdiff_t>(3 * sizeof(double)));
        }
    }
}

TEST_CASE("Traits - get_column_slice", "[traits]") {
    SECTION("Basic column slice - non-const") {
        elasticapp::MatrixType matrix(5, 10);  // 5 rows, 10 columns
        matrix.setZero();

        // Fill some columns with test data
        for (std::size_t col = 0; col < 10; ++col) {
            for (std::size_t row = 0; row < 5; ++row) {
                matrix(row, col) = static_cast<double>(col * 10 + row);
            }
        }

        // Get a slice of columns 2-5 (4 columns)
        auto slice = elasticapp::get_column_slice(matrix, 2, 4);

        // Verify dimensions
        REQUIRE(slice.rows() == 5);
        REQUIRE(slice.cols() == 4);

        // Verify data matches original matrix
        for (std::size_t col = 0; col < 4; ++col) {
            for (std::size_t row = 0; row < 5; ++row) {
                REQUIRE(slice(row, col) == matrix(row, col + 2));
            }
        }
    }

    SECTION("Column slice is a view, not a copy") {
        elasticapp::MatrixType matrix(3, 6);
        matrix.setZero();

        // Get a slice
        auto slice = elasticapp::get_column_slice(matrix, 1, 3);

        // Modify through the slice
        slice(0, 0) = 42.0;
        slice(1, 1) = 99.0;
        slice(2, 2) = 123.0;

        // Verify original matrix is modified
        REQUIRE(matrix(0, 1) == 42.0);   // slice(0,0) -> matrix(0,1)
        REQUIRE(matrix(1, 2) == 99.0);    // slice(1,1) -> matrix(1,2)
        REQUIRE(matrix(2, 3) == 123.0);   // slice(2,2) -> matrix(2,3)
    }

    SECTION("Basic column slice - const") {
        elasticapp::MatrixType matrix(4, 8);
        matrix.setZero();

        // Fill with test data
        for (std::size_t col = 0; col < 8; ++col) {
            for (std::size_t row = 0; row < 4; ++row) {
                matrix(row, col) = static_cast<double>(col * 100 + row);
            }
        }

        // Get a const slice
        const elasticapp::MatrixType& const_matrix = matrix;
        auto const_slice = elasticapp::get_column_slice(const_matrix, 3, 2);

        // Verify dimensions
        REQUIRE(const_slice.rows() == 4);
        REQUIRE(const_slice.cols() == 2);

        // Verify data matches
        REQUIRE(const_slice(0, 0) == matrix(0, 3));
        REQUIRE(const_slice(3, 1) == matrix(3, 4));
    }

    SECTION("Single column slice") {
        elasticapp::MatrixType matrix(5, 7);
        matrix.setZero();

        // Fill column 3 with test data
        for (std::size_t row = 0; row < 5; ++row) {
            matrix(row, 3) = static_cast<double>(row * 10);
        }

        auto slice = elasticapp::get_column_slice(matrix, 3, 1);

        REQUIRE(slice.rows() == 5);
        REQUIRE(slice.cols() == 1);

        for (std::size_t row = 0; row < 5; ++row) {
            REQUIRE(slice(row, 0) == static_cast<double>(row * 10));
        }
    }

    SECTION("Full matrix slice") {
        elasticapp::MatrixType matrix(6, 8);
        matrix.setZero();

        // Fill with test data
        for (std::size_t col = 0; col < 8; ++col) {
            for (std::size_t row = 0; row < 6; ++row) {
                matrix(row, col) = static_cast<double>(row * 8 + col);
            }
        }

        // Get slice of all columns
        auto slice = elasticapp::get_column_slice(matrix, 0, 8);

        REQUIRE(slice.rows() == 6);
        REQUIRE(slice.cols() == 8);

        // Verify all data matches
        for (std::size_t col = 0; col < 8; ++col) {
            for (std::size_t row = 0; row < 6; ++row) {
                REQUIRE(slice(row, col) == matrix(row, col));
            }
        }
    }

    SECTION("Slice at end of matrix") {
        elasticapp::MatrixType matrix(4, 10);
        matrix.setZero();

        // Fill last 3 columns
        for (std::size_t col = 7; col < 10; ++col) {
            for (std::size_t row = 0; row < 4; ++row) {
                matrix(row, col) = static_cast<double>(col * 100 + row);
            }
        }

        auto slice = elasticapp::get_column_slice(matrix, 7, 3);

        REQUIRE(slice.rows() == 4);
        REQUIRE(slice.cols() == 3);

        // Verify data
        REQUIRE(slice(0, 0) == matrix(0, 7));
        REQUIRE(slice(3, 2) == matrix(3, 9));
    }
}

TEST_CASE("Traits - get_block_slice", "[traits]") {
    SECTION("Basic block slice - non-const") {
        elasticapp::MatrixType matrix(8, 12);
        matrix.setZero();

        // Fill matrix with test data
        for (std::size_t row = 0; row < 8; ++row) {
            for (std::size_t col = 0; col < 12; ++col) {
                matrix(row, col) = static_cast<double>(row * 100 + col);
            }
        }

        // Get a block slice: rows 2-5 (4 rows), columns 3-7 (5 columns)
        auto slice = elasticapp::get_block_slice(matrix, 2, 4, 3, 5);

        // Verify dimensions
        REQUIRE(slice.rows() == 4);
        REQUIRE(slice.cols() == 5);

        // Verify data matches original matrix
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t col = 0; col < 5; ++col) {
                REQUIRE(slice(row, col) == matrix(row + 2, col + 3));
            }
        }
    }

    SECTION("Block slice is a view, not a copy") {
        elasticapp::MatrixType matrix(6, 10);
        matrix.setZero();

        // Get a block slice
        auto slice = elasticapp::get_block_slice(matrix, 1, 3, 2, 4);

        // Modify through the slice
        slice(0, 0) = 42.0;
        slice(1, 1) = 99.0;
        slice(2, 2) = 123.0;

        // Verify original matrix is modified
        REQUIRE(matrix(1, 2) == 42.0);   // slice(0,0) -> matrix(1,2)
        REQUIRE(matrix(2, 3) == 99.0);    // slice(1,1) -> matrix(2,3)
        REQUIRE(matrix(3, 4) == 123.0);   // slice(2,2) -> matrix(3,4)
    }

    SECTION("Basic block slice - const") {
        elasticapp::MatrixType matrix(5, 8);
        matrix.setZero();

        // Fill with test data
        for (std::size_t row = 0; row < 5; ++row) {
            for (std::size_t col = 0; col < 8; ++col) {
                matrix(row, col) = static_cast<double>(row * 50 + col);
            }
        }

        // Get a const block slice
        const elasticapp::MatrixType& const_matrix = matrix;
        auto const_slice = elasticapp::get_block_slice(const_matrix, 2, 2, 3, 3);

        // Verify dimensions
        REQUIRE(const_slice.rows() == 2);
        REQUIRE(const_slice.cols() == 3);

        // Verify data matches
        REQUIRE(const_slice(0, 0) == matrix(2, 3));
        REQUIRE(const_slice(1, 2) == matrix(3, 5));
    }

    SECTION("Single element block slice") {
        elasticapp::MatrixType matrix(5, 7);
        matrix.setZero();

        // Fill specific element
        matrix(3, 4) = 42.0;

        auto slice = elasticapp::get_block_slice(matrix, 3, 1, 4, 1);

        REQUIRE(slice.rows() == 1);
        REQUIRE(slice.cols() == 1);
        REQUIRE(slice(0, 0) == 42.0);
    }

    SECTION("Full matrix block slice") {
        elasticapp::MatrixType matrix(6, 8);
        matrix.setZero();

        // Fill with test data
        for (std::size_t row = 0; row < 6; ++row) {
            for (std::size_t col = 0; col < 8; ++col) {
                matrix(row, col) = static_cast<double>(row * 10 + col);
            }
        }

        // Get slice of entire matrix
        auto slice = elasticapp::get_block_slice(matrix, 0, 6, 0, 8);

        REQUIRE(slice.rows() == 6);
        REQUIRE(slice.cols() == 8);

        // Verify all data matches
        for (std::size_t row = 0; row < 6; ++row) {
            for (std::size_t col = 0; col < 8; ++col) {
                REQUIRE(slice(row, col) == matrix(row, col));
            }
        }
    }

    SECTION("Block slice at corner of matrix") {
        elasticapp::MatrixType matrix(7, 9);
        matrix.setZero();

        // Fill bottom-right corner
        for (std::size_t row = 4; row < 7; ++row) {
            for (std::size_t col = 6; col < 9; ++col) {
                matrix(row, col) = static_cast<double>(row * 100 + col);
            }
        }

        auto slice = elasticapp::get_block_slice(matrix, 4, 3, 6, 3);

        REQUIRE(slice.rows() == 3);
        REQUIRE(slice.cols() == 3);

        // Verify data
        REQUIRE(slice(0, 0) == matrix(4, 6));
        REQUIRE(slice(2, 2) == matrix(6, 8));
    }

    SECTION("Block slice with single row") {
        elasticapp::MatrixType matrix(5, 8);
        matrix.setZero();

        // Fill row 2
        for (std::size_t col = 0; col < 8; ++col) {
            matrix(2, col) = static_cast<double>(col * 10);
        }

        auto slice = elasticapp::get_block_slice(matrix, 2, 1, 0, 8);

        REQUIRE(slice.rows() == 1);
        REQUIRE(slice.cols() == 8);

        for (std::size_t col = 0; col < 8; ++col) {
            REQUIRE(slice(0, col) == static_cast<double>(col * 10));
        }
    }

    SECTION("Block slice with single column") {
        elasticapp::MatrixType matrix(6, 7);
        matrix.setZero();

        // Fill column 3
        for (std::size_t row = 0; row < 6; ++row) {
            matrix(row, 3) = static_cast<double>(row * 20);
        }

        auto slice = elasticapp::get_block_slice(matrix, 0, 6, 3, 1);

        REQUIRE(slice.rows() == 6);
        REQUIRE(slice.cols() == 1);

        for (std::size_t row = 0; row < 6; ++row) {
            REQUIRE(slice(row, 0) == static_cast<double>(row * 20));
        }
    }
}
