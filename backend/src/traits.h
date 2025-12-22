#pragma once

#include <Eigen/Dense>
#include <cstddef>

namespace elasticapp {

// Compile-time switch for matrix storage order
// Can be overridden via CMake: -DELASTICAPP_USE_ROW_MAJOR=ON
// Default is column-major (Eigen's default)
#ifndef ELASTICAPP_USE_ROW_MAJOR
#define ELASTICAPP_USE_ROW_MAJOR 0
#endif

// Matrix type based on storage order
#if ELASTICAPP_USE_ROW_MAJOR
using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
constexpr bool IsRowMajor = true;
constexpr bool IsColMajor = false;
#else
using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
constexpr bool IsRowMajor = false;
constexpr bool IsColMajor = true;
#endif

// Helper functions for computing strides for numpy array views
inline std::pair<ptrdiff_t, ptrdiff_t> compute_strides(std::size_t rows, std::size_t cols) {
    if constexpr (IsRowMajor) {
        // Row-major: stride between rows is cols * sizeof(double)
        // stride between columns is sizeof(double)
        return {static_cast<ptrdiff_t>(cols * sizeof(double)),
                static_cast<ptrdiff_t>(sizeof(double))};
    } else {
        // Column-major: stride between rows is sizeof(double)
        // stride between columns is rows * sizeof(double)
        return {static_cast<ptrdiff_t>(sizeof(double)),
                static_cast<ptrdiff_t>(rows * sizeof(double))};
    }
}

// Helper functions for matrix slicing operations
// These encapsulate Eigen-specific operations

// Get a view into a submatrix (columns for a rod)
// Returns a lightweight view - no copy, just a reference to the columns
inline auto get_column_slice(MatrixType& matrix, std::size_t start_col, std::size_t num_cols) {
    return matrix.middleCols(static_cast<Eigen::Index>(start_col),
                             static_cast<Eigen::Index>(num_cols));
}

inline auto get_column_slice(const MatrixType& matrix, std::size_t start_col, std::size_t num_cols) {
    return matrix.middleCols(static_cast<Eigen::Index>(start_col),
                             static_cast<Eigen::Index>(num_cols));
}

// Get a view into a submatrix (specific rows and columns)
// Returns a lightweight view - no copy, just a reference to the submatrix
inline auto get_block_slice(MatrixType& matrix,
                            std::size_t start_row, std::size_t num_rows,
                            std::size_t start_col, std::size_t num_cols) {
    return matrix.block(static_cast<Eigen::Index>(start_row),
                        static_cast<Eigen::Index>(start_col),
                        static_cast<Eigen::Index>(num_rows),
                        static_cast<Eigen::Index>(num_cols));
}

inline auto get_block_slice(const MatrixType& matrix,
                            std::size_t start_row, std::size_t num_rows,
                            std::size_t start_col, std::size_t num_cols) {
    return matrix.block(static_cast<Eigen::Index>(start_row),
                        static_cast<Eigen::Index>(start_col),
                        static_cast<Eigen::Index>(num_rows),
                        static_cast<Eigen::Index>(num_cols));
}

} // namespace elasticapp
