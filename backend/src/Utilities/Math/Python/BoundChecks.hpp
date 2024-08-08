#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>
#include <stdexcept>
#include <string>

// #include "Utilities/PrettyType.hpp"

namespace py_bindings {

  //****************************************************************************
  /*! \brief Check if a vector-like object access is in bounds. Throws
   * std::runtime_error if it is not.
   * \ingroup python_bindings
   */
  template <typename T>
  void bounds_check(const T& t, const std::size_t i) {
    if (i >= t.size()) {
      throw std::runtime_error{"Out of bounds access (" + std::to_string(i) +
                               ") into " + typeid(t).name() +  // pretty_type::name<T>() +
                               " of size " + std::to_string(t.size())};
    }
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check if a matrix-like object access is in bounds. Throws
   * std::runtime_error if it is not.
   * \ingroup python_bindings
   */
  template <typename T>
  void matrix_bounds_check(const T& matrix, const std::size_t row,
                           const std::size_t column) {
    if (row >= matrix.rows() or column >= matrix.columns()) {
      throw std::runtime_error{"Out of bounds access (" + std::to_string(row) +
                               ", " + std::to_string(column) +
                               ") into Matrix of size (" +
                               std::to_string(matrix.rows()) + ", " +
                               std::to_string(matrix.columns()) + ")"};
    }
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Check if a tensor-like object access is in bounds. Throws
   * std::runtime_error if it is not.
   * \ingroup python_bindings
   */
  template <typename T>
  void tensor_bounds_check(const T& tensor, const std::size_t page,
                           const std::size_t row, const std::size_t column) {
    if (page >= tensor.pages() or row >= tensor.rows() or
        column >= tensor.columns()) {
      throw std::runtime_error{
          "Out of bounds access (" + std::to_string(page) + ", " +
          std::to_string(row) + ", " + std::to_string(column) +
          ") into Tensor of size (" + std::to_string(tensor.pages()) + ", " +
          std::to_string(tensor.rows()) + ", " +
          std::to_string(tensor.columns()) + ")"};
    }
  }
  //****************************************************************************

}  // namespace py_bindings
