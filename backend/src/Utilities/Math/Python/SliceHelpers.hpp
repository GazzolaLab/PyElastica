//******************************************************************************
// Includes
//******************************************************************************

#pragma once
//
// #include "Utilities/MakeString.hpp"
//
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/Subtensor.h>
//
#include <pybind11/pybind11.h>
//
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>

namespace py_bindings {

  struct SliceInfo {
    //! Start of slice
    std::size_t start;
    //! Length of slice
    std::size_t slicelength;
  };

  //****************************************************************************
  /*!\brief Checks validitity of slice along Axis, raises error if invalid
   * \ingroup python_bindings
   */
  template <std::size_t Axis>
  auto check_slice(const std::size_t limit, pybind11::slice const slice) {
    std::size_t start, stop, step, slicelength;
    if (!slice.compute(limit, &start, &stop, &step, &slicelength))
      throw pybind11::error_already_set();
    if (step != 1)
      throw std::runtime_error(std::string(
          "step !=1 unsupported along axis ") + std::to_string(Axis));

    return SliceInfo{start, slicelength};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Takes 1D slices of an array
   * \ingroup python_bindings
   */
  template <typename T>
  auto array_slice(T& t, pybind11::slice slice) {
    constexpr std::size_t axis = 0UL;
    auto slice_info = check_slice<axis>(t.size(), std::move(slice));
    return ::blaze::subvector(t, slice_info.start, slice_info.slicelength);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Takes 2D slices of an array
   * \ingroup python_bindings
   */
  template <typename T>
  auto array_slice(T& t, std::tuple<pybind11::slice, pybind11::slice> slices) {
    constexpr std::size_t row_slice = 0UL;
    constexpr std::size_t col_slice = 1UL;
    auto slice_info = std::make_tuple(
        check_slice<row_slice>(t.rows(),
                               std::get<row_slice>(std::move(slices))),
        check_slice<col_slice>(t.columns(),
                               std::get<col_slice>(std::move(slices))));
    return ::blaze::submatrix(t, std::get<row_slice>(slice_info).start,
                              std::get<col_slice>(slice_info).start,
                              std::get<row_slice>(slice_info).slicelength,
                              std::get<col_slice>(slice_info).slicelength);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Takes 3D slices of an array
   * \ingroup python_bindings
   */
  template <typename T>
  auto array_slice(
      T& t,
      std::tuple<pybind11::slice, pybind11::slice, pybind11::slice> slices) {
    constexpr std::size_t page_slice = 0UL;
    constexpr std::size_t row_slice = 1UL;
    constexpr std::size_t col_slice = 2UL;
    auto slice_info = std::make_tuple(
        check_slice<page_slice>(t.pages(),
                                std::get<page_slice>(std::move(slices))),
        check_slice<row_slice>(t.rows(),
                               std::get<row_slice>(std::move(slices))),
        check_slice<col_slice>(t.columns(),
                               std::get<col_slice>(std::move(slices))));
    return ::blaze::subtensor(t, std::get<page_slice>(slice_info).start,
                              std::get<row_slice>(slice_info).start,
                              std::get<col_slice>(slice_info).start,
                              std::get<page_slice>(slice_info).slicelength,
                              std::get<row_slice>(slice_info).slicelength,
                              std::get<col_slice>(slice_info).slicelength);
  }
  //****************************************************************************

}  // namespace py_bindings
