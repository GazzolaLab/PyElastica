//******************************************************************************
// Includes
//******************************************************************************


#include "Utilities/DefineTypes.h"
#include "Utilities/MakeString.hpp"
//
#include "Utilities/Math/Python/SliceHelpers.hpp"
#include "Utilities/Math/Python/BoundChecks.hpp"
//
#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//
#include <blaze_tensor/math/DynamicTensor.h>

namespace py = pybind11;

namespace py_bindings {

  //****************************************************************************
  /*!\brief Helps bind a tensor type in \elastica
   * \ingroup python_bindings
   */
  void bind_blaze_tensor(py::module& m) {  // NOLINT
    using Real = ::elastica::real_t;
    using type = ::blaze::DynamicTensor<Real>;

    // Wrapper for basic type operations
    py::class_<type>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::size_t, std::size_t, std::size_t>(),
             py::arg("pages"), py::arg("rows"), py::arg("columns"))
        .def(py::init([](py::buffer buffer) {
               py::buffer_info info = buffer.request();
               // Sanity-check the buffer
               if (info.format != py::format_descriptor<Real>::format()) {
                 throw std::runtime_error(
                     "Incompatible format: expected a Real array.");
               }
               if (info.ndim != 3) {
                 throw std::runtime_error("Incompatible dimension.");
               }
               const auto pages = static_cast<std::size_t>(info.shape[0]);
               const auto rows = static_cast<std::size_t>(info.shape[1]);
               const auto columns = static_cast<std::size_t>(info.shape[2]);
               auto data = static_cast<Real*>(info.ptr);
               return type(pages, rows, columns, data);
             }),
             py::arg("buffer"))
        // Expose the data as a Python buffer so it can be cast into Numpy
        // arrays
        .def_buffer([](type& tensor) {
          return py::buffer_info(
              tensor.data(),
              // Size of one scalar
              sizeof(Real), py::format_descriptor<Real>::format(),
              // Number of dimensions
              3,
              // Size of the buffer
              {tensor.pages(), tensor.rows(), tensor.columns()},
              // Stride for each index (in bytes). Data is stored
              // in column-major layout (see `type.hpp`).
              {sizeof(Real) * tensor.rows() * tensor.spacing(),
               sizeof(Real) * tensor.spacing(), sizeof(Real)});
        })
        .def_property_readonly(
            "shape",
            +[](const type& self) {
              return std::tuple<std::size_t, std::size_t, std::size_t>(
                  self.pages(), self.rows(), self.columns());
            })
        // __getitem__ and __setitem__ are the subscript operators (M[*,*]).
        .def(
            "__getitem__",
            +[](const type& self,
                const std::tuple<std::size_t, std::size_t, std::size_t>& x) {
              tensor_bounds_check(self, std::get<0>(x), std::get<1>(x),
                               std::get<2>(x));
              return self(std::get<0>(x), std::get<1>(x), std::get<2>(x));
            })
        .def(
            "__getitem__",
            +[](type& self, std::tuple<py::slice, py::slice, py::slice> slice) {
              return array_slice(self, std::move(slice));
            })
        .def(
            "__setitem__",
            +[](type& self,
                const std::tuple<std::size_t, std::size_t, std::size_t>& x,
                const Real val) {
              tensor_bounds_check(self, std::get<0>(x), std::get<1>(x),
                                std::get<2>(x));
              self(std::get<0>(x), std::get<1>(x), std::get<2>(x)) = val;
            })
        // Need __str__ for converting to string/printing
        .def(
            "__str__", +[](const type& self) {
              return std::string(MakeString{} << self);
            });
  }
  //****************************************************************************

}  // namespace py_bindings
