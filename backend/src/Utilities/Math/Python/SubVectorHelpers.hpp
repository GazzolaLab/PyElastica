#pragma once

//******************************************************************************
// Includes
//******************************************************************************

//
#include "PythonBindings/BoundChecks.hpp"
#include "Utilities/PrintHelpers.hpp"
//
#include "Utilities/Math/Python/SliceHelpers.hpp"
//
#include <memory>
#include <sstream>
#include <tuple>
#include <utility>
//
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
//
#include <blaze/math/DynamicVector.h>
#include <blaze/math/Subvector.h>

namespace py_bindings {

  //****************************************************************************
  /*!\brief Generates bindings of vector views in \elastica
   * \ingroup python_bindings
   */
  template <typename VectorView>
  void bind_blaze_subvector_helper(
      pybind11::class_<VectorView>& py_vector_view) {  // NOLINT
    using type = VectorView;
    using ElementType = typename VectorView::ElementType;
    namespace py = pybind11;

    // Wrapper for basic DataVector operations
    py_vector_view
        // Expose the data as a Python buffer so it can be cast into Numpy
        // arrays
        .def_buffer([](type& t) {
          return py::buffer_info(t.data(),
                                 // Size of one scalar
                                 sizeof(ElementType),
                                 py::format_descriptor<ElementType>::format(),
                                 // Number of dimensions
                                 1,
                                 // Size of the buffer
                                 {t.size()},
                                 // Stride for each index (in bytes)
                                 {sizeof(ElementType)});
        })
        .def(
            "__iter__",
            [](const type& t) { return py::make_iterator(t.begin(), t.end()); },
            // Keep object alive while iterator exists
            py::keep_alive<0, 1>())
        // __len__ is for being able to write len(my_data_vector) in python
        .def("__len__", [](const type& t) { return t.size(); })
        // define shape mimicking numpy interface
        .def_property_readonly(
            "shape",
            +[](const type& t) { return std::tuple<std::size_t>(t.size()); })
        // __getitem__ and __setitem__ are the subscript operators
        // (operator[]). To define (and overload) operator() use __call__
        .def(
            "__getitem__",
            +[](const type& t, const size_t i) {
              bounds_check(t, i);
              return t[i];
            })
        .def(
            "__getitem__",
            +[](type& t, const py::slice slice) {
              return array_slice(t, std::move(slice));
            })
        .def(
            "__setitem__",
            +[](type& t, const size_t i, const ElementType v) {
              bounds_check(t, i);
              t[i] = v;
            })
        .def(py::self == py::self)
        .def(py::self != py::self);

    static const auto printer = [](type const& t) {
      // Blaze's default printing adds extra lines and spaces which is
      // not what we want
      std::ostringstream os;
      sequence_print_helper(os, t.begin(), t.end());
      return os.str();
    };

    // Need __str__ for converting to string/printing
    py_vector_view
        .def("__str__", +printer)
        // repr allows you to output the object in an interactive python
        // terminal using obj to get the "string REPResenting the object".
        .def("__repr__", +printer);
  }
  //****************************************************************************

}  // namespace py_bindings
