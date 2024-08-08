//******************************************************************************
// Includes
//******************************************************************************

//
#include "Utilities/DefineTypes.h"
#include "Utilities/PrintHelpers.hpp"
//
#include "Utilities/Math/Python/SliceHelpers.hpp"
#include "Utilities/Math/Python/BoundChecks.hpp"
//
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
//
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//
#include <blaze/math/DynamicVector.h>
//

namespace py = pybind11;

namespace py_bindings {

  //****************************************************************************
  /*!\brief Helps bind a vector type in \elastica
   * \ingroup python_bindings
   */
  void bind_blaze_vector(py::module& m) {  // NOLINT

    using Real = ::elastica::real_t;
    using type = ::blaze::DynamicVector<Real, ::blaze::columnVector,
                                        ::blaze::AlignedAllocator<Real>>;
    // Wrapper for basic DataVector operations
    auto py_vector =
        py::class_<type>(m, "Vector", py::buffer_protocol())
            .def(py::init<std::size_t>(), py::arg("size"))
            .def(py::init<std::size_t, Real>(), py::arg("size"),
                 py::arg("fill"))
            .def(py::init([](std::vector<Real> const& values) {
                   type result(values.size());
                   std::copy(values.begin(), values.end(), result.begin());
                   return result;
                 }),
                 py::arg("values"))
            .def(py::init([](py::buffer buffer) {
                   py::buffer_info info = buffer.request();
                   // Sanity-check the buffer
                   if (info.format != py::format_descriptor<Real>::format()) {
                     throw std::runtime_error(
                         "Incompatible format: expected a Real array");
                   }
                   if (info.ndim != 1) {
                     throw std::runtime_error("Incompatible dimension.");
                   }
                   const auto size = static_cast<std::size_t>(info.shape[0]);
                   auto data = static_cast<Real*>(info.ptr);
                   type result(size);
                   std::copy_n(data, result.size(), result.begin());
                   return result;
                 }),
                 py::arg("buffer"))
            // Expose the data as a Python buffer so it can be cast into Numpy
            // arrays
            .def_buffer([](type& t) {
              return py::buffer_info(t.data(),
                                     // Size of one scalar
                                     sizeof(Real),
                                     py::format_descriptor<Real>::format(),
                                     // Number of dimensions
                                     1,
                                     // Size of the buffer
                                     {t.size()},
                                     // Stride for each index (in bytes)
                                     {sizeof(Real)});
            })
            .def(
                "__iter__",
                [](const type& t) {
                  return py::make_iterator(t.begin(), t.end());
                },
                // Keep object alive while iterator exists
                py::keep_alive<0, 1>())
            // __len__ is for being able to write len(my_data_vector) in python
            .def("__len__", [](const type& t) { return t.size(); })
            .def_property_readonly(
                "shape",
                +[](const type& t) {
                  return std::tuple<std::size_t>(t.size());
                })
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
                "__setitem__", +[](type& t, const size_t i, const Real v) {
                  bounds_check(t, i);
                  t[i] = v;
                });

    static const auto printer = [](type const &t) {
        // Blaze's default printing adds extra lines and spaces which is
        // not what we want
        std::ostringstream os;
        sequence_print_helper(os, t.begin(), t.end());
        return os.str();
    };

    // Need __str__ for converting to string
    py_vector
        .def("__str__", +printer)
        // repr allows you to output the object in an interactive python
        // terminal using obj to get the "string REPResenting the object".
        .def("__repr__", +printer)
        .def(py::self += py::self)
        // Need to do math explicitly converting to DataVector because we don't
        // want to represent all the possible expression template types
        .def(
            "abs", +[](const type& t) { return type{abs(t)}; })
        .def(
            "acos", +[](const type& t) { return type{acos(t)}; })
        .def(
            "acosh", +[](const type& t) { return type{acosh(t)}; })
        .def(
            "asin", +[](const type& t) { return type{asin(t)}; })
        .def(
            "asinh", +[](const type& t) { return type{asinh(t)}; })
        .def(
            "atan", +[](const type& t) { return type{atan(t)}; })
        .def(
            "atan2",
            +[](const type& y, const type& x) { return type{atan2(y, x)}; })
        .def(
            "atanh", +[](const type& t) { return type{atanh(t)}; })
        .def(
            "cbrt", +[](const type& t) { return type{cbrt(t)}; })
        .def(
            "cos", +[](const type& t) { return type{cos(t)}; })
        .def(
            "cosh", +[](const type& t) { return type{cosh(t)}; })
        .def(
            "erf", +[](const type& t) { return type{erf(t)}; })
        .def(
            "erfc", +[](const type& t) { return type{erfc(t)}; })
        .def(
            "exp", +[](const type& t) { return type{exp(t)}; })
        .def(
            "exp2", +[](const type& t) { return type{exp2(t)}; })
        .def(
            "exp10", +[](const type& t) { return type{exp10(t)}; })
        //        .def(
        //            "fabs", +[](const type& t) { return type{fabs(t)}; })
        .def(
            "hypot",
            +[](const type& x, const type& y) { return type{hypot(x, y)}; })
        .def(
            "invcbrt", +[](const type& t) { return type{invcbrt(t)}; })
        .def(
            "invsqrt", +[](const type& t) { return type{invsqrt(t)}; })
        .def(
            "log", +[](const type& t) { return type{log(t)}; })
        .def(
            "log2", +[](const type& t) { return type{log2(t)}; })
        .def(
            "log10", +[](const type& t) { return type{log10(t)}; })
        .def(
            "max", +[](const type& t) { return Real{max(t)}; })
        .def(
            "min", +[](const type& t) { return Real{min(t)}; })
        .def(
            "pow",
            +[](const type& base, double exp) { return type{pow(base, exp)}; })
        .def(
            "sin", +[](const type& t) { return type{sin(t)}; })
        .def(
            "sinh", +[](const type& t) { return type{sinh(t)}; })
        .def(
            "sqrt", +[](const type& t) { return type{sqrt(t)}; })
        .def(
            "tan", +[](const type& t) { return type{tan(t)}; })
        .def(
            "tanh", +[](const type& t) { return type{tanh(t)}; })
        .def(
            "__pow__", +[](const type& base,
                           const double exp) { return type{pow(base, exp)}; })
        .def(
            "__add__", +[](const type& self,
                           const Real other) { return type{self + other}; })
        .def(
            "__radd__", +[](const type& self,
                            const Real other) { return type{other + self}; })
        .def(
            "__sub__", +[](const type& self,
                           const Real other) { return type{self - other}; })
        .def(
            "__rsub__", +[](const type& self,
                            const Real other) { return type{other - self}; })
        .def(
            "__mul__", +[](const type& self,
                           const Real other) { return type{self * other}; })
        .def(
            "__rmul__", +[](const type& self,
                            const Real other) { return type{other * self}; })
        // Need __div__ for python 2 and __truediv__ for python 3.
        .def(
            "__div__", +[](const type& self,
                           const Real other) { return type{self / other}; })
        .def(
            "__truediv__", +[](const type& self,
                               const Real other) { return type{self / other}; })
        .def(
            "__rdiv__", +[](const type& self,
                            const Real other) { return type{other / self}; })
        .def(
            "__rtruediv__",
            +[](const type& self, const Real other) {
              return type{other / self};
            })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(
            "__neg__", +[](const type& t) { return type{-t}; });
  }
  //****************************************************************************

}  // namespace py_bindings
