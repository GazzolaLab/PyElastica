//******************************************************************************
// Includes
//******************************************************************************
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
  void bind_tags(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyTags, m) {  // NOLINT
  m.doc() = R"pbdoc(
    Bindings for Elastica++ tag types
    )pbdoc";
  py_bindings::bind_tags(m);
}
