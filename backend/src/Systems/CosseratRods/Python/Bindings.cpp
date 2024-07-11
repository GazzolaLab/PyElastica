//******************************************************************************
// Includes
//******************************************************************************
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
  void bind_cosserat_rod(py::module& m);                  // NOLINT
  void bind_cosserat_rod_without_damping(py::module& m);  // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyCosseratRods, m) {  // NOLINT
  // TODO : detailed documentation
  m.doc() = R"pbdoc(
    Bindings for Elastica++ CosseratRod types
    )pbdoc";
  // Experimental : what are the drawbacks
  py::module::import("elasticapp.Arrays");
  py::module::import("elasticapp.Systems.Tags");
  //
  py_bindings::bind_cosserat_rod(m);
  py_bindings::bind_cosserat_rod_without_damping(m);
}
