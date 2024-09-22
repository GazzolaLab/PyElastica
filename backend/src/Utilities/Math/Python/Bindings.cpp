//******************************************************************************
// Includes
//******************************************************************************
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace py_bindings {
  void bind_blaze_vector(py::module& m);           // NOLINT
  // void bind_blaze_index_vector(py::module& m);     // NOLINT
  // void bind_blaze_subvector(py::module& m);        // NOLINT
  // void bind_blaze_index_subvector(py::module& m);  // NOLINT
  void bind_blaze_matrix(py::module& m);           // NOLINT
  // void bind_blaze_submatrix(py::module& m);        // NOLINT
  void bind_blaze_tensor(py::module& m);           // NOLINT
  // void bind_blaze_subtensor(py::module& m);        // NOLINT
}  // namespace py_bindings

PYBIND11_MODULE(_PyArrays, m) {  // NOLINT
  m.doc() = R"pbdoc(
    Bindings for Elastica++ array types
    )pbdoc";
  py_bindings::bind_blaze_vector(m);
  // py_bindings::bind_blaze_index_vector(m);
  // py_bindings::bind_blaze_subvector(m);
  // py_bindings::bind_blaze_index_subvector(m);
  py_bindings::bind_blaze_matrix(m);
  // py_bindings::bind_blaze_submatrix(m);
  py_bindings::bind_blaze_tensor(m);
  // py_bindings::bind_blaze_subtensor(m);
}
