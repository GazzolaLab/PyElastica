//******************************************************************************
// Includes
//******************************************************************************

//
#include <pybind11/pybind11.h>
//
#include "Systems/Systems.hpp"
//
#include "Systems/Python/BindTags.hpp"
//
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace py_bindings {

  //****************************************************************************
  /*!\brief Helps bind tags for all physical systems in \elastica
   * \ingroup python_bindings
   */
  void bind_tags(py::module& m) {  // NOLINT
    bind_tags<::elastica::PhysicalSystemPlugins>(m);
  }
  //****************************************************************************

}  // namespace py_bindings
