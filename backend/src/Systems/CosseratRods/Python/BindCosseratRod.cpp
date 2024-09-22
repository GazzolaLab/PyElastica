//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>
#include <string>
#include <utility>
//
#include <pybind11/pybind11.h>
//

//
#include "Systems/Block/Block.hpp"  // slice()
#include "Systems/CosseratRods/CosseratRods.hpp"
//
// #include "Systems/CosseratRods/Python/Generators/BindEnergyMethods.hpp"
#include "Systems/CosseratRods/Python/Generators/BindMemberAccess.hpp"
#include "Systems/CosseratRods/Python/Generators/BoundChecks.hpp"
#include "Systems/CosseratRods/Python/Generators/BindPyElasticaInterface.hpp"
#include "Systems/common/Python/Generators/BindVariableLocator.hpp"
//
#include "Utilities/Math/Python/SliceHelpers.hpp"
#include "Utilities/Size.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace py_bindings {

  //****************************************************************************
  /*!\brief Helps bind a cosserat rod block and slice in \elastica
   * \ingroup python_bindings
   */
  void bind_cosserat_rod(py::module& m) {  // NOLINT
    using UnwantedVariableTags =
        tmpl::list<::elastica::tags::_DummyElementVector,
                   ::elastica::tags::_DummyElementVector2,
                   ::elastica::tags::_DummyVoronoiVector>;

    {
      using type = ::elastica::cosserat_rod::detail::CosseratRodBlockSlice;

      // Wrapper for basic type operations
      auto py_cosserat_rod =
          py::class_<type>(m, "CosseratRod")
              .def("__len__", [](const type& t) { return cpp17::size(t); })
              .def(
                  "__str__", +[](const type& /*meta*/) {
                    return std::string("CosseratRod");
                  });
      bind_member_access<UnwantedVariableTags>(py_cosserat_rod);
      bind_variable_locator<UnwantedVariableTags>(py_cosserat_rod);
      bind_pyelastica_interface<UnwantedVariableTags>(py_cosserat_rod);
      // bind_energy_methods(py_cosserat_rod);
    }
    {
      using type = ::elastica::cosserat_rod::detail::CosseratRodBlockView;

      // Wrapper for basic type operations
      auto py_cosserat_rod =
          py::class_<type>(m, "CosseratRodView")
              .def("__len__", [](const type& t) { return cpp17::size(t); })
              .def(
                  "__str__",
                  +[](const type& /*meta*/) {
                    return std::string("CosseratRodView");
                  })
              .def(
                  "__getitem__",
                  +[](type& t, std::size_t index) {
                    variable_bounds_check(::blocks::n_units(t), index);
                    return ::blocks::slice(t, index);
                  })
              .def(
                  "__getitem__", +[](type& t, const py::slice slice) {
                    auto v = check_slice<0UL>(::blocks::n_units(t),
                                              std::move(slice));
                    return ::blocks::slice(t, v.start, v.start + v.slicelength);
                  });
      bind_member_access<UnwantedVariableTags>(py_cosserat_rod);
      bind_variable_locator<UnwantedVariableTags>(py_cosserat_rod);
      bind_pyelastica_interface<UnwantedVariableTags>(py_cosserat_rod);
    }
  }
  //****************************************************************************

}  // namespace py_bindings
