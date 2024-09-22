//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>
#include <iostream>
#include <pybind11/pybind11.h>
#include <string>
#include <utility>
//
//

//

#include "Systems/Block/Block.hpp"  // slice()
#include "Systems/CosseratRods.hpp"
#include "Systems/CosseratRods/CosseratRods.hpp"
// #include "Simulator/Simulator.hpp"
//
// #include "Systems/CosseratRods/Python/Generators/BindEnergyMethods.hpp"
#include "Systems/CosseratRods/Python/Generators/BindMemberAccess.hpp"
#include "Systems/CosseratRods/Python/Generators/BindPyElasticaInterface.hpp"
#include "Systems/CosseratRods/Python/Generators/BoundChecks.hpp"
#include "Systems/common/Python/Generators/BindVariableLocator.hpp"
//
#include "Utilities/Math/Python/SliceHelpers.hpp"
#include "Utilities/Size.hpp"
#include "Utilities/TMPL.hpp"
// #include "Utilities/Get.hpp"

namespace py = pybind11;

namespace py_bindings {

  using StraightInit = ::elastica::cosserat_rod::StraightCosseratRodInitializer;
  using Plugin = ::elastica::cosserat_rod::CosseratRod;
  using T = ::elastica::cosserat_rod::detail::CosseratRodBlock;

  using UnwantedVariableTags =
      tmpl::list<::elastica::tags::_DummyElementVector,
                 ::elastica::tags::_DummyElementVector2,
                 ::elastica::tags::_DummyVoronoiVector>;

  /*!\brief Helps bind a cosserat rod block and slice in \elastica
   * \ingroup python_bindings
   */
  void bind_tests(py::module& m) {  // NOLINT

    auto py_cosserat_rod =
        py::class_<T>(m, "_CosseratRodBlock")
            .def(py::init([](std::size_t n_elems) {
                   StraightInit straight_initializer{
                       // StraightInitialization Wrapper
                       StraightInit::NElement{n_elems},
                       StraightInit::Density{1.0},
                       StraightInit::Youngs{1.00e6},
                       StraightInit::ShearModulus{0.33e6},
                       StraightInit::Radius{0.1},
                       StraightInit::Length{1.0},
                       StraightInit::Origin{::elastica::Vec3{0.0, 0.0, 0.0}},
                       StraightInit::Direction{::elastica::Vec3{1.0, 0.0, 0.0}},
                       StraightInit::Normal{::elastica::Vec3{0.0, 1.0, 0.0}}};
                   T result{};
                   auto block_slice = result.emplace_back(
                       straight_initializer
                           .initialize<Plugin>());  // block slice
                   return result;
                 }),
                 py::arg("n_elems"))
            .def("__len__", [](const T& t) { return cpp17::size(t); })
            .def(
                "__str__",
                +[](const T& /*meta*/) {
                  return std::string("CosseratRodView");
                })
            .def(
                "__getitem__",
                +[](T& t, std::size_t index) {
                  variable_bounds_check(::blocks::n_units(t), index);
                  return ::blocks::slice(t, index);
                })
            .def(
                "__getitem__", +[](T& t, const py::slice slice) {
                  auto v =
                      check_slice<0UL>(::blocks::n_units(t), std::move(slice));
                  return ::blocks::slice(t, v.start, v.start + v.slicelength);
                });
    bind_member_access<UnwantedVariableTags>(py_cosserat_rod);
    bind_variable_locator<UnwantedVariableTags>(py_cosserat_rod);
    bind_pyelastica_interface<UnwantedVariableTags>(py_cosserat_rod);
  }
  //****************************************************************************

}  // namespace py_bindings

PYBIND11_MODULE(_PyExamples, m) {  // NOLINT
  m.doc() = R"pbdoc(
    Bindings for Elastica++ Test codes
    )pbdoc";
  py_bindings::bind_tests(m);
}
