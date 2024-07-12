#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/TypeTraits.hpp"
//
#include "Systems/common/Python/Generators/VariableFilter.hpp"
#include "Systems/common/Python/Generators/VariableLocator.hpp"
//
#include "Utilities/PrettyType.hpp"
//
#include <pybind11/pybind11.h>
//
#include <type_traits>

namespace py_bindings {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  // TODO : concept check?
  template <typename UnwantedVariableTags>
  struct BindVariableLocator {
    using This = BindVariableLocator<UnwantedVariableTags>;

    template <typename ModelsBlock>
    static void bind_variable_tags_as_property(
        pybind11::class_<ModelsBlock>& context) {
      using Arg = std::remove_reference_t<decltype(context)>;
      using type = typename Arg::type;

      using Variables = ::blocks::variables_t<type>;
      using VariablesToBeBound = tmpl::remove_if<
          Variables,
          tmpl::bind<tt::VariableFilter<UnwantedVariableTags>::template type,
                     tmpl::_1>>;

      namespace py = pybind11;

      // the tags are registered globally in the systems.tags module
      auto register_property = [](auto& bound_class_, auto tag) {
        using VariableTag = decltype(tag);
        std::string tag_name = pretty_type::short_name<VariableTag>();
        bound_class_.def_property_readonly_static(
            tag_name.c_str(),
            [](py::object /* self */) { return VariableTag{}; });
      };

      // first define variables as a read only property for named evaluation
      tmpl::for_each<VariablesToBeBound>([=, &context](auto v) {
        using Variable = tmpl::type_from<decltype(v)>;
        using VariableTag = ::blocks::parameter_t<Variable>;
        register_property(context, VariableTag{});
      });
    }

    template <typename ModelsBlock>
    static void bind_locator(pybind11::class_<ModelsBlock>& context) {
      using Arg = std::remove_reference_t<decltype(context)>;
      using type = typename Arg::type;
      using Locator = VariableLocator<type>;

      using Variables = ::blocks::variables_t<type>;
      using VariablesToBeBound = tmpl::remove_if<
          Variables,
          tmpl::bind<tt::VariableFilter<UnwantedVariableTags>::template type,
                     tmpl::_1>>;

      namespace py = pybind11;

      // Define getters and setters for loc_indexer
      auto indexer = py::class_<Locator>(context, "VariableLocator");

      indexer
          .def(
              "__str__", +[](Locator const&) { return "Variable indexer"; })
          .def(
              "__repr__", +[](Locator const&) { return "Variable indexer"; });

      // first define variables as a read only property for named evaluation
      tmpl::for_each<VariablesToBeBound>([&indexer](auto v) {
        using Variable = tmpl::type_from<decltype(v)>;
        using VariableTag = ::blocks::parameter_t<Variable>;

        indexer
            .def(
                "__getitem__",
                +[](Locator& t, VariableTag /*meta*/) {
                  return ::blocks::get<VariableTag>(t.data());
                },
                py::arg("tag"))
            .def(
                "__setitem__", +[](Locator& t, VariableTag /*meta*/,
                                   typename Variable::type value) {
                  ::blocks::get<VariableTag>(t.data()) = value;
                });
      });

      constexpr const char* variable_loc_doc = R"doc(
Access a variable by its tag. Similar in utility to pandas.DataFrame.loc from
the pandas module. Allowed tags are accessed from the context.
)doc";

      // Loc indexer is bound to class as a loc[] method
      context.def_property_readonly(
          "loc", [](type& self) { return Locator{self}; }, variable_loc_doc,
          // Keep object alive while indexer exists
          py::keep_alive<0, 1>());
    }

    template <typename ModelsBlock>
    static void apply(pybind11::class_<ModelsBlock>& system) {
      This::bind_variable_tags_as_property(system);
      This::bind_locator(system);
    }
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind tags to a CosseratRod in \elastica
   * \ingroup python_bindings
   *
   * \details
   * Tags are bound as static properties and can be used to obtain member
   * functions using the `loc` method, similar to pandas. The typical use is
   * to process multiple data-members simultaneously
   *
   * \tparam UnwantedVariableTags Tags that should not be bound/exposed to the
   * python side
   * \param a Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename Plugin>
  void bind_variable_locator(
      pybind11::class_<::blocks::BlockSlice<Plugin>>& system) {
    BindVariableLocator<UnwantedVariableTags>::apply(system);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind tags to a system BlockView in \elastica
   * \ingroup python_bindings
   * \see bind_variable_locator
   */
  template <typename UnwantedVariableTags, typename Plugin>
  void bind_variable_locator(
      pybind11::class_<::blocks::BlockView<Plugin>>& system) {
    BindVariableLocator<UnwantedVariableTags>::apply(system);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind tags to a system block in \elastica
   * \ingroup python_bindings
   * \see bind_variable_locator
   */
  template <typename UnwantedVariableTags, typename Plugin>
  void bind_variable_locator(
      pybind11::class_<::blocks::Block<Plugin>>& system) {
    BindVariableLocator<UnwantedVariableTags>::apply(system);
  }
  //****************************************************************************

}  // namespace py_bindings
