#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Block.hpp"
#include "Systems/Block/TypeTraits.hpp"
//
// TODO : refactor rank, and traits should belong to systems common
#include "Systems/CosseratRods/Traits/DataType/Rank.hpp"
//
#include "Systems/CosseratRods/common/Generators/BoundChecks.hpp"
#include "Systems/CosseratRods/common/Generators/VariableFilter.hpp"
//
#include "Utilities/ConvertCase/ConvertCase.hpp"
#include "Utilities/IgnoreUnused.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"
//
#include <pybind11/pybind11.h>
//
#include <string>

namespace py_bindings {

  namespace detail {
    // TODO : concept check?
    template <typename UnwantedVariableTags, typename ModelsBlock>
    void bind_member_access(pybind11::class_<ModelsBlock>& system) {
      using type = ModelsBlock;
      using Variables = ::blocks::variables_t<type>;
      using VariablesToBeBound = tmpl::remove_if<
          Variables,
          tmpl::bind<tt::VariableFilter<UnwantedVariableTags>::template type,
                     tmpl::_1>>;

      using ::elastica::cosserat_system::Rank;
      auto rank_doc_generator = make_overloader(
          [](Rank<1U> /*scalar*/) -> std::string { return "1D"; },
          [](Rank<2U> /*vector*/) -> std::string { return "2D"; },
          [](Rank<3U> /*tensor*/) -> std::string { return "3D"; });

      auto dim_doc_generator = make_overloader(
          [](Rank<1U> /*scalar*/) -> std::string { return ""; },
          [](Rank<2U> /*vector*/) -> std::string { return "3, "; },
          [](Rank<3U> /*tensor*/) -> std::string { return "3, 3, "; });

      // __getitem__ and __setitem__ are the subscript operators (M[*,*]).
      tmpl::for_each<VariablesToBeBound>([=, &system](auto v) {
        namespace cc = convert_case;
        using Variable = tmpl::type_from<decltype(v)>;
        using VariableTag = ::blocks::parameter_t<Variable>;
        using VariableRank = ::blocks::rank_t<Variable>;
        // system, node, element, voronoi
        using VariableStagger = typename Variable::Stagger;
        // Rank in case of a CosseratRod is Vector, Matrix, or Tensor, each of
        // which has a rank type-def, which we query

        std::string variable_name = pretty_type::short_name<Variable>();
        // cam use
        std::string tag_name = pretty_type::get_name<VariableTag>();
        std::string method_prefix = "get_";
        std::string method_name =
            method_prefix +
            cc::convert(variable_name, cc::FromPascalCase{}, cc::ToSnakeCase{});

        /*
         * Generate a docstring with the format:
         *
         * Gets the variable corresponding to the C++ tag
         * ::elastica::tags::AngularAcceleration. Returns a 2D (3, ) array
         * containing data of double type. See documentation of
         * AngularAcceleration for more details.
         */
        std::string method_doc =
            (MakeString{} << "Gets the variable corresponding to the C++ tag "
                          << tag_name << ". Returns a "
                          << rank_doc_generator(typename VariableRank::rank{})
                          // clang-format off
                              << " (" << dim_doc_generator(typename VariableRank::rank{}) << ") "
                          // clang-format on
                          << "array containing data of "
                          << pretty_type::short_name<::elastica::real_t>()
                          << " type. See documentation of " << tag_name
                          << " for more details.");

        system.def(
            method_name.c_str(),
            +[](type& self) {
              // roundabout
              return ::blocks::get<VariableTag>(self);
            },
            method_doc.c_str());
      });
    }

  }  // namespace detail

  //****************************************************************************
  /*!\brief Helps bind accessors to a CosseratRod in \elastica
   * \ingroup python_bindings
   *
   * \details
   * Binds member functions to access system members.The convention for the
   * member function is `get_{name}` where `{name}` is the
   * tag name in small letters. For example, to access the variable
   * elastica::tags::Position above, you can invoke the `get_position()` member
   * function from Python.
   *
   * \tparam UnwantedVariableTags Tags that should not be bound/exposed to the
   * python side
   * \param a Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename Plugin>
  void bind_member_access(
      pybind11::class_<::blocks::BlockSlice<Plugin>>& system) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(system);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a system block in \elastica
   * \ingroup python_bindings
   * \see bind_member_access
   */
  template <typename UnwantedVariableTags, typename Plugin>
  void bind_member_access(
      pybind11::class_<::blocks::Block<Plugin>>& system) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(system);
  }
  //****************************************************************************

}  // namespace py_bindings
