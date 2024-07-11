#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/TypeTraits.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
#include "Systems/common/Traits/DataType/Rank.hpp"
#include "Systems/CosseratRods/Traits/PlacementTraits/PlacementTraits.hpp"
//
#include "Systems/CosseratRods/Python/Generators/BoundChecks.hpp"
#include "Systems/common/Python/Generators/VariableFilter.hpp"
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
    void bind_member_access(pybind11::class_<ModelsBlock>& rod) {
      using type = ModelsBlock;
      using Variables = ::blocks::variables_t<type>;
      using VariablesToBeBound = tmpl::remove_if<
          Variables,
          tmpl::bind<tt::VariableFilter<UnwantedVariableTags>::template type,
                     tmpl::_1>>;

      using ::elastica::cosserat_rod::Rank;
      auto rank_doc_generator = make_overloader(
          [](Rank<1U> /*scalar*/) -> std::string { return "1D"; },
          [](Rank<2U> /*vector*/) -> std::string { return "2D"; },
          [](Rank<3U> /*tensor*/) -> std::string { return "3D"; });

      auto dim_doc_generator = make_overloader(
          [](Rank<1U> /*scalar*/) -> std::string { return ""; },
          [](Rank<2U> /*vector*/) -> std::string { return "3, "; },
          [](Rank<3U> /*tensor*/) -> std::string { return "3, 3, "; });

      namespace pt = ::elastica::cosserat_rod::placement_tags;
      auto stagger_doc_generator = make_overloader(
          [](pt::OnNode /*scalar*/) -> std::string { return "n_node"; },
          [](pt::OnElement /*vector*/) -> std::string { return "n_elem"; },
          [](pt::OnVoronoi /*tensor*/) -> std::string { return "n_voronoi"; },
          [](pt::OnRod /*tensor*/) -> std::string { return "1"; });

      // __getitem__ and __setitem__ are the subscript operators (M[*,*]).
      tmpl::for_each<VariablesToBeBound>([=, &rod](auto v) {
        namespace cc = convert_case;
        using Variable = tmpl::type_from<decltype(v)>;
        using VariableTag = ::blocks::parameter_t<Variable>;
        using VariableRank = ::blocks::rank_t<Variable>;
        // rod, node, element, voronoi
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
         * ::elastica::tags::AngularAcceleration. Returns a 2D (3, n_elem)
         array
         * containing data of double type. See documentation of
         * AngularAcceleration for more details.

         */
        std::string method_doc =
            (MakeString{} << "Gets the variable corresponding to the C++ tag "
                          << tag_name << ". Returns a "
                          << rank_doc_generator(typename VariableRank::rank{})
                          // clang-format off
                              << " (" << dim_doc_generator(typename VariableRank::rank{}) << stagger_doc_generator(VariableStagger{}) << ") "
                          // clang-format on
                          << "array containing data of "
                          // TODO: Refactor : this relies on blaze interface
                          << pretty_type::short_name<
                                 typename VariableRank::type::ElementType>()
                          << " type. See documentation of " << tag_name
                          << " for more details.");

        rod.def(
            method_name.c_str(),
            +[](type& self) {
              // roundabout
              return ::blocks::get<VariableTag>(self);
            },
            method_doc.c_str());

        // Defines the slice method, can be refactored into separate binder
        std::string slice_method_name = method_name + "_slice";

        std::string slice_method_doc =
            (MakeString{}
             << "Gets a slice of the variable corresponding to the C++ tag "
             << tag_name << ". Returns a "
             << rank_doc_generator(typename VariableRank::rank{})
             // clang-format off
                    << " (" << dim_doc_generator(typename VariableRank::rank{}) <<  ") "
             // clang-format on
             << "slice containing data of "
             // TODO: Refactor : this relies on blaze interface
             << pretty_type::short_name<
                    typename VariableRank::type::ElementType>()
             << " type. See documentation of " << tag_name
             << " for more details.");

        auto slice_method = +[](type& self, std::size_t index) {
          // roundabout
          variable_bounds_check(Variable::get_dofs(self.size()), index);
          return Variable::slice(::blocks::get<VariableTag>(self), index);
        };
        // overload
        rod.def(method_name.c_str(), slice_method, slice_method_doc.c_str());

        // seems overkill without adding anything
        elastica::ignore_unused(slice_method_name);
        // rod.def(slice_method_name.c_str(), slice_method,
        //         slice_method_doc.c_str());
      });
    }

  }  // namespace detail

  //****************************************************************************
  /*!\brief Helps bind accessors to a CosseratRod in \elastica
   * \ingroup python_bindings
   *
   * \details
   * Binds member functions to access rod members.The convention for the
   * member function is `get_{name}` where `{name}` is the
   * tag name in small letters. For example, to access the variable
   * elastica::tags::Position above, you can invoke the `get_position()` member
   * function from Python.
   *
   * \tparam UnwantedVariableTags Tags that should not be bound/exposed to the
   * python side
   * \param a Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<
          ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::BlockSlice, Components...>>>& rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a CosseratRod in \elastica
   * \ingroup python_bindings
   *
   * \details
   * Binds member functions to access rod members.The convention for the
   * member function is `get_{name}` where `{name}` is the
   * tag name in small letters. For example, to access the variable
   * elastica::tags::Position above, you can invoke the `get_position()` member
   * function from Python.
   *
   * \tparam UnwantedVariableTags Tags that should not be bound/exposed to the
   * python side
   * \param a Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<::blocks::BlockSlice<
          ::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::BlockSlice, Tag, Components...>>>&
          rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a BlockView of CosseratRod in \elastica
   * \ingroup python_bindings
   * \see bind_member_access
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlockView*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<
          ::blocks::BlockView<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::BlockView, Components...>>>& rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a BlockView of TaggedCosseratRod
   * \ingroup python_bindings
   * \see bind_member_access
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlockView*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<
          ::blocks::BlockView<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::BlockView, Tag, Components...>>>& rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a block of CosseratRod in \elastica
   * \ingroup python_bindings
   * \see bind_member_access
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<
          ::blocks::Block<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::Block, Components...>>>& rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind accessors to a block of CosseratRod in \elastica
   * \ingroup python_bindings
   * \see bind_member_access
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_member_access(
      pybind11::class_<
          ::blocks::Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::Block, Tag, Components...>>>& rod) {  // NOLINT
    detail::bind_member_access<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

}  // namespace py_bindings
