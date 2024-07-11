#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/TypeTraits.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
//
#include "Systems/common/Python/Generators/VariableFilter.hpp"
//
#include "Utilities/ConvertCase/ConvertCase.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/PrettyType.hpp"
//
#include <pybind11/pybind11.h>
//
#include <type_traits>

namespace py_bindings {

  namespace detail {

    // TODO : concept check?
    template <typename UnwantedVariableTags, typename ModelsCosseratRodBlock>
    void bind_pyelastica_interface(
        pybind11::class_<ModelsCosseratRodBlock>& context) {
      using Arg = std::remove_reference_t<decltype(context)>;
      using type = typename Arg::type;

      using Variables = ::blocks::variables_t<type>;
      using VariablesToBeBound = tmpl::remove_if<
          Variables,
          tmpl::bind<tt::VariableFilter<UnwantedVariableTags>::template type,
                     tmpl::_1>>;

      namespace py = pybind11;

#define MAP(TAG, STR) [](TAG /*meta*/) -> std::string { return STR; }

      // Tag to method mapping
      auto tag_to_method_map = make_overloader(
          MAP(elastica::tags::NElement, "n_elems"),
          MAP(elastica::tags::Position, "position_collection"),
          MAP(elastica::tags::Velocity, "velocity_collection"),
          MAP(elastica::tags::Acceleration, "acceleration_collection"),
          MAP(elastica::tags::AngularVelocity, "omega_collection"),
          MAP(elastica::tags::AngularAcceleration, "alpha_collection"),
          MAP(elastica::tags::Director, "director_collection"),
          MAP(elastica::tags::ReferenceElementLength, "rest_lengths"),
          MAP(elastica::tags::Material,
              "material"),  // pyelastica has density instead
          MAP(elastica::tags::ElementVolume, "volume"),
          MAP(elastica::tags::Mass, "mass"),
          MAP(elastica::tags::MassSecondMomentOfInertia,
              "mass_second_moment_of_inertia"),
          MAP(elastica::tags::InvMassSecondMomentOfInertia,
              "inv_mass_second_moment_of_inertia"),
          MAP(elastica::tags::ForceDampingRate,
              "dissipation_constant_for_forces"),
          MAP(elastica::tags::TorqueDampingRate,
              "dissipation_constant_for_torques"),
          MAP(elastica::tags::ReferenceVoronoiLength, "rest_voronoi_lengths"),
          MAP(elastica::tags::InternalLoads, "internal_forces"),
          MAP(elastica::tags::InternalTorques, "internal_torques"),
          MAP(elastica::tags::ExternalLoads, "external_forces"),
          MAP(elastica::tags::ExternalTorques, "external_torques"),
          MAP(elastica::tags::ElementLength, "lengths"),
          MAP(elastica::tags::Tangent, "tangents"),
          MAP(elastica::tags::ElementDimension, "radius"),
          MAP(elastica::tags::ElementDilatation, "dilatation"),
          MAP(elastica::tags::VoronoiDilatation, "voronoi_dilatation"),
          // fallback for other tags that are not defined in PyElastica
          [](auto tag) -> std::string {
            namespace cc = convert_case;
            return cc::convert(pretty_type::short_name<decltype(tag)>(),
                               cc::FromPascalCase{}, cc::ToSnakeCase{});
          });
#undef MAP

      // first define variables as a read only property for named evaluation
      tmpl::for_each<VariablesToBeBound>([=, &context](auto v) {
        using Variable = tmpl::type_from<decltype(v)>;
        using VariableTag = ::blocks::parameter_t<Variable>;
        std::string help_str =
            "Refer to documentation of " + pretty_type::get_name<VariableTag>();
        context.def_property(
            tag_to_method_map(VariableTag{}).c_str(),
            +[](type& self) { return ::blocks::get<VariableTag>(self); },
            +[](type& self, typename Variable::slice_type value) {
              ::blocks::get<VariableTag>(self) = value;
            },
            help_str.c_str());
      });
    }

  }  // namespace detail

  //****************************************************************************
  /*!\brief Helps bind pyelastica rod interface to a \elastica CosseratRod
   * \ingroup python_bindings
   *
   * \details
   * \param rod Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<
          ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::BlockSlice, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind pyelastica rod interface to a \elastica CosseratRod
   * \ingroup python_bindings
   *
   * \details
   * \param rod Pybind11 class of a CosseratRod
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<::blocks::BlockSlice<
          ::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::BlockSlice, Tag, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind pyelastica interface to a CosseratRod blockview in
   * \elastica
   * \ingroup python_bindings
   * \see bind_tags
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlockView*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<
          ::blocks::BlockView<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::BlockView, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind pyelastica interface to a CosseratRod blockview in
   * \elastica
   * \ingroup python_bindings
   * \see bind_tags
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlockView*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<
          ::blocks::BlockView<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::BlockView, Tag, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind pyelastica interface to a CosseratRod block in \elastica
   * \ingroup python_bindings
   * \see bind_tags
   */
  template <typename UnwantedVariableTags, typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<
          ::blocks::Block<::elastica::cosserat_rod::CosseratRodPlugin<
              CRT, ::blocks::Block, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Helps bind pyelastica interface to a CosseratRod block in \elastica
   * \ingroup python_bindings
   * \see bind_tags
   */
  template <typename UnwantedVariableTags, typename CRT, typename Tag,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  void bind_pyelastica_interface(
      pybind11::class_<
          ::blocks::Block<::elastica::cosserat_rod::TaggedCosseratRodPlugin<
              CRT, ::blocks::Block, Tag, Components...>>>& rod) {
    detail::bind_pyelastica_interface<UnwantedVariableTags>(rod);
  }
  //****************************************************************************

}  // namespace py_bindings
