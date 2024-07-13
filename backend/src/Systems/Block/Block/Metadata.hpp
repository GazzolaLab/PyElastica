#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/Types.hpp"
// #include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdVectorHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/WidthStream.hpp"  // TODO : move into CPP file
//
#include <ostream>
#include <string>
#include <algorithm> // transform
#include <iterator> // move iterator
#include <utility>  // move, pair
#include <vector>

namespace blocks {

  //****************************************************************************
  /*!\brief Variable metadata
   * \ingroup blocks
   *
   * Carries meta-data information about variables in a block plugin
   */
  struct VariableMetadata {
    using VariableAttribute = std::pair<std::string, std::string>;
    using VariableAttributes = std::vector<VariableAttribute>;
    VariableAttributes data;
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Plugin metadata
   * \ingroup blocks
   *
   * Carries meta-data information about  plugins
   */
  struct PluginMetadata {
    using Metadata = std::vector<VariableMetadata>;
    //! Name of plugin
    std::string name;
    //! Metadata associated with the plugin
    Metadata data;
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Customization point for generating metadata of plugins
   * \ingroup blocks
   *
   * \details
   * Metadata provides the customization point for specifying the
   * behavior of blocks::metadata() for any `Plugin` type. By default, \elastica
   * generates basic metadata about plugin variables : Metadata can be
   * customized to add more attributes to this metadata.
   * If no customized implementation is provided, \elastica falls back to this
   * default Metadata class, which does a no-operation.
   *
   * \usage
   * The no-operation above uses the static apply() function templated on the
   * variable
   * \code
   * auto attributes = Metadata<Plugin>::template apply<Variable>(os);
   * \endcode
   *
   * \section customization Metadata customization
   * As seen from the example above, the Metadata class needs to
   * define a static apply() function with the following signature
   * \snippet this apply_signature
   *
   * Hence to customize Metadata for your own plugin, we rely on template
   * specialization, typically done in the translation unit where the Plugin
   * type is defined. As an example of implementing Metadata for a
   * custom plugin, consider the following example
   *
   * \example
   * With the setup for Plugin shown below
   * \snippet Test_Metadata.cpp customized_plugin
   * the following code demonstrates the customization of Metadata
   * \snippet Test_Metadata.cpp customization_for_plugin
   *
   * \tparam Plugin Plugin for which blocks::metadata() need to be
   * customized
   *
   * \see Block, blocks::protocols::Plugin, blocks::Variable, blocks::metadata()
   */
  template <typename Plugin>
  struct Metadata {
    /// [apply_signature]
    // VariableAttributes is a std::vector<std::pair<std::string, std::string>>
    template <typename Variable>
    static inline auto apply() noexcept ->
        typename VariableMetadata::VariableAttributes {
      return {};
    }
    /// [apply_signature]
  };
  //****************************************************************************

  namespace detail {

    using VariableAttributeCollection =
        std::vector<typename VariableMetadata::VariableAttributes>;

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename Var>
    auto collect_attributes(tmpl::type_<Var>) {
      typename VariableMetadata::VariableAttributes attributes{};

      attributes.emplace_back("Name", Var::name()); //pretty_type::name<Var>());
      // attributes.emplace_back("Parameter",
      //                         pretty_type::get_name<parameter_t<Var>>());
      // attributes.emplace_back(
      //     "Initialized",
      //     tt::is_detected_v<initializer_t, Var> ? "true" : "false");
      // attributes.emplace_back("Rank", pretty_type::get_name<rank_t<Var>>());
      // attributes.emplace_back("Type",
      //                         pretty_type::get_name<typename Var::type>());
      return attributes;
    }

    template <typename Plugin, Requires<is_plugin_v<Plugin>> = nullptr>
    auto collect_attributes() -> VariableAttributeCollection {
      using Variables = variables_t<Plugin>;
      using CustomizationPoint = Metadata<Plugin>;

      VariableAttributeCollection metadata;

      tmpl::for_each<Variables>([&metadata](auto var) {
        auto attributes = collect_attributes(var);
        append(attributes, CustomizationPoint::template apply<
                               tmpl::type_from<decltype(var)>>());
        metadata.emplace_back(std::move(attributes));
      });

      return metadata;
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //============================================================================
  //
  //  FREE FUNCTIONS
  //
  //============================================================================

  // TODO : Move to CPP file
  //****************************************************************************
  /*!\brief Stream operator for VariableMetaData
   */
  inline std::ostream& operator<<(std::ostream& os,
                                  VariableMetadata const& metadata) {
    // This is a very simple YAML formatting.
    // For a better one, see YAML Archive
    for (auto const& data : metadata.data) {
      os << data.first << ": "
         << "\"" << data.second << "\""
         << "\n";
    }
    return os;
  }
  //****************************************************************************

  // TODO : Move to CPP file
  //****************************************************************************
  /*!\brief Stream operator for PluginMetaData
   */
  inline std::ostream& operator<<(std::ostream& os,
                                  PluginMetadata const& metadata) {
    // This is a very simple YAML formatting.
    // For a better one, see YAML Archive
    constexpr std::size_t line_length = 120UL;
    ::elastica::widthstream stream{line_length, os};

    stream << "Plugin: " << metadata.name << "\n";
    const std::string information_key("Variables:");

    stream << information_key
           << (metadata.data.empty() ? " No variables!" : "\n");

    int constexpr tab_width = 2;
    stream.indent(tab_width);

    for (auto const& data : metadata.data) {
      stream << "- "
             << "\n";
      stream.indent(tab_width);
      stream << data << "\n";
      stream.indent(-tab_width);
    }

    stream.indent(-tab_width);
    return os;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Generates variable information associated with a Plugin
   * \ingroup blocks
   *
   * \example
   * \code
   * using Plugin = MyCustomPlugin;
   * auto meta = blocks::metadata<Plugin>();
   * \endcode
   *
   * \tparam Plugin A valid plugin template conforming to protocols::Plugin
   */
  template <typename Plugin>
  auto metadata() -> PluginMetadata {
    auto raw_metadata = detail::collect_attributes<Plugin>();
    // PluginMetadata metadata{pretty_type::name<Plugin>(), {}};
    PluginMetadata metadata{Plugin::name(), {}};
    metadata.data.reserve(raw_metadata.size());

    // Convert vec<vec<attributes>> -> vec<VariableMetadata>
    std::transform(std::make_move_iterator(std::begin(raw_metadata)),
                   std::make_move_iterator(std::end(raw_metadata)),
                   std::back_inserter(metadata.data),
                   [](typename VariableMetadata::VariableAttributes&& meta) {
                     return VariableMetadata{std::move(meta)};
                   });
    return metadata;
  }
  //****************************************************************************

  //**Metadata functions********************************************************
  /*!\name Metadata functions */
  //@{

  //****************************************************************************
  /*!\brief Generates variable information associated with a type modeling the
   * block concept
   * \ingroup blocks
   *
   * \example
   * \code
   * auto block_like = ...; // Models the block concept
   * auto meta = blocks::metadata<Plugin>(block_like);
   * \endcode
   *
   * \tparam Plugin A valid plugin template conforming to protocols::Plugin
   */
  template <typename Plugin>
  auto metadata(Block<Plugin> const&) -> PluginMetadata {
    return metadata<Plugin>();
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  auto metadata(BlockSlice<Plugin> const&) -> PluginMetadata {
    return metadata<Plugin>();
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  auto metadata(ConstBlockSlice<Plugin> const&) -> PluginMetadata {
    return metadata<Plugin>();
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  auto metadata(BlockView<Plugin> const&) -> PluginMetadata {
    return metadata<Plugin>();
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Plugin>
  auto metadata(ConstBlockView<Plugin> const&) -> PluginMetadata {
    return metadata<Plugin>();
  }
  //****************************************************************************

  //@}
  //****************************************************************************

  namespace detail {

    // TODO : Move to CPP file
    inline auto form_map(PluginMetadata metadata) {
      using AttributeKey = std::string;
      using Name = std::string;
      using InnerMapType = std::unordered_map<AttributeKey, std::string>;
      using ReturnType = std::unordered_map<Name, InnerMapType>;
      ReturnType um;

      for (auto& vm : metadata.data) {
        auto& dst = um[(*vm.data.begin()).second];
        dst.insert(std::make_move_iterator(vm.data.begin() + 1),
                   std::make_move_iterator(vm.data.end()));
      }

      return um;
    }

  }  // namespace detail

}  // namespace blocks
