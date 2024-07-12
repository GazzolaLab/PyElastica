#pragma once

#include <tuple>

// #include "Options/Options.hpp"
// #include "Options/Parameter.hpp"
#include "Simulator/Materials.hpp"
#include "Systems/CosseratRods/Tags.hpp"
///
#include "Systems/CosseratRods/Initializers/detail/Transformations.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      namespace option_tags {

        // struct Name : Options::Parameter<std::size_t> {
        //   static constexpr Options::String help = {
        //       "Number of elements in the cosserat rod"};
        //   static type lower_bound() noexcept { return 2UL; }
        //   using TagType = tags::NElement;
        // };

        // struct NElement : public Options::Parameter<std::size_t> {
        //   using P = Options::Parameter<std::size_t>;
        //   using P::P;
        //   static constexpr Options::String help = {
        //       "Number of elements in the cosserat rod"};
        // checks for <= for now, so 2 is still valid
        struct NElement {
          using type = std::size_t;
          static type lower_bound() noexcept { return 1UL; }
          using TagType = ::elastica::tags::NElement;
        };

        // can be used for torque damping as well
        // struct Material : Options::Parameter<::elastica::MaterialID> {
        //   using P = Options::Parameter<::elastica::MaterialID>;
        //   using P::P;
        //   static constexpr Options::String help = {
        //       "Material of the cosserat rod"};
        struct Material {
          using type = ::elastica::MaterialID;
          using TagType = ::elastica::tags::Material;
        };

      }  // namespace option_tags

      class CommonInitializerToAllCosseratRods {
       public:
        using NElement = option_tags::NElement;
        using Material = option_tags::Material;
        using RequiredParameters = tmpl::list<NElement, Material>;
        using DefaultParameters = tmpl::list<>;

       protected:
        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& options_cache) const /*noexcept*/ {
          using Tag = NElement;
          check_lower_bounds(options_cache, tmpl::list<Tag>{});
          // Make a special concession for n elements alone?!
          return std::make_tuple(
              blocks::initialize<typename Tag::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<Tag>(options_cache).value())](...) ->
                  typename Tag::type { return value; }),
              blocks::initialize<typename Material::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<Material>(options_cache).value())](...) ->
                  typename Material::type { return value; }));
        }
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
