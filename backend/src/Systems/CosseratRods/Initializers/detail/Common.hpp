#pragma once

#include <tuple>

// #include "Options/Options.hpp"
// #include "Options/Parameter.hpp"
#include "Simulator/Materials.hpp"
#include "Systems/CosseratRods/Initializers/Parameter.hpp"
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
        struct NElement : public elastica::Parameter<std::size_t> {
          using elastica::Parameter<std::size_t>::Parameter;

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
        // struct Material {
        //   using type = ::elastica::MaterialID;
        //   using TagType = ::elastica::tags::Material;
        // };

        struct Density : elastica::Parameter<elastica::real_t> {
          using elastica::Parameter<elastica::real_t>::Parameter;
          using type = elastica::real_t;
          using TagType = ::elastica::tags::Density;
        };

        struct Youngs : elastica::Parameter<elastica::real_t> {
          using elastica::Parameter<elastica::real_t>::Parameter;
          using type = elastica::real_t;
          using TagType = ::elastica::tags::Youngs;
        };

        struct ShearModulus : elastica::Parameter<elastica::real_t> {
          using elastica::Parameter<elastica::real_t>::Parameter;
          using type = elastica::real_t;
          using TagType = ::elastica::tags::ShearModulus;
        };

      }  // namespace option_tags

      class CommonInitializerToAllCosseratRods {
       public:
        using NElement = option_tags::NElement;
        using Density = option_tags::Density;
        using Youngs = option_tags::Youngs;
        using ShearModulus = option_tags::ShearModulus;
        // using Material = option_tags::Material;
        using RequiredParameters =
            tmpl::list<NElement, Density, Youngs, ShearModulus>;
        using DefaultParameters = tmpl::list<>;

        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& options_cache) const /*noexcept*/ {
          check_lower_bounds(options_cache, tmpl::list<NElement>{});
          // Make a special concession for n elements alone?!
          return std::make_tuple(
              blocks::initialize<typename NElement::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<NElement>(options_cache).value())](...) ->
                  typename NElement::type { return value; }),
              blocks::initialize<typename Density::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<Density>(options_cache).value())](...) ->
                  typename Density::type { return value; }),
              blocks::initialize<typename Youngs::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<Youngs>(options_cache).value())](...) ->
                  typename Youngs::type { return value; }),
              blocks::initialize<typename ShearModulus::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<ShearModulus>(options_cache).value())](
                      ...) -> typename ShearModulus::type { return value; }));
        }
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
