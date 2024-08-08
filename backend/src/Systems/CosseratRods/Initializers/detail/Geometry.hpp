#pragma once

#include <tuple>

// #include "Options/CustomTypes.hpp"  // Vec3 support
// #include "Options/Options.hpp"
// #include "Options/Parameter.hpp"
///
#include "Systems/common/Warnings/UserWarnings.hpp"
///
#include "Systems/CosseratRods/Initializers/Parameter.hpp"
#include "Systems/CosseratRods/Tags.hpp"
///
#include "Utilities/DefineTypes.h"
#include "Utilities/Generators.hpp"
#include "Utilities/Math/Normalize.hpp"
#include "Utilities/Math/Orthogonal.hpp"
#include "Utilities/Math/Rot3.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/Math/Zero.hpp"
#include "Utilities/TMPL.hpp"
///
#include "Systems/CosseratRods/Initializers/detail/Common.hpp"
#include "Systems/CosseratRods/Initializers/detail/Transformations.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      namespace option_tags {

        struct Radius : public elastica::Parameter<elastica::real_t> {
          using elastica::Parameter<elastica::real_t>::Parameter;
          using type = elastica::real_t;
          static type lower_bound() noexcept { return 0.0; }
          using TagType = ::elastica::tags::ElementDimension;
          // static constexpr Options::String help = {
          //     "Constant radius of the cosserat rod"};
        };

        struct Length : public elastica::Parameter<elastica::real_t> {
          using elastica::Parameter<elastica::real_t>::Parameter;
          using type = elastica::real_t;
          static type lower_bound() noexcept { return 0.0; }
          // static constexpr Options::String help = {
          //     "Length of the cosserat rod"};
        };

        struct Origin : public elastica::Parameter<Vec3> {
          using elastica::Parameter<Vec3>::Parameter;
          using type = Vec3;
          // static constexpr Options::String help = {
          //     "Position of first node of the cosserat rod"};
        };

        struct Direction : public elastica::Parameter<Vec3> {
          using elastica::Parameter<Vec3>::Parameter;
          using type = Vec3;
          // static constexpr Options::String help = {
          //     "Direction from start to end of the cosserat rod"};
        };

        struct Normal : public elastica::Parameter<Vec3> {
          using elastica::Parameter<Vec3>::Parameter;
          using type = Vec3;
          // static constexpr Options::String help = {
          //     "Common normal of all elements of the cosserat rod"};
        };

      }  // namespace option_tags

      namespace {

        struct ReferenceCurvatureDefaultParameter {
          using TagType = tags::ReferenceCurvature;
          static auto initialize(...) -> Vec3 { return zero3(); }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

        struct ReferenceShearStretchStrainDefaultParameter {
          using TagType = tags::ReferenceShearStretchStrain;
          static auto initialize(...) -> Vec3 { return zero3(); }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

        struct DilatationDefaultParameter {
          using TagType = tags::ElementDilatation;
          static auto initialize(...) -> real_t { return 1.0; }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

      }  // namespace

      class StraightRod {
       public:
        using Radius = option_tags::Radius;
        using Length = option_tags::Length;
        using Origin = option_tags::Origin;
        using Direction = option_tags::Direction;
        using Normal = option_tags::Normal;

        using RequiredParameters =
            tmpl::list<Radius, Length, Origin, Direction, Normal>;
        using DefaultParameters =
            tmpl::list<ReferenceCurvatureDefaultParameter,
                       ReferenceShearStretchStrainDefaultParameter,
                       DilatationDefaultParameter>;

       protected:
        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& options_cache) const /*noexcept*/ {
          // pass directly as references
          // return detail::pass_as_wrapped<tmpl::list<NElement,
          // Radius>>(data());

          check_lower_bounds(options_cache, tmpl::list<Radius, Length>{});

          auto zero_check = [&](auto v) {
            using Variable = tmpl::type_from<decltype(v)>;
            const Vec3 input = (std::get<Variable>(options_cache).value());
            if (::elastica::is_zero(input)) {
              throw std::logic_error(std::string(typeid(Variable).name()) +
                                     " is zero!");
            }
            return input;
          };
          const Vec3 direction =
              ::elastica::normalize(zero_check(tmpl::type_<Direction>{}));
          const Vec3 normal =
              ::elastica::normalize(zero_check(tmpl::type_<Normal>{}));

          // check if they are in the same direction
          if (::elastica::is_zero_length(direction % normal)) {
            throw std::logic_error(
                " Direction and normal are in the same direction!");
          }

          const Vec3 origin = std::get<Origin>(options_cache).value();
          const Vec3 rod_end{origin + std::get<Length>(options_cache).value() *
                                          direction};
          // common so fine to use
          const std::size_t n_elements =
              std::get<option_tags::NElement>(options_cache).value();

          // check if L > 2d
          systems_warning_if(
              [&]() -> bool {
                const auto per_element_length =
                    std::get<Length>(options_cache).value() / n_elements;
                const auto rad = std::get<Radius>(options_cache).value();
                return per_element_length < (2 * rad);
              },
              [&](auto& log) -> void {
                auto msg = R"error(
Element length of a cosserat rod is less than twice its radius. This is usually
a sign that there are more elements than required. Even with these increased
number of elements, our equations still converge, so you can still safely use
the rod with these parameters.

But note that in case you are using this rod within our CollisionSystem, you
will encounter spurious forces within the rod. In this case, consider reducing
the number of elements within a rod. The parameters that triggered this warning
are:
)error";

                log << msg
                    << "Radius : " << std::get<Radius>(options_cache).value()
                    << "\n"
                    << "Length : " << std::get<Length>(options_cache).value()
                    << "\n"
                    << "NElement : "
                    << std::get<option_tags::NElement>(options_cache).value();
              });

          return std::make_tuple(
              blocks::initialize<typename Radius::TagType>(
                  // do a copy here of tuple of initializers
                  [value = (std::get<Radius>(options_cache).value())](...) ->
                  typename Radius::type { return value; }),
              blocks::initialize<tags::Position>(
                  [lg = ::elastica::linspace_generator(
                       origin, rod_end,
                       // this is not ideal, as we take in information
                       // from a cosserat rod
                       n_elements + 1UL)](std::size_t index) -> Vec3 {
                    return lg(index);
                  }),
              blocks::initialize<tags::Director>(
                  [dir = ::elastica::
                       make_orthogonal_bases_from_normal_and_tangent(
                           normal, direction)](std::size_t) -> Rot3 {
                    return dir;
                  }));
        }
      };

    }  // namespace detail

    struct GeometryInitializers {
      using StraightRod = detail::StraightRod;
      // using StraightRodWithTwist = detail::StraightRodWithTwist;
      // template <typename P, typename D>
      // using UserDefinedRod = detail::UserDefinedRod<P, D>;
    };

  }  // namespace cosserat_rod

}  // namespace elastica
