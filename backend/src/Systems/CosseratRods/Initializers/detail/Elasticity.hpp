#pragma once

#include <stdexcept>
#include <tuple>

// #include "Options/Options.hpp"
// #include "Options/Parameter.hpp"
///
#include "Simulator/Materials.hpp"
///
#include "Systems/CosseratRods/Tags.hpp"
///
#include "Utilities/DefineTypes.h"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/TMPL.hpp"
///
#include "Systems/CosseratRods/Initializers/detail/Common.hpp"
#include "Systems/CosseratRods/Initializers/detail/Transformations.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      namespace option_tags {
        // can be used for torque damping as well
        // struct ForceDampingRate : Options::Parameter<real_t> {
        //   using P = Options::Parameter<real_t>;
        //   using P::P;
        //   static constexpr Options::String help = {
        //       "Coefficient to damp forces in proportion to velocity"};
        struct ForceDampingRate {
          using type = real_t;
          static type lower_bound() noexcept { return 0.0; }
          using TagType = tags::ForceDampingRate;
        };

        // can be used for torque damping as well
        // struct TorqueDampingRate : Options::Parameter<real_t> {
        //   using P = Options::Parameter<real_t>;
        //   using P::P;
        //   static constexpr Options::String help = {
        //       "Coefficient to damp torques in proportion to angular velocity"};
        struct TorqueDampingRate {
          using type = real_t;
          static type lower_bound() noexcept { return 0.0; }
          using TagType = tags::TorqueDampingRate;
        };

      }  // namespace option_tags

      class LinearHyperElasticModel {
       public:
        // the other required parameter comes from common
        using RequiredParameters = tmpl::list<>;
        using DefaultParameters = tmpl::list<>;

        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& options_cache) const /*noexcept*/ {
          const auto material_id =
              std::get<option_tags::Material>(options_cache).value();
          const real_t E = Material::get_youngs_modulus(material_id);
          const real_t G = Material::get_shear_modulus(material_id);
          return std::make_tuple(
              blocks::initialize<tags::BendingTwistRigidityMatrix>(
                  [E, G](std::size_t /* meta index */) {
                    return Vec3{E, E, G};
                  }),
              blocks::initialize<tags::ShearStretchRigidityMatrix>(
                  [E, G](std::size_t /* meta index */) {
                    return Vec3{G, G, E};
                  }));
        }
      };

      class LinearHyperElasticModelWithDamping
          : public LinearHyperElasticModel {
       private:
        using P = LinearHyperElasticModel;

       public:
        using ForceDampingRate = option_tags::ForceDampingRate;
        using TorqueDampingRate = option_tags::TorqueDampingRate;
        using ForceDampingCoefficient [[deprecated(R"error(
ForceDampingCoefficient is deprecated and will be removed in a future release.
Use ForceDampingRate instead.
)error")]] = ForceDampingRate;
        using TorqueDampingCoefficient [[deprecated(R"error(
TorqueDampingCoefficient is deprecated and will be removed in a future release.
Use TorqueDampingRate instead.
)error")]] = TorqueDampingRate;

        // parent has no required parameters
        using RequiredParameters =
            tmpl::list<ForceDampingRate, TorqueDampingRate>;
        using DefaultParameters = tmpl::list<>;

       protected:
        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& options_cache) const /*noexcept*/ {
          tmpl::list<ForceDampingRate, TorqueDampingRate> tags{};
          try {
            check_lower_bounds(options_cache, tags);
          } catch (std::domain_error& exception) {
            std::string context_msg = std::string(
                R"error(
If your intention is to use a rod without damping and you are currently using
`elastica::cosserat_rod::CosseratRod`, consider the undamped alternative
`elastica::cosserat_rod::CosseratRodWithoutDamping`.
)error");
            throw std::domain_error(exception.what() + context_msg);
          }

          return std::tuple_cat(
              // for things that don't need to be processed, pass as wrapped
              P::get_required_parameters(options_cache),
              pass_as_wrapped(options_cache, tags));
        }
      };

    }  // namespace detail

    struct ElasticityInitializers {
      using LinearHyperElasticModel = detail::LinearHyperElasticModel;
      using LinearHyperElasticModelWithDamping =
          detail::LinearHyperElasticModelWithDamping;
    };

  }  // namespace cosserat_rod

}  // namespace elastica
