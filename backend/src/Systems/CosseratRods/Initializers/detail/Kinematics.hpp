#pragma once

#include <cstddef>

///
#include "Simulator/Materials.hpp"
///
#include "Systems/CosseratRods/Tags.hpp"
///
#include "Utilities/DefineTypes.h"
#include "Utilities/TMPL.hpp"
///
#include "Systems/CosseratRods/Initializers/detail/Common.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      namespace {

        struct VelocityDefaultParameter {
          using TagType = tags::Velocity;
          static auto initialize(...) -> real_t { return 0.0; }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

        struct AngularVelocityDefaultParameter {
          using TagType = tags::AngularVelocity;
          static auto initialize(...) -> real_t { return 0.0; }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

        struct AccelerationDefaultParameter {
          using TagType = tags::Acceleration;
          static auto initialize(...) -> real_t { return 0.0; }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

        struct AngularAccelerationDefaultParameter {
          using TagType = tags::AngularAcceleration;
          static auto initialize(...) -> real_t { return 0.0; }
          using type = decltype(blocks::initialize<TagType>(initialize));
          type data = blocks::initialize<TagType>(initialize);
        };

      }  // namespace

      class DefaultRodKinematics {
       public:
        using RequiredParameters = tmpl::list<>;
        using DefaultParameters =
            tmpl::list<VelocityDefaultParameter, AccelerationDefaultParameter,
                       AngularVelocityDefaultParameter,
                       AngularAccelerationDefaultParameter>;

       protected:
        template <typename OptionsTuple>
        decltype(auto) get_required_parameters(
            OptionsTuple const& /*meta*/) const /*noexcept*/ {
          return std::make_tuple();
        }
      };

    }  // namespace detail

    struct KinematicsInitializer {
      using DefaultRodKinematics = detail::DefaultRodKinematics;
    };

  }  // namespace cosserat_rod

}  // namespace elastica
