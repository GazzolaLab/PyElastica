#pragma once

#include <string>

#include "Systems/CosseratRods/Tags.hpp"
#include "Utilities/DefineTypes.h"  // for real_t
#include "Utilities/TMPL.hpp"
///
#include "Systems/CosseratRods/Initializers/ComposeInitializers.hpp"
#include "Systems/CosseratRods/Initializers/Initializers.hpp"

namespace elastica {

  namespace cosserat_rod {

    class StraightCosseratRodInitializer
        : public ComposeInitializers<
              Initializers::CosseratRod,                                     //
              Initializers::Geometry::StraightRod,                           //
              Initializers::Elasticity::LinearHyperElasticModel,  //
              Initializers::Kinematics::DefaultRodKinematics> {
     public:
      using P = ComposeInitializers<
          Initializers::CosseratRod,                                     //
          Initializers::Geometry::StraightRod,                           //
          Initializers::Elasticity::LinearHyperElasticModel,  //
          Initializers::Kinematics::DefaultRodKinematics>;
      using P::P;

     public:
      using options = RequiredParameters;
    };

    // namespace detail {

    //   struct CosseratRodOptionGroup {
    //     // static constexpr Options::String help = {
    //     //     "Options for initializing Cosserat rods"};
    //   };

    //   struct straight_rod {
    //     using type = StraightCosseratRodInitializer;
    //     // static constexpr Options::String help = {"Initialize a straight rod"};
    //     using group = detail::CosseratRodOptionGroup;
    //   };

    // }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
