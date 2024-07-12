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

    template <typename Pos, typename Dir>
    class UserDefinedCosseratRodInitializer
        : public ComposeInitializers<
              Initializers::CosseratRod,                                     //
              Initializers::Geometry::UserDefinedRod<Pos, Dir>,              //
              Initializers::Elasticity::LinearHyperElasticModelWithDamping,  //
              Initializers::Kinematics::DefaultRodKinematics> {
     public:
      using P = ComposeInitializers<
          Initializers::CosseratRod,                                     //
          Initializers::Geometry::UserDefinedRod<Pos, Dir>,              //
          Initializers::Elasticity::LinearHyperElasticModelWithDamping,  //
          Initializers::Kinematics::DefaultRodKinematics>;
      using P::P;

     public:
      using options = typename P::RequiredParameters;
    };

    template <typename Pos, typename Dir>
    class UserDefinedCosseratRodWithoutDampingInitializer
        : public ComposeInitializers<
              Initializers::CosseratRod,                          //
              Initializers::Geometry::UserDefinedRod<Pos, Dir>,   //
              Initializers::Elasticity::LinearHyperElasticModel,  //
              Initializers::Kinematics::DefaultRodKinematics> {
     public:
      using P = ComposeInitializers<
          Initializers::CosseratRod,                          //
          Initializers::Geometry::UserDefinedRod<Pos, Dir>,   //
          Initializers::Elasticity::LinearHyperElasticModel,  //
          Initializers::Kinematics::DefaultRodKinematics>;
      using P::P;

     public:
      using options = typename P::RequiredParameters;
    };

  }  // namespace cosserat_rod

}  // namespace elastica
