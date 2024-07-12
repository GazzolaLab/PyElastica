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

    namespace detail {

      struct CosseratRodOptionGroup {
        // static constexpr Options::String help = {
        //     "Options for initializing Cosserat rods"};
      };

    }  // namespace detail

    class StraightCosseratRodInitializer
        : public ComposeInitializers<
              Initializers::CosseratRod,                                     //
              Initializers::Geometry::StraightRod,                           //
              Initializers::Elasticity::LinearHyperElasticModelWithDamping,  //
              Initializers::Kinematics::DefaultRodKinematics> {
     public:
      using P = ComposeInitializers<
          Initializers::CosseratRod,                                     //
          Initializers::Geometry::StraightRod,                           //
          Initializers::Elasticity::LinearHyperElasticModelWithDamping,  //
          Initializers::Kinematics::DefaultRodKinematics>;
      using P::P;

     public:
      using options = RequiredParameters;
    };

    namespace detail {

      struct straight_rod {
        using type = StraightCosseratRodInitializer;
        // static constexpr Options::String help = {"Initialize a straight rod"};
        using group = detail::CosseratRodOptionGroup;
      };

    }  // namespace detail

    class StraightCosseratRodWithoutDampingInitializer
        : public ComposeInitializers<
              Initializers::CosseratRod,                          //
              Initializers::Geometry::StraightRod,                //
              Initializers::Elasticity::LinearHyperElasticModel,  //
              Initializers::Kinematics::DefaultRodKinematics> {
     public:
      using P = ComposeInitializers<
          Initializers::CosseratRod,                          //
          Initializers::Geometry::StraightRod,                //
          Initializers::Elasticity::LinearHyperElasticModel,  //
          Initializers::Kinematics::DefaultRodKinematics>;
      using P::P;

     public:
      using options = RequiredParameters;
    };

    namespace detail {

      struct straight_rod_without_damping {
        using type = StraightCosseratRodWithoutDampingInitializer;
        // static constexpr Options::String help = {
        //     "Initialize a straight rod without damping"};
        using group = detail::CosseratRodOptionGroup;
      };

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica
