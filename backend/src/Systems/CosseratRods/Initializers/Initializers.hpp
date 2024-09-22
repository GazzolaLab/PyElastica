#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/CosseratRods/Initializers/detail/Common.hpp"
#include "Systems/CosseratRods/Initializers/detail/Elasticity.hpp"
#include "Systems/CosseratRods/Initializers/detail/Geometry.hpp"
#include "Systems/CosseratRods/Initializers/detail/Kinematics.hpp"

namespace elastica {

  namespace cosserat_rod {

    struct Initializers {
      using CosseratRod = detail::CommonInitializerToAllCosseratRods;
      using Geometry = GeometryInitializers;
      using Elasticity = ElasticityInitializers;
      using Kinematics = KinematicsInitializer;
    };

  }  // namespace cosserat_rod

}  // namespace elastica
