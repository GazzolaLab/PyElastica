#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/Tags.hpp"
#include "Systems/common/Tags.hpp"

namespace elastica {

  namespace tags {

    //==========================================================================
    //
    //  TAG DEFINITIONS
    //
    //==========================================================================

    // 1D Geometry types
    /*
    struct ReferenceElementLength;
    struct ReferenceCurvature;
    struct ReferenceVoronoiLength;
    struct ElementLength;
    struct ElementDilatation;
    struct VoronoiLength;
    struct VoronoiDilatation;
    struct Curvature;
    struct Tangent;
    struct ShearStretchStrain;
    */

    // 2D Geometry
    /*
    struct Dimension;
    struct Volume;

    // Elasticity
    struct InternalLoads;
    struct InternalTorques;

    // Linear hyper-elasticity
    struct BendingTwistRigidityMatrix;
    struct ShearStretchRigidityMatrixCollectionTag;

    // Linear visco-elasticity
    struct ForceDampingCoefficient;
    struct TorqueDampingCoefficient;
    */

  }  // namespace tags

}  // namespace elastica
