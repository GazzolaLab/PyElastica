#pragma once

#include "system.h"

namespace elasticapp {

// CosseratRod-specific variable tags
// These variable types are now made internal to this translation unit
namespace {
    // Node variables
    struct Position : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "position";
    };
    struct Velocity : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "velocity";
    };
    struct Acceleration : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "acceleration";
    };
    struct Mass : Placement::OnNode, DataType::Scalar {
        static constexpr std::string_view name = "mass";
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // For dynamic matrices, Constant requires dimensions: Constant(rows, cols, value)
        inline static MatrixType ghost_value = MatrixType::Constant(1, 1, 1.0);
    };
    struct InternalForces : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "internal_forces";
    };
    struct ExternalForces : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "external_forces";
    };

    // Element variables
    struct Omega : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "omega";
    };
    struct Alpha : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "alpha";
    };
    struct Director : Placement::OnElement, DataType::Matrix {
        static constexpr std::string_view name = "director";
    };
    struct RestLengths : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "rest_lengths";
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // For dynamic matrices, Constant requires dimensions: Constant(rows, cols, value)
        inline static MatrixType ghost_value = MatrixType::Constant(1, 1, 1.0);
    };
    struct Density : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "density";
    };
    struct Volume : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "volume";
    };
    struct MassSecondMomentOfInertia : Placement::OnElement, DataType::Matrix {
        static constexpr std::string_view name = "mass_second_moment_of_inertia";
    };
    struct InvMassSecondMomentOfInertia : Placement::OnElement, DataType::Matrix {
        static constexpr std::string_view name = "inv_mass_second_moment_of_inertia";
    };
    struct InternalTorques : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "internal_torques";
    };
    struct ExternalTorques : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "external_torques";
    };
    struct Lengths : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "lengths";
    };
    struct Tangents : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "tangents";
    };
    struct Radius : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "radius";
    };
    struct Dilatation : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "dilatation";
    };
    struct DilatationRate : Placement::OnElement, DataType::Scalar {
        static constexpr std::string_view name = "dilatation_rate";
    };
    struct Sigma : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "sigma";
    };
    struct RestSigma : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "rest_sigma";
    };
    struct InternalStress : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "internal_stress";
    };
    struct ShearMatrix : Placement::OnElement, DataType::Matrix {
        static constexpr std::string_view name = "shear_matrix";
    };

    // Voronoi variables
    struct RestVoronoiLengths : Placement::OnVoronoi, DataType::Scalar {
        static constexpr std::string_view name = "rest_voronoi_lengths";
        // Note: Not constexpr because Eigen dynamic matrices don't have constexpr constructors
        // For dynamic matrices, Constant requires dimensions: Constant(rows, cols, value)
        inline static MatrixType ghost_value = MatrixType::Constant(1, 1, 1.0);
    };
    struct VoronoiDilatation : Placement::OnVoronoi, DataType::Scalar {
        static constexpr std::string_view name = "voronoi_dilatation";
    };
    struct Kappa : Placement::OnVoronoi, DataType::Vector {
        static constexpr std::string_view name = "kappa";
    };
    struct RestKappa : Placement::OnVoronoi, DataType::Vector {
        static constexpr std::string_view name = "rest_kappa";
    };
    struct InternalCouple : Placement::OnVoronoi, DataType::Vector {
        static constexpr std::string_view name = "internal_couple";
    };
    struct BendMatrix : Placement::OnVoronoi, DataType::Matrix {
        static constexpr std::string_view name = "bend_matrix";
    };
}

// CosseratRodSystem is a System with all variables from CosseratRod
// Variables are organized by placement (Node, Element, Voronoi)
using CosseratRodSystem = System<
    // Node variables
    Position,
    Velocity,
    Acceleration,
    Mass,
    InternalForces,
    ExternalForces,

    // Element variables
    Omega,
    Alpha,
    Director,
    RestLengths,
    Density,
    Volume,
    MassSecondMomentOfInertia,
    InvMassSecondMomentOfInertia,
    InternalTorques,
    ExternalTorques,
    Lengths,
    Tangents,
    Radius,
    Dilatation,
    DilatationRate,
    Sigma,
    RestSigma,
    InternalStress,
    ShearMatrix,

    // Voronoi variables
    RestVoronoiLengths,
    VoronoiDilatation,
    Kappa,
    RestKappa,
    InternalCouple,
    BendMatrix
>;

} // namespace elasticapp
