#pragma once

#include "system.h"

namespace elasticapp {

// CosseratRod-specific variable tags
// These variable types are now made internal to this translation unit
namespace system::cosserat_rod {
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
        inline static MatrixType ghost_value = MatrixType::Constant(1, 1, 1.0);
    };
    struct InternalForces : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "internal_forces";
    };
    struct ExternalForces : Placement::OnNode, DataType::Vector {
        static constexpr std::string_view name = "external_forces";
    };

    // Element variables
    struct Director : Placement::OnElement, DataType::Matrix {
        static constexpr std::string_view name = "director";
    };
    struct Omega : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "omega";
    };
    struct Alpha : Placement::OnElement, DataType::Vector {
        static constexpr std::string_view name = "alpha";
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
    system::cosserat_rod::Position,
    system::cosserat_rod::Velocity,
    system::cosserat_rod::Acceleration,
    system::cosserat_rod::Mass,
    system::cosserat_rod::InternalForces,
    system::cosserat_rod::ExternalForces,

    // Element variables
    system::cosserat_rod::Omega,
    system::cosserat_rod::Alpha,
    system::cosserat_rod::Director,
    system::cosserat_rod::RestLengths,
    system::cosserat_rod::Density,
    system::cosserat_rod::Volume,
    system::cosserat_rod::MassSecondMomentOfInertia,
    system::cosserat_rod::InvMassSecondMomentOfInertia,
    system::cosserat_rod::InternalTorques,
    system::cosserat_rod::ExternalTorques,
    system::cosserat_rod::Lengths,
    system::cosserat_rod::Tangents,
    system::cosserat_rod::Radius,
    system::cosserat_rod::Dilatation,
    system::cosserat_rod::DilatationRate,
    system::cosserat_rod::Sigma,
    system::cosserat_rod::RestSigma,
    system::cosserat_rod::InternalStress,
    system::cosserat_rod::ShearMatrix,

    // Voronoi variables
    system::cosserat_rod::RestVoronoiLengths,
    system::cosserat_rod::VoronoiDilatation,
    system::cosserat_rod::Kappa,
    system::cosserat_rod::RestKappa,
    system::cosserat_rod::InternalCouple,
    system::cosserat_rod::BendMatrix
>;

} // namespace elasticapp
