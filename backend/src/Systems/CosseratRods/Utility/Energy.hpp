#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
#include "Systems/CosseratRods/Tags.hpp"
#include "Utilities/DefineTypes.h"
#include "blaze/Blaze.h"
#include "blaze_tensor/Blaze.h"

// TODO: Documentation need update
namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      //************************************************************************
      /*!\brief Helper for computing rod energy
      // \ingroup cosserat_rod
      */
      template <typename CRT>
      struct EnergyComputation {
       public:
        using Traits = CRT;
        using real_type = typename Traits::real_type;

       public:
        //**********************************************************************
        template <typename MassType, typename VelocityType>
        static auto compute_translational_energy(
            MassType const& mass_batch, VelocityType const& velocity_batch)
            -> real_type {
          return real_type(0.5) *
                 blaze::dot(mass_batch, blaze::sum<blaze::columnwise>(
                                            velocity_batch % velocity_batch));
        }
        //**********************************************************************

        //**********************************************************************
        template <typename OmegaType, typename DilatationType,
                  typename MSMoIType>
        static auto compute_rotational_energy(
            OmegaType const& omega_batch,
            DilatationType const& dilatation_batch,
            MSMoIType const& mass_second_moment_of_inertia_batch) -> real_type {
          std::size_t const n_elems =
              ::elastica::cosserat_rod::get_batch_elems(omega_batch);
          // TODO: Something is wrong here. why get_batch_elems is global?
          std::size_t const dimension = 3UL;  // to avoid linalg kernels fail
          // allocating temp container (3, cosserat_rod.n_elems) for temporary
          // storage
          // TODO refactor, assign variable name to dummy variable if it appears
          // frequently
          auto J_omega_batch =
              blaze::DynamicMatrix<real_type>(dimension, n_elems);
          Traits::Operations::batch_matvec(
              J_omega_batch, mass_second_moment_of_inertia_batch, omega_batch);
          auto J_omega_upon_e_batch =
              J_omega_batch %
              blaze::expand(blaze::trans(blaze::pow(dilatation_batch, -1)),
                            dimension);
          auto rotational_energy_batch =
              Traits::Operations::batch_dot(omega_batch, J_omega_upon_e_batch);
          return real_type(0.5) * blaze::sum(rotational_energy_batch);
        }
        //**********************************************************************

        //**********************************************************************
        template <typename StrainType, typename ShearMatrixType,
                  typename LengthType>
        static auto compute_shear_energy(
            StrainType const& strain, StrainType const& rest_strain,
            ShearMatrixType const& shear_matrix_batch,
            LengthType const& rest_length_batch) -> real_type {
          std::size_t const n_elems = ::elastica::cosserat_rod::get_batch_elems(
              strain);  // TODO: Something is wrong here. why get_batch_elems is
                        // global?
          // TODO : can this be constexpr?
          std::size_t const dimension = 3UL;  // to avoid linalg kernels fail
          auto strain_diff_batch = strain - rest_strain;
          // allocating temp container (3, n_elems) for temporary storage
          // TODO refactor to avoid temp memory allocations
          auto shear_internal_stress_batch =
              blaze::DynamicMatrix<real_type>(dimension, n_elems);
          Traits::Operations::batch_matvec(shear_internal_stress_batch,
                                           shear_matrix_batch,
                                           strain_diff_batch);
          auto shear_internal_stress_dot_strain_diff_batch =
              Traits::Operations::batch_dot(strain_diff_batch,
                                            shear_internal_stress_batch);
          return real_type(0.5) *
                 blaze::sum(shear_internal_stress_dot_strain_diff_batch *
                            rest_length_batch);
        }
        //**********************************************************************

        //**********************************************************************
        template <typename CurvatureType, typename BendMatrixType,
                  typename VoronoiLengthType>
        static auto compute_bend_energy(
            CurvatureType const& curvature_batch,
            CurvatureType const& rest_curvature_batch,
            BendMatrixType const& bend_matrix_batch,
            VoronoiLengthType const& rest_voronoi_length_batch) -> real_type {
          std::size_t const n_voronoi =
              ::elastica::cosserat_rod::get_batch_elems(
                  curvature_batch);  // TODO: Something is wrong here. why
                                     // get_batch_elems is global?
          std::size_t const dimension = 3UL;  // to avoid linalg kernels fail
          auto curvature_diff_batch = curvature_batch - rest_curvature_batch;
          // allocating temp container (3, n_elems) for temporary storage
          // TODO refactor to avoid temp memory allocations
          auto bend_internal_torque_batch =
              blaze::DynamicMatrix<real_type>(dimension, n_voronoi);
          Traits::Operations::batch_matvec(bend_internal_torque_batch,
                                           bend_matrix_batch,
                                           curvature_diff_batch);
          auto bend_internal_torque_dot_curvature_diff_batch =
              Traits::Operations::batch_dot(curvature_diff_batch,
                                            bend_internal_torque_batch);
          return real_type(0.5) *
                 blaze::sum(bend_internal_torque_dot_curvature_diff_batch *
                            rest_voronoi_length_batch);
        }
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace detail

  }  // namespace cosserat_rod

  //****************************************************************************
  /*!\brief Computes translational kinetic energy of the cosserat_rod.
  // \ingroup cosserat_rod
  //
  // \details
  // Computes translational kinetic energy of the cosserat_rod given nodal
  // masses and velocities.
  //
  // \example
  // The following shows a typical use of the compute_translational_energy()
  // function with the expected (correct) result also shown.
  // \snippet test_gov_eqns.cpp compute_translational_energy_example
  //
  // \param[in] cosserat_rod whose translational kinetic energy will be computed
  //
  // \return translational kinetic energy expression type
  //
  // \see fill later?
  */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_translational_energy(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const& cosserat_rod)
      -> ::elastica::real_t {
    return cosserat_rod::detail::EnergyComputation<CRT>::
        compute_translational_energy(
            ::blocks::get<::elastica::tags::Mass>(cosserat_rod),
            ::blocks::get<::elastica::tags::Velocity>(cosserat_rod));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Computes rotational kinetic energy of the cosserat_rod.
  // \ingroup cosserat_rod
  //
  // \details
  // Computes rotational kinetic energy of the cosserat_rod given elemental
  // mass moment of inertia and angular velocities.
  //
  // \example
  // The following shows a typical use of the compute_rotational_energy()
  // function with the expected (correct) result also shown.
  // \snippet test_gov_eqns.cpp compute_rotational_energy_example
  //
  // \param[in] cosserat_rod object whose rotational KE will be computed
  //
  // \return rotational kinetic energy expression type
  //
  // \see fill later?
  */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_rotational_energy(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const& cosserat_rod) {
    return cosserat_rod::detail::EnergyComputation<CRT>::
        compute_rotational_energy(
            ::blocks::get<::elastica::tags::AngularVelocity>(cosserat_rod),
            ::blocks::get<::elastica::tags::ElementDilatation>(cosserat_rod),
            ::blocks::get<::elastica::tags::MassSecondMomentOfInertia>(
                cosserat_rod));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Computes bending mode energy of the system
  // \ingroup cosserat_rod
  //
  // \details
  // Computes bending mode energy of the system given voronoidal
  // bend matrix and curvature.
  //
  // \example
  // The following shows a typical use of the compute_bend_energy()
  // function with the expected (correct) result also shown.
  // \snippet test_gov_eqns.cpp compute_bend_energy_example
  //
  // \param[in] cosserat_rod object whose bend mode energy will be computed
  //
  // \return bend mode energy expression type
  //
  // \see fill later?
  */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_bend_energy(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const& cosserat_rod) {
    return cosserat_rod::detail::EnergyComputation<CRT>::compute_bend_energy(
        ::blocks::get<::elastica::tags::Curvature>(cosserat_rod),
        ::blocks::get<::elastica::tags::ReferenceCurvature>(cosserat_rod),
        ::blocks::get<::elastica::tags::BendingTwistRigidityMatrix>(
            cosserat_rod),
        ::blocks::get<::elastica::tags::ReferenceVoronoiLength>(cosserat_rod));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Computes shearing mode energy of the cosserat_rod.
  // \ingroup cosserat_rod
  //
  // \details
  // Computes shearing mode energy of the cosserat_rod given elemental
  // shear matrix and shear strain.
  //
  // \example
  // The following shows a typical use of the compute_shear_energy()
  // function with the expected (correct) result also shown.
  // \snippet test_gov_eqns.cpp compute_shear_energy_example
  //
  // \param[in] cosserat_rod object whose shear mode energy will be computed
  //
  // \return shear mode energy expression type
  //
  // \see fill later?
  */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_shear_energy(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const& cosserat_rod) {
    return cosserat_rod::detail::EnergyComputation<CRT>::compute_shear_energy(
        ::blocks::get<::elastica::tags::ShearStretchStrain>(cosserat_rod),
        ::blocks::get<::elastica::tags::ReferenceShearStretchStrain>(
            cosserat_rod),
        ::blocks::get<::elastica::tags::ShearStretchRigidityMatrix>(
            cosserat_rod),
        ::blocks::get<::elastica::tags::ReferenceElementLength>(cosserat_rod));
  }
  //****************************************************************************

}  // namespace elastica
