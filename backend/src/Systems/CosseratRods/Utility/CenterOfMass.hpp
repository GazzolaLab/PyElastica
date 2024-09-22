#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block.hpp"
#include "Systems/CosseratRods/BlockSlice.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
#include "Systems/CosseratRods/Tags.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "blaze/Blaze.h"
#include "blaze_tensor/Blaze.h"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Helper for computing center of mass
      // \ingroup cosserat_rod
      */
      template <typename CRT>
      struct CenterOfMassComputation {
       public:
        using Traits = CRT;
        using real_type = typename Traits::real_type;

       public:
        //**********************************************************************
        /*!\brief Computes center of mass position of the body.
        //
        // \details
        // Computes center of mass position of the body given nodal masses and
        // positions. \n
        // COM_position = sum(body.mass_batch{i} * body.position_batch{i})
        // / sum(body.mass_batch{i})
        //
        // \example
        // The following shows a typical use of the
        // compute_position_center_of_mass() function with the expected
        // (correct) result also shown.
        // \snippet test_gov_eqns.cpp compute_position_center_of_mass_example
        //
        // \param[in] body object whose center of mass position is to be
        computed
        //
        // \return COM_position expression type
        //
        // \see fill later?
        */
        template <typename MassType, typename DimensionType,
                  typename PositionType>
        static inline decltype(auto) compute_position_center_of_mass(
            MassType const& mass_batch, DimensionType const& dimension_batch,
            PositionType const& position_batch) {
          // TODO: Check math, and move blaze operation into trait
          auto mass_times_position_batch =
              Traits::Operations::expand_for_broadcast(mass_batch) %
              position_batch;

          // TODO : refactor to not use blaze
          return Traits::Operations::transpose(
              blaze::sum<blaze::rowwise>(mass_times_position_batch) /
              blaze::sum(mass_batch));
        }
        //**********************************************************************

        //**********************************************************************
        /*!\brief Computes center of mass velocity of the body.
        //
        // \details
        // Computes center of mass velocity of the body given nodal masses and
        // velocities. \n
        // COM_velocity = sum(body.mass_batch{i} * body.velocity_batch{i})
        // / sum(body.mass_batch{i})
        //
        // \example
        // The following shows a typical use of the
        // compute_velocity_center_of_mass() function with the expected
        (correct)
        // result also shown.
        // \snippet test_gov_eqns.cpp compute_velocity_center_of_mass_example
        //
        // \param[in] body object whose center of mass velocity is to be
        computed
        //
        // \return COM_velocity expression type
        //
        // \see fill later?
        */
        template <typename MassType, typename DimensionType,
                  typename VelocityType>
        auto compute_velocity_center_of_mass(
            MassType const& mass_batch, DimensionType const& dimension_batch,
            VelocityType const& velocity_batch) {
          // TODO: Check math, and move blaze operation into trait
          auto mass_times_velocity_batch =
              Traits::Operations::expand_for_broadcast(mass_batch,
                                                       dimension_batch) %
              velocity_batch;
          // TODO : refactor to not use blaze
          return blaze::trans(
              blaze::sum<blaze::rowwise>(mass_times_velocity_batch) /
              blaze::sum(mass_batch));
        }
        //**********************************************************************
      };
      /*! \endcond */
      //************************************************************************
    }  // namespace detail

  }  // namespace cosserat_rod

  //****************************************************************************
  /*!\brief Computes the center of mass of a CosseratRod
   * \ingroup cosserat_rod
   *
   * \usage
   * \code
     Vec3 com = ::elastica::compute_center_of_mass(rod);
   * \endcode
   *
   * \param cosserat_rod The cosserat rod whose com is to be calculated
   */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_center_of_mass(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const&
          cosserat_rod) noexcept -> Vec3 {
    // TODO : Shift the logic here directly maybe?
    return cosserat_rod::detail::CenterOfMassComputation<CRT>::
        compute_position_center_of_mass(
            ::blocks::get<elastica::tags::Mass>(cosserat_rod),
            ::blocks::get<elastica::tags::ElementDimension>(cosserat_rod),
            ::blocks::get<elastica::tags::Position>(cosserat_rod));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Computes velocity of the center of mass of a CosseratRod
   * \ingroup cosserat_rod
   *
   * \usage
   * \code
     Vec3 com = ::elastica::compute_center_of_mass_velocity(rod);
   * \endcode
   *
   * \param cosserat_rod The cosserat rod whose com is to be calculated
   */
  template <typename CRT,
            template <typename /*CRT*/, typename /*InitializedBlock*/>
            class... Components>
  auto compute_center_of_mass_velocity(
      ::blocks::BlockSlice<::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ::blocks::BlockSlice, Components...>> const&
          cosserat_rod) noexcept -> Vec3 {
    // TODO : Shift the logic here directly maybe?
    return cosserat_rod::detail::CenterOfMassComputation<CRT>::
        compute_velocity_center_of_mass(
            ::blocks::get<elastica::tags::Mass>(cosserat_rod),
            ::blocks::get<elastica::tags::ElementDimension>(cosserat_rod),
            ::blocks::get<elastica::tags::Velocity>(cosserat_rod));
  }
  //****************************************************************************

}  // namespace elastica
