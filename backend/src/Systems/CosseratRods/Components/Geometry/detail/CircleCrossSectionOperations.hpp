#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cmath>
#include <utility>

#include "Systems/CosseratRods/Components/Geometry/Protocols.hpp"
#include "Systems/CosseratRods/Components/Geometry/Types.hpp"
///
#include "Systems/CosseratRods/Components/Geometry/detail/CosseratRodCrossSectionInterface.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
//
#include "Utilities/ProtocolHelpers.hpp"
// Implementation details
#include "Utilities/Math/Sqrt.hpp"
#include "Utilities/Math/Square.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Operations corresponding to a circular cross section
         * \ingroup cosserat_rod_component
         *
         * \details
         * CircleCrossSectionOperations contains functions for characterizing a
         * circular cross section. Its intended use is as the third template
         * parameter in CosseratRodCrossSectionInterface, which is then
         * eventually used within the Blocks framework for a Cosserat rod
         * data-structure.
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see CosseratRodCrossSectionInterface, WithCircularCosseratRod
         */
        //**********************************************************************
        template <typename CRT>
        class CircleCrossSectionOperations {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! Real number type
          using real_type = typename Traits::real_type;
          //********************************************************************

         public:
          //**Geometry1D methods************************************************
          /*!\name Geometry1D methods*/
          //@{

          //********************************************************************
          /*!\brief computes cross section area
           *
           * \param radius radius of the circle
           */
          template <typename Radius>
          static inline decltype(auto) compute_cross_section_area(
              Radius const& radius) COSSERATROD_LIB_NOEXCEPT {
            return static_cast<real_type>(M_PI) * sq(radius);
          }
          //********************************************************************

          //********************************************************************
          /*!\brief computes second moment of area I_1
           *
           * \param radius radius of the circle
           */
          template <typename Radius>
          static inline decltype(auto) compute_second_moment_of_area(
              Radius const& radius) COSSERATROD_LIB_NOEXCEPT {
            return static_cast<real_type>(0.25 * M_1_PI) *
                   sq(compute_cross_section_area(radius));
          }
          //********************************************************************

          //********************************************************************
          /*!\brief Updates dimension (aka radius) for given volume and length
           *
           * \param volume Volume of the rod
           * \param length Length of the rod
           */
          template <typename Volume, typename Length>
          static inline decltype(auto)
          update_dimension(/* TODO rename: compute_dimension_conserve_volume */
                           Volume const& volume,
                           Length const& length) COSSERATROD_LIB_NOEXCEPT {
            constexpr auto inv_pi = static_cast<real_type>(M_1_PI);
            // radius[k] = np.sqrt(volume[k] / lengths[k] / np.pi)
            return sqrt(inv_pi * volume / length);
          }
          //********************************************************************

          //********************************************************************
          /*!\brief Returns the (real) shape factor for a shape, derived from
           * theory of elasticity
           */
          static inline constexpr auto shape_factor() COSSERATROD_LIB_NOEXCEPT
              -> real_type {
            // Value taken based on best correlation (A from fig. 2):
            // \alpha = \frac{6 + 12 \sigma + 6 \sigma^2}{7 + 12 \sigma + 4
            // \sigma^2} evaluated at Poisson ratio (\sigma) = 0.5, from "On
            // Timoshenko's correction for shear in vibrating beams" by Kaneko,
            // 1975
            return static_cast<real_type>(13.5) / static_cast<real_type>(14.);
          }
          //********************************************************************

          //@}
          //********************************************************************
        };

      }  // namespace detail

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica
