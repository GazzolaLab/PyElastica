#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/Kinematics/Tags.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //////////////////////////////////////////////////////////////////////////
      //
      // Forward declarations of kinematics types in the interface
      //
      //////////////////////////////////////////////////////////////////////////

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      // Forward declare all Layers and policies
      template <typename /* Cosserat Rod Traits */, typename /* Block */>
      class WithRodKinematics;
      /*! \endcond */
      //************************************************************************

    }  // namespace component

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_dilatation_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_curvature_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_shear_stretch_strain_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock>
    void update_acceleration(
        component::WithRodKinematics<Traits, ComputationalBlock>& block_like)
        COSSERATROD_LIB_NOEXCEPT;
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica
